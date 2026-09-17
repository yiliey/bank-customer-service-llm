"""Hybrid child-chunk retrieval, BGE reranking, and parent expansion.

This module is a reusable version of the original ``recall_rerank.py``.  It
keeps the original retrieval behaviour while replacing AutoDL-specific paths
and the hard-coded CUDA device with explicit configuration. The FAISS index is
built offline from Child chunks. Online retrieval and reranking also operate on
Children; ``build_rag_prompt`` then expands selected Children to their stored
``parent_text`` for coherent generation context.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


@dataclass(frozen=True)
class RetrievalConfig:
    index_path: Path
    chunks_path: Path
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    reranker_model: str = "BAAI/bge-reranker-large"
    device: str = "cuda"
    vector_top_k: int = 20
    bm25_top_k: int = 20
    rerank_top_k: int = 3
    rerank_threshold: float = 0.1


class HybridRetriever:
    """Load the saved Child index and return reranked Child candidates."""

    def __init__(self, config: RetrievalConfig) -> None:
        self.config = config
        self._validate_paths()

        self.index = faiss.read_index(str(config.index_path))
        with config.chunks_path.open("rb") as file:
            self.chunks: list[dict[str, Any]] = pickle.load(file)

        self.embedding_model = SentenceTransformer(
            config.embedding_model,
            device=config.device,
        )
        if config.device.startswith("cuda"):
            self.embedding_model.half()

        self.reranker = CrossEncoder(config.reranker_model, device=config.device)
        tokenized_chunks = [list(jieba.cut(chunk["text"])) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks, k1=1.5, b=0.75)

    def _validate_paths(self) -> None:
        for path in (self.config.index_path, self.config.chunks_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required RAG asset does not exist: {path}")

    @staticmethod
    def _is_mostly_chinese(text: str, threshold: float = 0.5) -> bool:
        if not text:
            return False
        chinese_chars = sum("\u4e00" <= char <= "\u9fff" for char in text)
        return chinese_chars / len(text) >= threshold

    def _vector_retrieve(self, query: str) -> list[dict[str, Any]]:
        query_vector = self.embedding_model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)
        scores, indices = self.index.search(query_vector, k=self.config.vector_top_k)

        return [
            {
                "chunk": self.chunks[index],
                "chunk_idx": int(index),
                "vector_score": float(score),
            }
            for score, index in zip(scores[0], indices[0])
            if index >= 0
        ]

    def _bm25_retrieve(self, query: str) -> list[dict[str, Any]]:
        scores = self.bm25.get_scores(list(jieba.cut(query)))
        top_indices = np.argsort(scores)[::-1][: self.config.bm25_top_k]
        return [
            {
                "chunk": self.chunks[index],
                "chunk_idx": int(index),
                "bm25_score": float(scores[index]),
            }
            for index in top_indices
            if scores[index] > 0
        ]

    def _hybrid_retrieve(self, query: str) -> list[dict[str, Any]]:
        merged: dict[int, dict[str, Any]] = {}
        for result in self._vector_retrieve(query):
            merged[result["chunk_idx"]] = result

        for result in self._bm25_retrieve(query):
            index = result["chunk_idx"]
            if index in merged:
                merged[index]["bm25_score"] = result["bm25_score"]
            else:
                merged[index] = result

        return [
            candidate
            for candidate in merged.values()
            if self._is_mostly_chinese(candidate["chunk"]["text"])
        ]

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        candidates = self._hybrid_retrieve(query)
        if not candidates:
            return []

        pairs = [[query, item["chunk"]["text"]] for item in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda item: item[0], reverse=True)

        results: list[dict[str, Any]] = []
        for score, candidate in ranked[: self.config.rerank_top_k]:
            if float(score) >= self.config.rerank_threshold:
                candidate["rerank_score"] = float(score)
                results.append(candidate)
        return results


def build_rag_prompt(query: str, results: list[dict[str, Any]]) -> str:
    """Expand selected Children to Parents and build the grounded prompt."""
    if not results:
        return f"""用户问题：{query}

当前没有检索到足够相关的参考资料。请简要说明暂无相关资料，并建议用户以银行或监管机构的官方信息为准。回答不超过100字。"""

    references: list[str] = []
    for number, result in enumerate(results, start=1):
        chunk = result["chunk"]
        text = chunk.get("parent_text", chunk["text"])
        title = chunk.get("title", "未知来源")
        references.append(f"【参考资料{number}】来源：{title}\n{text}")

    context = "\n\n".join(references)
    return f"""用户问题：{query}

以下是检索到的参考资料：
{context}

请严格遵守以下要求：
1. 只使用参考资料中能够支持的信息，不得编造。
2. 如果资料不足以回答，明确回复“暂无相关资料”。
3. 使用“您”称呼用户，语气专业、亲切、自然。
4. 回答简洁，不超过150字。
5. 不要在答案中输出相关性分数。"""
