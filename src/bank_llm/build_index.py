"""Build the Parent–Child FAISS index used by the bank RAG pipeline.

Offline pipeline:
    PDF -> title-based Parent sections -> semantic Child chunks
        -> Child embeddings -> FAISS index + chunk metadata

Short Parents are indexed directly. Long Parents are split into Child chunks;
each Child stores ``parent_text`` so online retrieval can expand a reranked
Child back to its coherent Parent context.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pdfplumber
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


TITLE_PATTERNS = [
    r"^\d+[、.．]\s*.{2,40}$",
    r"^[一二三四五六七八九十]+[、.．]\s*.{2,40}$",
    r"^（[一二三四五六七八九十]+）\s*.{2,40}$",
    r"^\d+\.\d+\s*.{2,40}$",
    r"^第[一二三四五六七八九十\d]+[章节部分]\s*.{0,30}$",
]
TITLE_RE = re.compile("|".join(f"({pattern})" for pattern in TITLE_PATTERNS))


@dataclass(frozen=True)
class IndexConfig:
    pdf_dir: Path
    output_dir: Path
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    device: str = "cuda"
    batch_size: int = 64
    min_parent_length: int = 80
    max_parent_length: int = 800
    semantic_threshold: float = 0.85
    min_child_length: int = 60
    recursive: bool = False


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_title(line: str) -> bool:
    line = line.strip()
    return bool(line and len(line) <= 50 and TITLE_RE.match(line))


def clean_page_text(text: str) -> str:
    """Remove common PDF footers, URLs, and obvious extraction artifacts."""
    boilerplate = (
        "毕马威华振会计师事务所",
        "毕马威企业咨询",
        "合伙制事务所",
        "私营担保有限公司",
        "毕马威国际",
        "毕马威会计师事务所",
        "版权所有",
        "不得转载",
        "毕马威中国各办公室",
    )
    cleaned: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or any(term in line for term in boilerplate):
            continue
        if "kpmg.com" in line.lower() or line.startswith("©"):
            continue
        if re.search(r"https?://", line):
            continue

        if len(line) >= 6 and not re.match(r"^[\d,.\s%]+$", line):
            repeated = sum(left == right for left, right in zip(line, line[1:]))
            if repeated / len(line) > 0.3:
                continue

        chinese = sum("\u4e00" <= char <= "\u9fff" for char in line)
        ascii_letters = sum(char.isalpha() and char.isascii() for char in line)
        if len(line) > 3 and chinese and ascii_letters:
            ratio = ascii_letters / len(line)
            if 0.1 < ratio < 0.5 and len(line) < 20:
                continue
        cleaned.append(line)
    return "\n".join(cleaned)


def extract_pdf(pdf_path: Path) -> list[str]:
    pages: list[str] = []
    try:
        with pdfplumber.open(pdf_path) as document:
            for page in document.pages:
                text = page.extract_text()
                if text and len(text.strip()) > 10:
                    cleaned = clean_page_text(text)
                    if cleaned:
                        pages.append(cleaned)
    except Exception as error:
        print(f"[warning] Failed to parse {pdf_path.name}: {error}")
    return pages


def split_parents_by_title(pages: list[str], min_length: int) -> list[str]:
    lines = [
        line.strip()
        for page in pages
        for line in page.splitlines()
        if line.strip()
    ]
    parents: list[str] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        text = "\n".join(current).strip()
        if len(text) >= min_length:
            parents.append(text)
        elif text and parents:
            parents[-1] += "\n" + text
        current = []

    for line in lines:
        if is_title(line) and current:
            flush()
        current.append(line)
    flush()
    return parents


def fallback_page_groups(
    pages: list[str], max_length: int, min_length: int
) -> list[str]:
    groups: list[str] = []
    current = ""
    for page in pages:
        if not current or len(current) + len(page) <= max_length:
            current = f"{current}\n{page}".strip()
        else:
            if len(current) >= min_length:
                groups.append(current)
            current = page
    if len(current) >= min_length:
        groups.append(current)
    return groups


def _merge_short_segments(segments: list[str], min_length: int) -> list[str]:
    merged: list[str] = []
    pending = ""
    for segment in segments:
        pending += segment
        if len(pending) >= min_length:
            merged.append(pending)
            pending = ""
    if pending:
        if merged:
            merged[-1] += pending
        else:
            merged.append(pending)
    return merged


def semantic_split(
    text: str,
    model: SentenceTransformer,
    threshold: float,
    min_length: int,
) -> list[str]:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[。！？\n])", text)
        if len(sentence.strip()) > 5
    ]
    if len(sentences) <= 2:
        return [text]

    vectors = model.encode(sentences, normalize_embeddings=True, batch_size=32)
    boundaries = {
        index + 1
        for index in range(len(sentences) - 1)
        if float(np.dot(vectors[index], vectors[index + 1])) < threshold
    }

    raw_segments: list[str] = []
    current: list[str] = []
    for index, sentence in enumerate(sentences, start=1):
        current.append(sentence)
        if index in boundaries:
            raw_segments.append("".join(current))
            current = []
    if current:
        raw_segments.append("".join(current))
    return _merge_short_segments(raw_segments, min_length)


def collect_parent_chunks(config: IndexConfig) -> list[dict[str, Any]]:
    pattern = "**/*.pdf" if config.recursive else "*.pdf"
    pdf_files = sorted(config.pdf_dir.glob(pattern))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found under {config.pdf_dir}")

    parents: list[dict[str, Any]] = []
    for pdf_path in tqdm(pdf_files, desc="Parsing PDFs"):
        pages = extract_pdf(pdf_path)
        if not pages:
            continue
        sections = split_parents_by_title(pages, config.min_parent_length)
        if not sections:
            sections = fallback_page_groups(
                pages,
                config.max_parent_length,
                config.min_parent_length,
            )

        relative_path = str(pdf_path.relative_to(config.pdf_dir))
        for index, text in enumerate(sections):
            parents.append(
                {
                    "text": text,
                    "title": pdf_path.stem,
                    "source": relative_path,
                    "parent_id": f"{pdf_path.stem}_{index}",
                }
            )
    if not parents:
        raise RuntimeError("PDF files were found, but no valid Parent sections were extracted.")
    return parents


def create_retrieval_units(
    parents: list[dict[str, Any]],
    model: SentenceTransformer,
    config: IndexConfig,
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for parent in tqdm(parents, desc="Creating Child chunks"):
        if len(parent["text"]) <= config.max_parent_length:
            units.append({**parent, "is_child": False})
            continue

        children = semantic_split(
            parent["text"],
            model,
            config.semantic_threshold,
            config.min_child_length,
        )
        for child_index, child_text in enumerate(children):
            units.append(
                {
                    **parent,
                    "text": child_text,
                    "parent_text": parent["text"],
                    "is_child": True,
                    "child_id": child_index,
                }
            )
    return units


def build_index(config: IndexConfig) -> dict[str, Any]:
    if not config.pdf_dir.is_dir():
        raise NotADirectoryError(f"PDF directory does not exist: {config.pdf_dir}")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    parents = collect_parent_chunks(config)
    model = SentenceTransformer(config.embedding_model, device=config.device)
    if config.device.startswith("cuda"):
        model.half()
    units = create_retrieval_units(parents, model, config)

    texts = [unit["text"] for unit in units]
    vectors = model.encode(
        texts,
        batch_size=config.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    index_path = config.output_dir / "banking.index"
    chunks_path = config.output_dir / "chunks.pkl"
    stats_path = config.output_dir / "index_stats.json"
    faiss.write_index(index, str(index_path))
    with chunks_path.open("wb") as file:
        pickle.dump(units, file)

    stats = {
        "parent_count": len(parents),
        "retrieval_unit_count": len(units),
        "child_count": sum(unit["is_child"] for unit in units),
        "embedding_dimension": int(vectors.shape[1]),
        "embedding_model": config.embedding_model,
        "config": {
            **asdict(config),
            "pdf_dir": str(config.pdf_dir),
            "output_dir": str(config.output_dir),
        },
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-model", default="BAAI/bge-large-zh-v1.5")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-parent-length", type=int, default=80)
    parser.add_argument("--max-parent-length", type=int, default=800)
    parser.add_argument("--semantic-threshold", type=float, default=0.85)
    parser.add_argument("--min-child-length", type=int, default=60)
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_index(
        IndexConfig(
            pdf_dir=args.pdf_dir,
            output_dir=args.output_dir,
            embedding_model=args.embedding_model,
            device=args.device,
            batch_size=args.batch_size,
            min_parent_length=args.min_parent_length,
            max_parent_length=args.max_parent_length,
            semantic_threshold=args.semantic_threshold,
            min_child_length=args.min_child_length,
            recursive=args.recursive,
        )
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

