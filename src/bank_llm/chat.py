"""End-to-end command-line chat for the bank customer-service RAG system.

Pipeline:
    question -> FAISS/BM25 Child retrieval -> BGE Child reranking
             -> parent_text expansion -> grounded prompt
             -> Qwen2.5-3B-Instruct + optional LoRA -> answer

The LoRA path may point to the final SFT adapter or to a DPO adapter trained by
continuing from the SFT adapter.  If LoRA has already been merged into the base
model, omit ``--adapter`` and pass the merged directory to ``--model``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from bank_llm.retrieval import HybridRetriever, RetrievalConfig, build_rag_prompt


SYSTEM_PROMPT = """你是一位专业的中文银行客服人员。你的回答必须以检索到的银行或监管资料为依据；资料不足时明确说明，不猜测、不编造。"""


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_generator(model_path: str, adapter_path: str | None) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def generate_answer(
    prompt: str,
    tokenizer: Any,
    model: Any,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered, return_tensors="pt")
    input_device = next(model.parameters()).device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    sampling = temperature > 0
    generation_args: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": sampling,
        "repetition_penalty": 1.05,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if sampling:
        generation_args.update(temperature=temperature, top_p=top_p)

    output_ids = model.generate(**inputs, **generation_args)
    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def print_sources(results: list[dict[str, Any]]) -> None:
    if not results:
        print("\n[Sources] No sufficiently relevant document was retrieved.")
        return

    print("\n[Sources]")
    for number, result in enumerate(results, start=1):
        chunk = result["chunk"]
        title = chunk.get("title", "Unknown source")
        score = result.get("rerank_score", 0.0)
        print(f"  {number}. {title} (rerank={score:.3f})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Base or merged model path/name.")
    parser.add_argument("--adapter", help="Optional final SFT/DPO LoRA adapter path.")
    parser.add_argument("--index", type=Path, required=True, help="FAISS index path.")
    parser.add_argument("--chunks", type=Path, required=True, help="chunks.pkl path.")
    parser.add_argument("--embedding-model", default="BAAI/bge-large-zh-v1.5")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-large")
    parser.add_argument("--retrieval-device", default=default_device())
    parser.add_argument("--vector-top-k", type=int, default=20)
    parser.add_argument("--bm25-top-k", type=int, default=20)
    parser.add_argument("--rerank-top-k", type=int, default=3)
    parser.add_argument("--rerank-threshold", type=float, default=0.1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--show-sources", action="store_true")
    parser.add_argument("--query", help="Ask one question and exit; otherwise use interactive mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retriever = HybridRetriever(
        RetrievalConfig(
            index_path=args.index,
            chunks_path=args.chunks,
            embedding_model=args.embedding_model,
            reranker_model=args.reranker_model,
            device=args.retrieval_device,
            vector_top_k=args.vector_top_k,
            bm25_top_k=args.bm25_top_k,
            rerank_top_k=args.rerank_top_k,
            rerank_threshold=args.rerank_threshold,
        )
    )
    tokenizer, model = load_generator(args.model, args.adapter)

    def answer(query: str) -> None:
        results = retriever.retrieve(query)
        prompt = build_rag_prompt(query, results)
        response = generate_answer(
            prompt,
            tokenizer,
            model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"\nAssistant: {response}")
        if args.show_sources:
            print_sources(results)

    if args.query:
        answer(args.query.strip())
        return

    print("Bank RAG chat is ready. Type exit, quit, or q to stop.")
    while True:
        try:
            query = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in {"exit", "quit", "q"}:
            break
        if query:
            answer(query)


if __name__ == "__main__":
    main()
