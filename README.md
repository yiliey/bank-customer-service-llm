# Bank Customer Service LLM Post-training + Hybrid RAG

A domain-adapted Chinese bank customer-service system built on
Qwen2.5-3B-Instruct. The project combines LoRA post-training with a hybrid
FAISS/BM25 retriever, BGE reranking, and Parent–Child context expansion.

This repository is a cleaned reconstruction of an experimental project. It
separates verified artifacts from historical components whose runtime outputs
were not retained.

## System architecture

### Model post-training

```text
Qwen2.5-3B-Instruct
  -> LoRA SFT on 3,999 domain examples
  -> LoRA DPO on 2,000 preference pairs
  -> final LoRA adapter or merged model
```

SFT teaches domain response patterns and service style. DPO continues training
the SFT LoRA adapter with preference data. The final adapter can be loaded on
the base model or merged once for deployment.

### Offline knowledge-base construction

```text
Bank and regulatory PDFs
  -> title-based Parent sections
  -> semantic Child chunks for long Parents
  -> BGE Child embeddings
  -> FAISS index + chunk metadata
```

Short Parents are indexed directly. Every indexed Child stores its
`parent_text`, so retrieval remains precise while generation receives coherent
context.

### Online inference

```text
Question
  -> FAISS dense recall + BM25 sparse recall
  -> merge and deduplicate Child candidates
  -> BGE cross-encoder reranking
  -> expand selected Children to parent_text
  -> grounded prompt
  -> Qwen2.5-3B + final LoRA
  -> answer
```

RAG is an external knowledge layer. It does not modify model parameters; it
adds retrieved evidence to the prompt used by the post-trained model.

## Repository status

| Component | Status | Evidence |
| --- | --- | --- |
| SFT data | Available in the private archive | 3,999 cleaned examples |
| LoRA SFT | Verified | Rank 16, alpha 32, LR `5e-5`, 6 epochs |
| SFT evaluation | Verified | Saved LLaMA-Factory prediction results |
| DPO data | Available in the private archive | 2,000 preference pairs |
| DPO training | Historical implementation | Final config, adapter, and logs were not retained |
| Final Rank-32 model | Referenced by code | Export path records Rank 32 / LR `1e-4`; weights are missing |
| Parent–Child indexing | Included | `build_index.py` |
| Hybrid retrieval | Included | `retrieval.py` |
| End-to-end chat | Included | `chat.py` |
| RAGAS evaluation | Recovered in the private archive | Script and 30 QA pairs exist; result CSV is missing |
| DataInf selection | Recovered in the private archive | Implementation exists; score tensors are missing |

## Verified SFT results

Evaluation used 535 cleaned test samples. The results below come directly from
saved LLaMA-Factory output files.

| Metric | Base Qwen2.5-3B | LoRA SFT | Absolute gain |
| --- | ---: | ---: | ---: |
| BLEU-4 | 6.12 | 27.61 | +21.48 |
| ROUGE-1 | 27.95 | 52.48 | +24.53 |
| ROUGE-2 | 5.66 | 24.74 | +19.08 |
| ROUGE-L | 17.81 | 39.04 | +21.24 |

No DPO reward or RAG precision/recall number is reported because the original
result files were not retained.

## Repository layout

```text
bank-customer-service-llm/
├── configs/
│   ├── dpo_reconstructed.yaml
│   └── sft_verified_rank16_lr5e-5.yaml
├── data/
│   ├── README.md
│   └── dataset_info.json
├── src/bank_llm/
│   ├── build_index.py
│   ├── chat.py
│   └── retrieval.py
├── .gitignore
└── requirements.txt
```

The full LLaMA-Factory source tree, model checkpoints, PDF corpus, FAISS index,
and generated datasets are intentionally not vendored in this repository.

Before training, place the SFT, test, and DPO JSON files described in
[`data/README.md`](data/README.md) under `data/`.

## Installation

Python 3.10+ and a CUDA environment are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

For LoRA SFT/DPO training, install a compatible LLaMA-Factory release
separately. The historical environment used LLaMA-Factory `0.9.3.dev0`.

## Build the Parent–Child index

```bash
python -m bank_llm.build_index \
  --pdf-dir /path/to/pdfs \
  --output-dir artifacts/index
```

Replace `/path/to/...` placeholders with your own PDF or model directories.
Paths can be relative to the repository root or absolute.

The command creates:

```text
artifacts/index/banking.index
artifacts/index/chunks.pkl
artifacts/index/index_stats.json
```

Important defaults:

- Embedding model: `BAAI/bge-large-zh-v1.5`
- Maximum Parent length before semantic splitting: 800 characters
- Semantic boundary threshold: 0.85
- Minimum Child length: 60 characters
- Index: normalized embeddings with FAISS inner-product search

All values can be changed through command-line arguments.

## Run end-to-end chat

### Base model plus LoRA adapter

```bash
python -m bank_llm.chat \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter /path/to/final-lora-adapter \
  --index artifacts/index/banking.index \
  --chunks artifacts/index/chunks.pkl \
  --show-sources
```

### Merged model

```bash
python -m bank_llm.chat \
  --model /path/to/merged-model \
  --index artifacts/index/banking.index \
  --chunks artifacts/index/chunks.pkl
```

For one-shot inference, add:

```bash
--query "商业银行资本管理新规主要内容是什么"
```

## Training with LLaMA-Factory

The verified SFT configuration is provided at
[`configs/sft_verified_rank16_lr5e-5.yaml`](configs/sft_verified_rank16_lr5e-5.yaml).

```bash
llamafactory-cli train configs/sft_verified_rank16_lr5e-5.yaml
```

The original bank DPO WebUI state was not retained. The included
[`configs/dpo_reconstructed.yaml`](configs/dpo_reconstructed.yaml) documents
the correct continuation structure but is explicitly marked as reconstructed:

```text
base model + SFT LoRA adapter
  -> stage=dpo
  -> ranking dataset
  -> continue the same LoRA adapter
```

Do not treat the reconstructed DPO hyperparameters as a historical record.

## Acknowledgements

- Qwen2.5-3B-Instruct
- LLaMA-Factory
- BGE large Chinese embeddings and BGE reranker
- FAISS and BM25
