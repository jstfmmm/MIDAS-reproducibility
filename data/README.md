# Local data and model directories

This directory is a placeholder only. The clean repository intentionally does not contain datasets, model weights, retrieval indexes, cached web pages, or checkpoints.

- Put processed JSON/JSONL files under `datasets/`.
- Put downloaded encoders, generators, and the BEM model under `model/`, or pass a Hugging Face model ID/local path on the command line.
- Put local FAISS/Wikipedia snapshots expected by the Chinese Wiki retriever under `index/`.
- Keep generated results and checkpoints under the ignored root `output/` directory.
