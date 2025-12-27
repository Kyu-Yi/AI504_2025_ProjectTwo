# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project overview

This repository contains a single training script, `20254086.py`, for AI504 Project 2: training a GPT-2 language model on the ELI5 dataset. The script relies on a separate `base.py` (provided externally by course staff) for data preparation and loading.

Key characteristics:
- Uses Hugging Face GPT-2 (small) via `transformers`.
- Uses a TA-provided `base.py` module to obtain datasets/dataloaders.
- Trains briefly (few epochs) with early stopping, then saves test-set logits to `{STUDENT_ID}.npy` in a strict shape for grading.

## Commands

All commands assume the working directory is the repo root (`/home/kyu-yi/AI504_2025_ProjectTwo`). Ensure `base.py` from the course starter code is placed next to `20254086.py`.

### Environment setup

Use any Python environment manager (venv, conda, etc.). After activating it, install runtime dependencies inferred from imports:

```bash
pip install torch transformers datasets tqdm numpy
```

### Run training & generate logits

Train GPT-2 on the ELI5 data (using helpers from `base.py`) and save test logits to `20254086.npy`:

```bash
python 20254086.py
```

This script will:
- Print versions of `python`, `torch`, `transformers`, and `datasets`.
- Import `base.py`, build tokenizer/model, construct dataloaders from `base`, train with early stopping, then write `./20254086.npy` containing test logits.

### Adjusting or resuming experiments

For quick experimentation, you will typically edit `20254086.py` directly and re-run the script:
- Training configuration is centralized in `main()` (lines defining `epochs`, `lr`, `wd`, `accum`).
- Sequence length, vocabulary size, and test sample count are controlled by the module-level constants `SEQ_LEN`, `VOCAB_SIZE`, and `TEST_SAMPLES`.

There is no separate test suite or CLI entrypoint; the primary workflow is to invoke `python 20254086.py` after code changes.

## Code structure and architecture

The project is a single-module training pipeline with a clear separation between data orchestration, model construction, and training logic.

### External dependency: `base.py`

The `base` module (not checked into this repo) is expected to be colocated with `20254086.py` and to expose at least one of the following helpers:
- `get_dataloaders(tokenizer, bs_train, bs_val, bs_test)`
- `get_datasets(tokenizer)` (paired with PyTorch `DataLoader` construction in this script)
- `prepare_datasets(tokenizer)` plus `ELI5Dataset(split=...)`

`get_dataloaders(tokenizer)` in `20254086.py` dispatches based on which helpers exist in `base` and always returns three PyTorch `DataLoader` instances: `(train_dl, val_dl, test_dl)`.

When editing data-related behavior, prefer to adjust `base.py` (if allowed by the assignment) rather than duplicating dataset logic in this file.

### Tokenizer and model construction

- `build_tokenizer()`
  - Loads `GPT2TokenizerFast` from the `gpt2` checkpoint.
  - Ensures a valid `pad_token` by assigning `eos_token` if needed.

- `build_model(tokenizer)`
  - Loads `GPT2LMHeadModel` from the `gpt2` checkpoint.
  - Calls `resize_token_embeddings(len(tokenizer))` so the embedding matrix accounts for the pad token.

These functions encapsulate all Hugging Face–specific setup; if you swap architectures or checkpoints, do it here.

### Training & evaluation flow

- Global configuration and seeding
  - `STUDENT_ID`, `OUT_PATH`, `SEQ_LEN`, `VOCAB_SIZE`, and `TEST_SAMPLES` define the output contract for grading.
  - `hard_seed()` enforces deterministic behavior across NumPy, Python `random`, and PyTorch (CPU and CUDA).
  - If `base.set_seed` exists, it is called with `0`; otherwise the local `hard_seed(0)` is used.
  - `DEVICE` auto-selects GPU if available (`cuda`) or falls back to CPU.

- Epoch training loop
  - `train_one_epoch(model, loader, optim, sched=None, grad_accum=2, max_norm=1.0)`
    - Handles gradient accumulation, gradient clipping, optimizer stepping, and optional LR scheduling.
    - Supports both dict-style batches (`{"input_ids", "attention_mask", "labels"}`) and tuple-style batches (`(input_ids, ...)`).

- Perplexity evaluation
  - `evaluate_ppl(model, loader)`
    - Runs the model in eval mode on a given loader, collects loss values, and returns `exp(mean_loss)` (capped to avoid overflow).

- Main training driver
  - `main()` wires everything together:
    - Builds tokenizer and dataloaders from `base`.
    - Builds GPT-2 and moves it to `DEVICE`.
    - Configures `AdamW` and a linear warmup scheduler via `get_linear_schedule_with_warmup`.
    - Runs a small number of epochs, tracking best validation perplexity with a patience-based early stopping scheme.
    - After training, saves test logits and (optionally) prints test perplexity.

### Test logits and grading contract

- `save_test_logits(model, loader, out_path)`
  - Runs the model on the test loader and collects raw logits `[batch, seq_len, vocab_size]` on CPU.
  - Enforces exact shapes using the global constants:
    - Truncates to `SEQ_LEN` tokens.
    - Truncates to the first `VOCAB_SIZE` vocabulary entries.
    - Truncates the batch dimension to `TEST_SAMPLES`.
  - Asserts the final NumPy array has shape `(TEST_SAMPLES, SEQ_LEN, VOCAB_SIZE)` and saves it to `OUT_PATH` via `np.save`.

Any future modifications must preserve this contract (output filename and shape) for automated grading to succeed.
