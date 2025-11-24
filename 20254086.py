# 20254086.py
# AI504 Project 2 — Training a Language Model on ELI5
# Student ID: 20254086

import os, math, time, platform, warnings, random
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

STUDENT_ID = "20254086"
OUT_PATH = f"./{STUDENT_ID}.npy"
SEQ_LEN = 200
VOCAB_SIZE = 50257
TEST_SAMPLES = 75

def print_env():
    try:
        import torch, transformers, datasets  # noqa
        print(f"[ENV] Python={platform.python_version()} | torch={torch.__version__} | "
              f"transformers={transformers.__version__} | datasets={datasets.__version__}")
    except Exception as e:
        print(f"[ENV] version print failed: {e}")

print_env()

# base here
try:
    import base  # provided by TA
except Exception as e:
    print("[ERROR] Could not import base.py. Put 20254086.py and base.py together.")
    raise e

# ---- Seed = 0 (both TA's function and hardening here) ----
def hard_seed(s: int = 0):
    np.random.seed(s); random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if hasattr(base, "set_seed"):
    base.set_seed(0)
else:
    hard_seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# tokenizer & model (GPT-2 small) - still undecided
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, get_linear_schedule_with_warmup

def build_tokenizer() -> GPT2TokenizerFast:
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def build_model(tok: GPT2TokenizerFast) -> GPT2LMHeadModel:
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tok))  # account for pad token
    return model

# Get dataloaders from base
def get_dataloaders(tok: GPT2TokenizerFast) -> Tuple[DataLoader, DataLoader, DataLoader]:
    bs_train, bs_eval = 8, 8  

    if hasattr(base, "get_dataloaders"):
        print("[INFO] base.get_dataloaders(...)")
        return base.get_dataloaders(tok, bs_train, bs_eval, bs_eval)

    if hasattr(base, "get_datasets"):
        print("[INFO] base.get_datasets(...) → building DataLoaders")
        tr, va, te = base.get_datasets(tok)
        return (DataLoader(tr, batch_size=bs_train, shuffle=True),
                DataLoader(va, batch_size=bs_eval, shuffle=False),
                DataLoader(te, batch_size=bs_eval, shuffle=False))

    if hasattr(base, "prepare_datasets"):
        print("[INFO] base.prepare_datasets(...) then base.ELI5Dataset")
        base.prepare_datasets(tok)

    if hasattr(base, "ELI5Dataset"):
        print("[INFO] base.ELI5Dataset(split=...)")
        tr = base.ELI5Dataset(split="train")
        va = base.ELI5Dataset(split="validation")
        te = base.ELI5Dataset(split="test")
        return (DataLoader(tr, batch_size=bs_train, shuffle=True),
                DataLoader(va, batch_size=bs_eval, shuffle=False),
                DataLoader(te, batch_size=bs_eval, shuffle=False))

    raise RuntimeError("base.py did not expose expected helpers (get_dataloaders/get_datasets/ELI5Dataset).")

# ---- Training / evaluation ----
def evaluate_ppl(model: GPT2LMHeadModel, loader: DataLoader) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(DEVICE)
                attn = batch.get("attention_mask", (input_ids != 0).long()).to(DEVICE)
                labels = batch.get("labels", input_ids).to(DEVICE)
            else:
                input_ids = batch[0].to(DEVICE)
                attn = (input_ids != 0).long().to(DEVICE)
                labels = input_ids
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            losses.append(out.loss.detach().float().item())
    m = float(np.mean(losses)) if losses else 1e9
    return math.exp(min(m, 20))

def train_one_epoch(model, loader, optim, sched=None, grad_accum=2, max_norm=1.0):
    model.train()
    running, steps = 0.0, 0
    pbar = tqdm(loader, desc="Train", leave=False)
    optim.zero_grad(set_to_none=True)
    for i, batch in enumerate(pbar, 1):
        if isinstance(batch, dict):
            input_ids = batch["input_ids"].to(DEVICE)
            attn = batch.get("attention_mask", (input_ids != 0).long()).to(DEVICE)
            labels = batch.get("labels", input_ids).to(DEVICE)
        else:
            input_ids = batch[0].to(DEVICE)
            attn = (input_ids != 0).long().to(DEVICE)
            labels = input_ids

        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        loss = out.loss / grad_accum
        loss.backward()

        if i % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optim.step()
            if sched is not None: sched.step()
            optim.zero_grad(set_to_none=True)

        steps += 1
        running += loss.item() * grad_accum
        pbar.set_postfix({"loss": f"{running/steps:.4f}"})
    return running / max(1, steps)

# save raw test logits with required shape
def save_test_logits(model: GPT2LMHeadModel, loader: DataLoader, out_path: str):
    model.eval()
    chunks = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Logits[Test]", leave=False):
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(DEVICE)
                attn = batch.get("attention_mask", (input_ids != 0).long()).to(DEVICE)
            else:
                input_ids = batch[0].to(DEVICE)
                attn = (input_ids != 0).long().to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attn).logits  # [B, T, V]
            chunks.append(logits.detach().cpu().to(torch.float32))

    logits = torch.cat(chunks, dim=0)  # [N, T, V]
    # Enforce exact dims
    if logits.shape[1] != SEQ_LEN:
        logits = logits[:, :SEQ_LEN, :]
    if logits.shape[2] != VOCAB_SIZE:
        logits = logits[:, :, :VOCAB_SIZE]
    if logits.shape[0] > TEST_SAMPLES:
        logits = logits[:TEST_SAMPLES]
    npy = logits.numpy()

    # Final contract checks
    assert npy.ndim == 3
    assert npy.shape[1] == SEQ_LEN, npy.shape
    assert npy.shape[2] == VOCAB_SIZE, npy.shape

    print(f"[OK] Saving logits to {out_path} with shape {npy.shape}")
    np.save(out_path, npy)

def main():
    tokenizer = build_tokenizer()
    train_dl, val_dl, test_dl = get_dataloaders(tokenizer)
    model = build_model(tokenizer).to(DEVICE)

    # Lightweight fine-tune config (fits Colab Free, 24h window)
    epochs = 2                      # bump to 3 locally if time allows
    lr, wd = 5e-5, 0.01
    accum = 2
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    total_steps = max(1, (len(train_dl) // accum) * epochs)
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=max(1, total_steps // 10),
                                            num_training_steps=total_steps)
    print(f"[INFO] epochs={epochs} lr={lr} steps≈{total_steps} grad_accum={accum}")

    best_ppl, patience, no_improve = float("inf"), 2, 0
    for ep in range(1, epochs + 1):
        tr = train_one_epoch(model, train_dl, optim, sched, grad_accum=accum, max_norm=1.0)
        vp = evaluate_ppl(model, val_dl)
        print(f"Epoch {ep:02d}/{epochs} | train_loss={tr:.4f} | val_ppl={vp:.2f}")
        if vp + 0.5 < best_ppl:
            best_ppl, no_improve = vp, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("[INFO] Early stopping.")
                break

    # Save test logits for grading
    save_test_logits(model, test_dl, OUT_PATH)

    # Optional reference metric (not required for grading)
    try:
        tp = evaluate_ppl(model, test_dl)
        print(f"[INFO] Test perplexity (for reference): {tp:.2f}")
    except Exception as e:
        print(f"[WARN] Could not compute test perplexity: {e}")

if __name__ == "__main__":
    main()
