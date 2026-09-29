"""Matched frozen-feature readouts extracted from the paper's refit implementation."""
import os
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from bank import EPOCHS, project, require, validate
from evaluate import HEADS, positive_f1, swap_probabilities, validate_scores


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class SpanHead(nn.Module):
    def __init__(self, width, categories, variant):
        super().__init__()
        require(variant in HEADS, "Unsupported head")
        self.variant = variant
        self.modulation = nn.Linear(categories, width, bias=False)
        self.span = nn.Linear(width, 1)

    def forward(self, h, p):
        z = self.modulation(p)[:, None, :]
        x = h * z.tanh() if self.variant == "multiplicative" else (h + z).tanh()
        return self.span(x).squeeze(-1)


def predict(head, bank, p=None):
    p = bank["p"] if p is None else p
    device = next(head.parameters()).device
    result = []
    with torch.no_grad():
        for start in range(0, len(p), 16):
            end = min(start + 16, len(p))
            h = torch.as_tensor(bank["hidden"][start:end], device=device)
            probability = torch.as_tensor(p[start:end], device=device)
            q = head(h, probability).sigmoid().cpu().numpy()
            result.extend(project(bank, i, q[i - start]) for i in range(start, end))
    return np.concatenate(result)


def dev_score(values, bank):
    scores = []
    for i in np.flatnonzero(bank["span_applicable"]):
        start, stop = bank["offsets"][i:i + 2]
        scores.append(positive_f1(bank["gold_primary"][start:stop], values[start:stop] >= .5))
    require(bool(scores), "Development partition has no valid span references")
    return float(np.mean(scores))


def train_pair(train, dev, device="cpu", epochs=None):
    validate(train); validate(dev)
    require(str(train["split"]) == "train" and str(dev["split"]) == "dev", "Train/dev partitions required")
    for key in ("dataset", "seed", "model_sha256"):
        require(np.array_equal(train[key], dev[key]), f"Different frozen model identity: {key}")
    require(not set(train["ids"]) & set(dev["ids"]), "Train/dev record overlap")
    require(not set(train["group"]) & set(dev["group"]), "Train/dev exact-text overlap")
    require(train["hidden"].shape[1:] == dev["hidden"].shape[1:], "Feature dimensions differ")
    require(train["span_mask"].any() and dev["span_applicable"].any(), "No training/development supervision")
    ds, seed = str(train["dataset"]), int(train["seed"])
    epochs = EPOCHS[ds] if epochs is None else epochs
    require(1 <= epochs <= EPOCHS[ds], "Invalid schedule")
    width, categories = train["hidden"].shape[-1], train["p"].shape[-1]
    heads = {}
    for variant in HEADS:
        seed_everything(seed)
        heads[variant] = SpanHead(width, categories, variant).to(device)
    require(all(torch.equal(v, heads[HEADS[1]].state_dict()[k])
                for k, v in heads[HEADS[0]].state_dict().items()), "Initial arrays differ")
    count = sum(p.numel() for p in heads[HEADS[0]].parameters())
    require(count == width * categories + width + 1, "Parameter budget mismatch")
    optimizers = {v: torch.optim.AdamW(h.parameters(), lr=.001, weight_decay=.01) for v, h in heads.items()}
    best, chosen, history = {v: -1. for v in HEADS}, {}, []
    # ponytail: precomputed banks reside in RAM; use external sharded/memory-mapped features if RAM becomes limiting.
    for epoch in range(1, epochs + 1):
        order = np.random.default_rng(seed + epoch).permutation(len(train["ids"]))
        losses = {v: [] for v in HEADS}
        for start in range(0, len(order), 16):
            ix = order[start:start + 16]
            h = torch.as_tensor(train["hidden"][ix], device=device)
            p = torch.as_tensor(train["p"][ix], device=device)
            mask = torch.as_tensor(train["span_mask"][ix], device=device)
            if not mask.any():
                continue
            gold = torch.as_tensor(train["span_gold"][ix], device=device)
            for variant, head in heads.items():
                optimizer = optimizers[variant]
                optimizer.zero_grad(set_to_none=True)
                logits = head(h, p)
                loss = F.binary_cross_entropy_with_logits(logits[mask], gold[mask])
                require(bool(torch.isfinite(loss)), "Non-finite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
                optimizer.step()
                losses[variant].append(float(loss.detach()))
        entry = dict(epoch=epoch)
        for variant, head in heads.items():
            score = dev_score(predict(head, dev), dev)
            entry[variant + "_loss"] = float(np.mean(losses[variant]))
            entry[variant + "_dev_f1"] = score
            if score > best[variant]:
                best[variant] = score
                chosen[variant] = dict(state={k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                                       epoch=epoch, dev_score=score)
        history.append(entry)
    metadata = dict(dataset=ds, seed=seed, model_sha256=str(train["model_sha256"]), width=width,
                    categories=categories, max_length=train["hidden"].shape[1], epochs=epochs,
                    parameters_per_head=count, batch_size=16, learning_rate=.001, weight_decay=.01,
                    gradient_clip=1., threshold=.5, selection="earliest best development native span F1")
    for variant, head in heads.items():
        head.load_state_dict(chosen[variant]["state"])
        require(chosen[variant]["epoch"] == 1 + int(np.argmax([r[variant + "_dev_f1"] for r in history])),
                "Wrong development checkpoint")
        chosen[variant].update(metadata, variant=variant)
    return heads, chosen, history


def score_pair(heads, bank):
    validate(bank)
    swapped, _, _ = swap_probabilities(bank["p"], bank["gold_class"], bank["primary_mask"])
    omitted = {"hidden", "span_gold", "span_mask", "post_mask", "projection", "word_map"}
    result = {k: v for k, v in bank.items() if k not in omitted}
    for variant in HEADS:
        head = heads[variant]
        require(head.variant == variant, "Wrong head identity")
        result[variant + "_native"] = predict(head, bank)
        result[variant + "_swap"] = predict(head, bank, swapped)
        eye = np.eye(bank["p"].shape[1], dtype=np.float32)
        result[variant + "_onehot"] = np.stack([
            predict(head, bank, np.broadcast_to(e, bank["p"].shape).copy()) for e in eye])
    return validate_scores(result)
