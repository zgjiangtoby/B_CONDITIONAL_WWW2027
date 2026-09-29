"""Train paired readouts, export scores, or estimate paired effects on external banks."""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

import bank
from evaluate import HEADS, paired_results
from readouts import SpanHead, score_pair, seed_everything, train_pair


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("smoke")
    p = sub.add_parser("fit")
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--dev", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("score")
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--heads", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("analyze")
    p.add_argument("--banks", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "smoke":
        from smoke import main as smoke
        smoke()
    elif args.command == "fit":
        bank.require(not args.out.exists(), "Output directory already exists")
        tr, dv = bank.load(args.train), bank.load(args.dev)
        _, states, history = train_pair(tr, dv, args.device)
        args.out.mkdir(parents=True, exist_ok=False)
        for variant, state in states.items():
            state.update(train_sha256=bank.sha256(args.train), dev_sha256=bank.sha256(args.dev))
            torch.save(state, args.out / (variant + ".pt"))
        write_csv(args.out / "history.csv", history)
    elif args.command == "score":
        bank.require(not args.out.exists(), "Output file already exists")
        data = bank.load(args.bank)
        bank.require(str(data["split"]) in ("dev", "test"), "Score only a development/test bank")
        seed_everything(int(data["seed"]))
        heads = {}
        for variant in HEADS:
            state = torch.load(args.heads / (variant + ".pt"), map_location="cpu", weights_only=True)
            for key in ("dataset", "seed", "model_sha256"):
                bank.require(state[key] == data[key].item(), f"Head/model mismatch: {key}")
            bank.require(state["variant"] == variant and state["epochs"] == bank.EPOCHS[state["dataset"]],
                         "Incomplete or mismatched paired-head fit")
            bank.require(state["width"] == data["hidden"].shape[-1] and
                         state["max_length"] == data["hidden"].shape[1], "Feature dimensions differ")
            head = SpanHead(state["width"], state["categories"], variant).to(args.device)
            head.load_state_dict(state["state"], strict=True)
            heads[variant] = head.eval()
        bank.save(args.out, score_pair(heads, data))
    else:
        bank.require(not args.out.exists(), "Output file already exists")
        write_csv(args.out, paired_results([bank.load(p, features=False) for p in args.banks]))
    if args.command != "smoke":
        print(f"Completed {args.command}: {args.out}")


if __name__ == "__main__":
    main()
