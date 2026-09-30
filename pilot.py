"""Generated CPU-only checks; no saved fixture or downloaded asset is required."""
import copy
import hashlib
from pathlib import Path
import tempfile

import numpy as np
import torch

import bank
from evaluate import HEADS, Frame, paired_results, record_effects, signed_counts, swap_probabilities
from readouts import SpanHead, score_pair, train_pair


def fixture(dataset="hatexplain", split="train", seed=17):
    n, length, width = 4, 6, 8
    p = np.full((n, len(bank.CLASSES[dataset])), .01, dtype=np.float32)
    labels = {"tama": [0, 1, 2, -1], "hatexplain": [0, 2, 0, 1], "plead": [0, 1, 2, 4]}[dataset]
    predictions = {"tama": [1, 2, 2, 0], "hatexplain": [2, 0, 1, 1], "plead": [1, 0, 4, 4]}[dataset]
    p[np.arange(n), predictions] = 1 - .01 * (p.shape[1] - 1)
    refs = [[[1, 0, 0], [0, 1, 0]], [[0, 1, 0], [0, 1, 1], [0, 0, 1]], [[0, 0, 0]], []]
    if dataset == "tama":
        refs = [[x[0]] if x else [] for x in refs]
    word_map = np.tile(np.array([-1, 0, 1, 1, 2, -1], np.int64), (n, 1))
    post = word_map >= 0
    span_gold = np.zeros((n, length), np.float32)
    for i in range(n):
        primary = bank.reference_masks(refs[i], 3)[0]
        span_gold[i, post[i]] = primary[word_map[i, post[i]]]
    kwargs = dict(dataset=dataset, split=split, seed=seed,
                  model_sha256=hashlib.sha256(f"generated:{dataset}:{seed}".encode()).hexdigest(),
                  ids=[f"{split}_{i}" for i in range(n)], groups=[f"{split}_group_{i // 2}" for i in range(n)],
                  hidden=np.random.default_rng(seed + len(split)).normal(size=(n, length, width)).astype(np.float32),
                  p=p, gold_class=np.array(labels, np.int64), token_lengths=[3] * n, references=refs,
                  span_gold=span_gold, post_mask=post,
                  span_mask=post & np.array([True, True, True, dataset == "tama"])[:, None])
    if dataset == "hatexplain":
        kwargs["word_map"] = word_map
    else:
        projection = bank.character_projection([(0, 2), (3, 5), (6, 8)],
                                               [(0, 0), (0, 2), (3, 4), (4, 5), (6, 8), (0, 0)])
        kwargs["projection"] = np.tile(projection, (n, 1))
    if dataset == "tama":
        kwargs.update(p1=np.array([[.1, .8, .1], [.8, .1, .1], [.1, .8, .1], [.8, .1, .1]], np.float32),
                      gold_target=np.array([1, 1, 1, 0], np.int64), span_valid=np.ones(n, bool))
    return bank.make_bank(**kwargs)


def rejects(call):
    try:
        call()
    except (ValueError, KeyError):
        return
    raise AssertionError("Invalid input was accepted")


def main():
    all_scores = []
    with tempfile.TemporaryDirectory(prefix="conditional-source-smoke-") as temporary:
        tmp = Path(temporary)
        for dataset in bank.CLASSES:
            tr, dv, test = [fixture(dataset, split) for split in ("train", "dev", "test")]
            original_h, original_p = tr["hidden"].copy(), tr["p"].copy()
            models, states, history = train_pair(tr, dv, epochs=2)
            assert np.array_equal(tr["hidden"], original_h) and np.array_equal(tr["p"], original_p)
            assert states[HEADS[0]]["parameters_per_head"] == states[HEADS[1]]["parameters_per_head"]
            scores = score_pair(models, test)
            bank.save(tmp / f"{dataset}_features.npz", test)
            restored = bank.load(tmp / f"{dataset}_features.npz")
            assert np.array_equal(restored["hidden"], test["hidden"])
            for head in HEADS:
                assert states[head]["epoch"] == 1 + int(np.argmax([r[head + "_dev_f1"] for r in history]))
                torch.save(states[head], tmp / f"{head}.pt")
                state = torch.load(tmp / f"{head}.pt", map_location="cpu", weights_only=True)
                restored_head = SpanHead(state["width"], state["categories"], head)
                restored_head.load_state_dict(state["state"])
                torch.testing.assert_close(restored_head(torch.from_numpy(test["hidden"]), torch.from_numpy(test["p"])),
                                           models[head](torch.from_numpy(test["hidden"]), torch.from_numpy(test["p"])), atol=0, rtol=0)
                for row in record_effects(scores, head):
                    assert abs(row["g_native"] - row["g_identity"] - row["hardening"]) < 1e-12
                    assert -1e-12 <= row["g_native"] + row["stress_d"] <= row["stress_r"] + 1e-12
            swapped, active, unique = swap_probabilities(test["p"], test["gold_class"], test["primary_mask"])
            assert np.array_equal(np.sort(swapped, 1), np.sort(test["p"], 1))
            assert np.array_equal(swapped[~active], test["p"][~active])
            assert (swapped.argmax(1)[active & unique] == test["gold_class"][active & unique]).all()
            if dataset == "tama":
                assert test["primary_mask"].tolist() == [True, False, False, False]
                assert test["span_mask"][3].any()  # Existing non-TA negative supervision.
            else:
                assert test["primary_mask"].tolist() == [True, True, False, False]
                assert test["span_applicable"].tolist() == [True, True, True, False]
                assert not test["span_mask"][3].any()
            np.testing.assert_allclose(bank.project(test, 0, np.array([0, .2, .4, .8, .7, 0], np.float32)),
                                       [.2, .6, .7], atol=1e-7, rtol=0)
            invalid = dict(test, primary_mask=~test["primary_mask"])
            rejects(lambda: bank.validate(invalid))
            invalid = dict(test, gold_primary=~test["gold_primary"])
            rejects(lambda: bank.validate(invalid))
            rejects(lambda: train_pair(tr, dict(dv, group=tr["group"]), epochs=1))
            # Seed clones exercise paired aggregation, not a claim of four trained fits.
            for seed in bank.SEEDS:
                item = copy.deepcopy(scores)
                item["seed"] = np.array(seed)
                all_scores.append(item)
        estimates = paired_results(all_scores)
        mean_rows = [r for r in estimates if r["seed"] == "four_seed_mean"]
        assert mean_rows and all(r["seed_sd"] is None or r["seed_sd"] == 0 for r in mean_rows)
        rejects(lambda: paired_results(all_scores[:-1]))
        # Reference correction interaction must be paired within each post.
        for ds in ("hatexplain", "plead"):
            lookup = {(r["reference"], r["metric"]): r["estimate"] for r in mean_rows
                      if r["dataset"] == ds and r["head"] == "additive_minus_multiplicative"
                      and r["analysis"] == "reference_minus_primary" and r["cohort"] == "type_errors"}
            for ref in ("strict", "union", "individual_mean"):
                assert abs(lookup[ref, "g_swap"] - lookup[ref, "swap_q"] + lookup[ref, "native_q"]) < 1e-12
        frame = Frame("generated", ["a", "b", "c"], {"a": "x", "b": "x", "c": "y"})
        point, draws = frame.means(np.array([[1.], [2.], [3.]]), np.array([True, True, False]))
        assert point[0] == 1.5 and np.isnan(draws).any() and (draws[np.isfinite(draws)] == 1.5).all()
        assert not any(tmp_path.suffix == ".zip" for tmp_path in tmp.iterdir())
    for gold in range(16):
        g = [(gold >> i) & 1 for i in range(4)]
        for before in range(16):
            p = [(before >> i) & 1 for i in range(4)]
            for after in range(16):
                signed_counts(g, p, [(after >> i) & 1 for i in range(4)])
    print("PASS: CPU paired fitting/scoring for all three schemas; checkpoint roundtrip; missing/empty references; "
          "native cohorts; exact projection; paired four-seed/reference bootstrap; 4096 signed-F1 cases. No data retained.")


if __name__ == "__main__":
    main()
