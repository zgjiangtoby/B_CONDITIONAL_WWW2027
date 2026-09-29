"""External frozen-feature connector. No corpus, labels, or checkpoint is bundled."""
import hashlib
import json
from pathlib import Path

import numpy as np

SEEDS = (17, 29, 43, 59)
CLASSES = {
    "tama": ("death_threat", "sexual_assault", "sexual_explicit", "physical_harm",
             "radiation_of_threats", "attacks_on_credibility", "misogynistic", "homophobic",
             "religious", "political_sectarian", "racist", "general"),
    "hatexplain": ("hatespeech", "normal", "offensive"),
    "plead": ("comparison", "derogation", "threatening", "hatecrime", "nothate"),
}
POSITIVE = {"tama": tuple(range(12)), "hatexplain": (0, 2), "plead": (0, 1, 2, 3)}
EPOCHS = {"tama": 30, "hatexplain": 8, "plead": 16}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def exact_group(value, dataset):
    require(dataset in CLASSES, "Unknown corpus")
    if dataset == "hatexplain":
        require(isinstance(value, (list, tuple)) and all(isinstance(t, str) for t in value),
                "HateXplain groups require the exact released token sequence")
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    require(isinstance(value, str), "TAMA/PLEAD groups require exact original text")
    return hashlib.sha256(value.encode()).hexdigest()


def character_projection(token_spans, subword_spans):
    """Original character-intersection weights; uncovered evaluation tokens stay zero."""
    a = np.asarray(token_spans, dtype=np.int64).reshape(-1, 2)
    b = np.asarray(subword_spans, dtype=np.int64).reshape(-1, 2)
    overlap = np.maximum(0, np.minimum(a[:, None, 1], b[None, :, 1]) -
                         np.maximum(a[:, None, 0], b[None, :, 0]))
    denominator = overlap.sum(1, keepdims=True)
    return np.divide(overlap, denominator, out=np.zeros_like(overlap, dtype=np.float32),
                     where=denominator != 0)


def project(bank, i, probability):
    start, stop = bank["offsets"][i:i + 2]
    if str(bank["dataset"]) != "hatexplain":
        return bank["projection"][start:stop] @ probability
    words = bank["word_map"][i]
    visible = words >= 0
    counts = np.bincount(words[visible], minlength=stop - start)
    sums = np.bincount(words[visible], weights=probability[visible], minlength=stop - start)
    return np.divide(sums, counts, out=np.zeros(stop - start, np.float64),
                     where=counts > 0).astype(np.float32)


def reference_masks(references, token_count):
    """An empty sequence means missing; a valid all-zero vector remains a reference."""
    refs = [np.asarray(r) for r in references]
    require(all(r.shape == (token_count,) and np.isin(r, (0, 1)).all() for r in refs),
            "Reference length/nonbinary label mismatch")
    if not refs:
        zero = np.zeros(token_count, dtype=bool)
        return zero, zero.copy(), zero.copy()
    votes = np.asarray(refs, dtype=bool).sum(0)
    return 2 * votes >= len(refs), 2 * votes > len(refs), votes > 0


def make_bank(*, dataset, split, seed, model_sha256, ids, groups, hidden, p, gold_class,
              token_lengths, references, span_gold, span_mask, post_mask,
              projection=None, word_map=None, p1=None, gold_target=None, span_valid=None):
    """Connect already adapted records/features; see README for corpus-specific supervision."""
    require(dataset in CLASSES, "Unknown corpus")
    n = len(ids)
    require(len(references) == len(token_lengths) == n, "One reference list/length per post")
    offsets = np.r_[0, np.cumsum(token_lengths)].astype(np.int64)
    owners, masks, lengths = [], [], []
    primary, strict, union = [], [], []
    for i, (refs, length) in enumerate(zip(references, token_lengths)):
        a, b, c = reference_masks(refs, length)
        primary.extend(a); strict.extend(b); union.extend(c)
        for ref in refs:
            owners.append(i); masks.extend(ref); lengths.append(length)
    bank = dict(dataset=np.array(dataset), split=np.array(split), seed=np.array(seed),
                model_sha256=np.array(model_sha256), ids=np.asarray(ids), group=np.asarray(groups),
                hidden=np.asarray(hidden), p=np.asarray(p), gold_class=np.asarray(gold_class),
                offsets=offsets, span_gold=np.asarray(span_gold), span_mask=np.asarray(span_mask),
                post_mask=np.asarray(post_mask), gold_primary=np.asarray(primary, bool),
                gold_strict=np.asarray(strict, bool), gold_union=np.asarray(union, bool),
                individual_record=np.asarray(owners, np.int64),
                individual_offsets=np.r_[0, np.cumsum(lengths)].astype(np.int64),
                individual_gold=np.asarray(masks, bool))
    for key, value in (("projection", projection), ("word_map", word_map), ("p1", p1),
                       ("gold_target", gold_target), ("span_valid", span_valid)):
        if value is not None:
            bank[key] = np.asarray(value)
    available = np.bincount(bank["individual_record"], minlength=n) > 0
    bank["span_applicable"] = np.isin(bank["gold_class"], POSITIVE[dataset]) & available
    bank["primary_mask"] = bank["span_applicable"] & np.isin(bank["p"].argmax(1), POSITIVE[dataset]) & (
        bank["p"].argmax(1) != bank["gold_class"])
    if dataset == "tama":
        require(p1 is not None and gold_target is not None and span_valid is not None,
                "TAMA requires native target probabilities, target labels and validity flags")
        bank["primary_mask"] &= bank["p1"].argmax(1) == bank["gold_target"]
    return validate(bank)


def validate(bank, features=True):
    """Fail on unsupported/missing inputs, without inferring labels or alignment."""
    for key in ("dataset", "split", "seed", "model_sha256"):
        require(key in bank and bank[key].shape == (), f"Missing scalar {key}")
    ds = str(bank["dataset"])
    require(ds in CLASSES and int(bank["seed"]) in SEEDS, "Unsupported corpus/seed")
    require(str(bank["split"]) in ("train", "dev", "test"), "Unsupported partition")
    digest = str(bank["model_sha256"])
    require(len(digest) == 64 and all(x in "0123456789abcdef" for x in digest), "Invalid model SHA-256")
    ids = bank["ids"]
    n, c = len(ids), len(CLASSES[ds])
    require(n > 0 and ids.shape == (n,) and ids.dtype.kind in "US" and len(set(ids)) == n,
            "IDs must be unique strings")
    require(bank["group"].shape == (n,) and bank["group"].dtype.kind in "US" and
            all(str(x) for x in bank["group"]), "Missing exact-text groups")
    offsets = bank["offsets"]
    require(offsets.dtype.kind in "iu" and offsets.shape == (n + 1,) and offsets[0] == 0 and
            (np.diff(offsets) > 0).all(), "Invalid token offsets")
    t = int(offsets[-1])
    require(bank["p"].dtype == np.float32 and bank["p"].shape == (n, c), "p must be float32[N,C]")
    require(np.isfinite(bank["p"]).all() and (bank["p"] >= 0).all() and
            np.allclose(bank["p"].sum(1), 1, atol=2e-6, rtol=0), "Invalid probabilities")
    gold = bank["gold_class"]
    require(gold.shape == (n,) and gold.dtype.kind in "iu" and
            ((gold >= (-1 if ds == "tama" else 0)) & (gold < c)).all(), "Invalid existing categories")
    for key in ("gold_primary", "gold_strict", "gold_union"):
        require(bank[key].shape == (t,) and bank[key].dtype == bool, f"Invalid {key}")
    owners, io, ig = (bank[k] for k in ("individual_record", "individual_offsets", "individual_gold"))
    require(owners.ndim == 1 and owners.dtype.kind in "iu" and ((owners >= 0) & (owners < n)).all(),
            "Invalid reference owners")
    require(io.dtype.kind in "iu" and io.shape == (len(owners) + 1,) and io[0] == 0 and
            io[-1] == len(ig) and ig.ndim == 1 and ig.dtype == bool and
            np.array_equal(np.diff(io), np.diff(offsets)[owners]), "Invalid reference boundaries")
    counts = np.bincount(owners, minlength=n)
    applicable = np.isin(gold, POSITIVE[ds]) & (counts > 0)
    for i in range(n):
        a, b = offsets[i:i + 2]
        refs = [ig[io[j]:io[j + 1]] for j in np.flatnonzero(owners == i)]
        expected = reference_masks(refs, b - a)
        require(all(np.array_equal(bank[k][a:b], v) for k, v in
                    zip(("gold_primary", "gold_strict", "gold_union"), expected)), "Reference criteria disagree")
    pred = bank["p"].argmax(1)
    primary = applicable & np.isin(pred, POSITIVE[ds]) & (pred != gold)
    if ds == "tama":
        require(bank["p1"].shape == (n, 3) and bank["p1"].dtype == np.float32 and
                np.isfinite(bank["p1"]).all() and (bank["p1"] >= 0).all() and
                np.allclose(bank["p1"].sum(1), 1, atol=2e-6, rtol=0), "Invalid TAMA target probabilities")
        require(bank["gold_target"].shape == (n,) and bank["gold_target"].dtype.kind in "iu" and
                np.isin(bank["gold_target"], (0, 1, 2)).all(), "Invalid TAMA target labels")
        require(bank["span_valid"].shape == (n,) and bank["span_valid"].dtype == bool,
                "Missing original TAMA span validity")
        ta = bank["gold_target"] == 1
        require(np.array_equal(ta, gold >= 0) and (counts[ta & bank["span_valid"]] == 1).all() and
                not counts[~(ta & bank["span_valid"])].any(), "TAMA needs one valid targeted reference")
        primary &= bank["p1"].argmax(1) == bank["gold_target"]
    for key, expected in (("span_applicable", applicable), ("primary_mask", primary)):
        require(bank[key].dtype == bool and np.array_equal(bank[key], expected), f"Wrong fixed {key}")
    if not features:
        return bank
    h = bank["hidden"]
    require(h.ndim == 3 and h.shape[0] == n and h.shape[1] > 0 and h.shape[2] > 0 and
            h.dtype == np.float32 and np.isfinite(h).all(), "hidden must be finite float32[N,L,d]")
    shape = h.shape[:2]
    require(bank["span_gold"].shape == shape and bank["span_gold"].dtype == np.float32 and
            np.isin(bank["span_gold"], (0, 1)).all(), "Invalid subword gold")
    for key in ("span_mask", "post_mask"):
        require(bank[key].shape == shape and bank[key].dtype == bool, f"Invalid {key}")
    supervised = bank["span_valid"] if ds == "tama" else applicable
    require(np.array_equal(bank["span_mask"], bank["post_mask"] & supervised[:, None]),
            "Corpus supervision mask differs from the frozen protocol")
    require(not bank["span_gold"][~bank["post_mask"]].any(), "Positive gold on non-post subwords")
    if ds == "tama":
        require(not bank["span_gold"][gold < 0].any(), "TAMA non-TA supervision must be negative")
    if ds == "hatexplain":
        wm = bank["word_map"]
        require(wm.shape == shape and wm.dtype.kind in "iu" and
                ((wm >= -1) & (wm < np.diff(offsets)[:, None])).all(), "Invalid official-token word map")
        require(np.array_equal(wm >= 0, bank["post_mask"]), "Post mask differs from word map")
        for i in range(n):
            expected = np.zeros(shape[1], dtype=np.float32)
            visible = wm[i] >= 0
            expected[visible] = bank["gold_primary"][offsets[i] + wm[i, visible]]
            require(np.array_equal(expected, bank["span_gold"][i]), "HateXplain subword gold differs from primary")
    else:
        weights = bank["projection"]
        require(weights.shape == (t, shape[1]) and weights.dtype == np.float32 and
                np.isfinite(weights).all() and (weights >= 0).all(), "Invalid character-overlap projection")
        sums = weights.sum(1)
        require((np.isclose(sums, 0, atol=1e-6) | np.isclose(sums, 1, atol=1e-6)).all(),
                "Projection rows must be normalized or uncovered")
        for i in range(n):
            a, b = offsets[i:i + 2]
            require(not weights[a:b, ~bank["post_mask"][i]].any(), "Projection uses non-post subwords")
    return bank


def load(path, features=True):
    with np.load(path, allow_pickle=False) as archive:
        bank = dict(archive)
    return validate(bank, features=features)


def save(path, bank):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **bank)
