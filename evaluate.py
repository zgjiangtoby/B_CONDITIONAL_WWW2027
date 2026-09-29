"""Fixed-fit correction/reference contrasts, paired over posts, heads and seeds."""
import numpy as np

from bank import CLASSES, POSITIVE, SEEDS, require, validate

REPS, BOOTSTRAP_SEED = 2000, 20260923
HEADS = ("multiplicative", "additive")


def positive_f1(gold, predicted):
    gold, predicted = np.asarray(gold, bool), np.asarray(predicted, bool)
    require(gold.shape == predicted.shape and gold.ndim == 1, "Mismatched F1 masks")
    denominator = int(gold.sum() + predicted.sum())
    return float(2 * (gold & predicted).sum() / denominator) if denominator else 1.


def swap_probabilities(p, gold, eligible):
    pred = p.argmax(1)
    active = eligible & (pred != gold)
    unique = (p == p.max(1, keepdims=True)).sum(1) == 1
    out = p.copy()
    i = np.flatnonzero(active)
    out[i, pred[i]], out[i, gold[i]] = p[i, gold[i]], p[i, pred[i]]
    require(np.array_equal(np.sort(out, axis=1), np.sort(p, axis=1)), "Swap changed confidence multiset")
    require((out.argmax(1)[active & unique] == gold[active & unique]).all(), "Swap failed on unique maximum")
    return out, active, unique


def references(bank, i):
    start, stop = bank["offsets"][i:i + 2]
    refs = {"primary": [bank["gold_primary"][start:stop]]}
    if str(bank["dataset"]) != "tama":
        refs.update({name: [bank["gold_" + name][start:stop]] for name in ("strict", "union")})
        owners, offsets = bank["individual_record"], bank["individual_offsets"]
        refs["individual_mean"] = [bank["individual_gold"][offsets[j]:offsets[j + 1]]
                                   for j in np.flatnonzero(owners == i)]
        require(bool(refs["individual_mean"]), "Missing reference on an applicable post")
    return refs


def signed_counts(gold, before, after):
    gold, before, after = (np.asarray(v, bool) for v in (gold, before, after))
    q, qp = positive_f1(gold, before), positive_f1(gold, after)
    m, k = int(gold.sum()), int(after.sum())
    dt = int((gold & after).sum()) - int((gold & before).sum())
    df = int((~gold & after).sum()) - int((~gold & before).sum())
    tp = (2 - q) * dt / (m + k) if m else 0.
    fp = -q * df / (m + k) if m else 0.
    empty = qp - q if not m else 0.
    require(abs(tp + fp + empty - (qp - q)) < 1e-12, "Signed F1 identity failed")
    return dict(tp_term=tp, fp_term=fp, empty_term=empty, delta_tp=float(dt), delta_fp=float(df))


def geometry(primary, reference, predicted):
    primary, reference, predicted = (np.asarray(x, bool) for x in (primary, reference, predicted))
    return dict(reference_tokens=float(reference.sum()), predicted_tokens=float(predicted.sum()),
                true_positive=float((reference & predicted).sum()),
                added_hits=float((~primary & reference & predicted).sum()),
                removed_hits=float((primary & ~reference & predicted).sum()),
                reference_shift=positive_f1(reference, predicted) - positive_f1(primary, predicted))


def validate_scores(bank):
    validate(bank, features=False)
    t, c = int(bank["offsets"][-1]), len(CLASSES[str(bank["dataset"])])
    inactive = np.repeat(~bank["primary_mask"], np.diff(bank["offsets"]))
    for head in HEADS:
        for branch, shape in (("native", (t,)), ("swap", (t,)), ("onehot", (c, t))):
            v = bank[head + "_" + branch]
            require(v.shape == shape and np.isfinite(v).all() and ((v >= 0) & (v <= 1)).all(),
                    f"Invalid {head}/{branch} scores")
        require(np.array_equal(bank[head + "_native"][inactive], bank[head + "_swap"][inactive]),
                "Inactive posts changed under swap")
    return bank


def record_effects(bank, head):
    rows = []
    _, _, unique = swap_probabilities(bank["p"], bank["gold_class"], bank["primary_mask"])
    for i in np.flatnonzero(bank["span_applicable"]):
        a, b = bank["offsets"][i:i + 2]
        pred, gold = int(bank["p"][i].argmax()), int(bank["gold_class"][i])
        native = bank[head + "_native"][a:b] >= .5
        endpoints = bank[head + "_onehot"][:, a:b] >= .5
        swap = bank[head + "_swap"][a:b] >= .5
        for ref, masks in references(bank, i).items():
            q, hard, corrected, qs = np.mean([
                [positive_f1(mask, v) for v in (native, endpoints[pred], endpoints[gold], swap)]
                for mask in masks], axis=0)
            endpoint_q = np.mean([[positive_f1(mask, endpoints[j]) for j in POSITIVE[str(bank["dataset"])]]
                                  for mask in masks], axis=0)
            row = dict(id=str(bank["ids"][i]), reference=ref, native_q=float(q),
                       predicted_onehot_q=float(hard), reference_onehot_q=float(corrected), swap_q=float(qs),
                       g_native=float(corrected - q), g_identity=float(corrected - hard),
                       hardening=float(hard - q), g_swap=float(qs - q),
                       stress_d=float(q - endpoint_q.min()), stress_r=float(endpoint_q.max() - endpoint_q.min()),
                       native_empty=float(not native.any()), argmax_tie=float(not unique[i]))
            require(abs(row["g_native"] - row["g_identity"] - row["hardening"]) < 1e-12, "Gain decomposition failed")
            for op, prediction in (("onehot", endpoints[gold]), ("swap", swap)):
                values = [signed_counts(mask, native, prediction) for mask in masks]
                row.update({op + "_" + k: float(np.mean([r[k] for r in values])) for k in values[0]})
            for op, prediction in (("native", native), ("onehot", endpoints[gold]), ("swap", swap)):
                values = [geometry(bank["gold_primary"][a:b], mask, prediction) for mask in masks]
                row.update({op + "_" + k: float(np.mean([r[k] for r in values])) for k in values[0]})
            rows.append(row)
    return rows


def bounds(samples):
    good = np.isfinite(samples)
    values = samples[good]
    lo, hi = np.quantile(values, [.025, .975]) if len(values) else (None, None)
    return dict(lo=float(lo) if lo is not None else None, hi=float(hi) if hi is not None else None,
                valid_replicates=int(good.sum()), empty_replicates=int((~good).sum()))


class Frame:
    """Same full-applicable-frame draws as the source experiment, including empty subgroup draws."""
    def __init__(self, name, ids, text_groups, reps=REPS):
        self.name, self.ids = name, sorted(ids)
        require(len(self.ids) == len(set(self.ids)) and bool(self.ids), "Invalid bootstrap frame")
        self.groups, self.inverse = np.unique([text_groups[k] for k in self.ids], return_inverse=True)
        self.weights = np.random.default_rng(BOOTSTRAP_SEED).multinomial(
            len(self.groups), np.ones(len(self.groups)) / len(self.groups), size=reps)

    def sample_sums(self, values):
        values = np.asarray(values, dtype=float)
        require(values.shape[0] == len(self.ids) and np.isfinite(values).all(), "Invalid bootstrap values")
        grouped = np.zeros((len(self.groups),) + values.shape[1:])
        np.add.at(grouped, self.inverse, values)
        return (self.weights @ grouped.reshape(len(self.groups), -1)).reshape(
            (len(self.weights),) + values.shape[1:])

    def means(self, values, mask):
        sums = self.sample_sums(np.column_stack((mask, values * mask[:, None])))
        samples = np.divide(sums[:, 1:], sums[:, :1], out=np.full_like(sums[:, 1:], np.nan),
                            where=sums[:, :1] > 0)
        points = values[mask].mean(0) if mask.any() else np.full(values.shape[1], np.nan)
        return points, samples


def paired_results(banks):
    """One or more corpora, each with all four independently fitted seed banks."""
    by = {}
    for bank in banks:
        validate_scores(bank)
        key = str(bank["dataset"]), int(bank["seed"])
        require(str(bank["split"]) == "test" and key not in by, "Expected one test bank per corpus/seed")
        by[key] = bank
    require(bool(by), "No score banks")
    for ds in {key[0] for key in by}:
        require({s for d, s in by if d == ds} == set(SEEDS), "Four seeds are required per corpus")
        first = by[ds, SEEDS[0]]
        for seed in SEEDS[1:]:
            for field in ("ids", "group", "offsets", "gold_class", "span_applicable", "gold_primary",
                          "gold_strict", "gold_union", "individual_record", "individual_offsets", "individual_gold"):
                require(np.array_equal(first[field], by[ds, seed][field]), f"Unmatched seed population/reference: {field}")
    rows, draws = [], {}
    for (ds, seed), bank in sorted(by.items()):
        indices = np.flatnonzero(bank["span_applicable"])
        frame = Frame(f"{ds}_test_all_span_applicable", bank["ids"][indices].tolist(),
                      dict(zip(bank["ids"], bank["group"])))
        index = {str(v): i for i, v in enumerate(bank["ids"])}
        order = np.array([index[rid] for rid in frame.ids])
        typed = bank["primary_mask"][order]
        missed = ~np.isin(bank["p"][order].argmax(1), POSITIVE[ds])
        effects = {h: {(r["reference"], r["id"]): r for r in record_effects(bank, h)} for h in HEADS}
        refs = sorted({key[0] for key in effects[HEADS[0]]})
        metrics = [k for k in next(iter(effects[HEADS[0]].values())) if k not in ("id", "reference")]
        values = {}
        for ref in refs:
            for head in HEADS:
                values[head, ref] = np.array([[effects[head][ref, rid][m] for m in metrics] for rid in frame.ids])
            values["additive_minus_multiplicative", ref] = values["additive", ref] - values["multiplicative", ref]

        def emit(analysis, head, ref, cohort, matrix, mask, names=metrics):
            point, samples = frame.means(matrix, mask)
            for j, metric in enumerate(names):
                key = ds, analysis, head, ref, cohort, metric
                row = dict(dataset=ds, seed=seed, analysis=analysis, head=head, reference=ref,
                           cohort=cohort, metric=metric, estimate=float(point[j]) if np.isfinite(point[j]) else None,
                           n=int(mask.sum()), groups=len(np.unique(frame.inverse[mask])),
                           bootstrap_pool_n=len(frame.ids), bootstrap_pool_groups=len(frame.groups),
                           **bounds(samples[:, j]))
                rows.append(row)
                draws.setdefault(key, {})[seed] = row, samples[:, j]

        for ref in refs:
            for head in (*HEADS, "additive_minus_multiplicative"):
                v = values[head, ref]
                emit("absolute_quality", head, ref, "type_errors", v, typed)
                native_names = ("native_q", "native_empty")
                emit("absolute_quality", head, ref, "all_applicable", v[:, [metrics.index(m) for m in native_names]],
                     np.ones(len(order), bool), native_names)
                if ds != "tama":
                    names = ("native_q", "reference_onehot_q", "g_native")
                    emit("absolute_quality", head, ref, "missed_positive", v[:, [metrics.index(m) for m in names]],
                         missed, names)
                if ref != "primary":
                    emit("reference_minus_primary", head, ref, "type_errors", v - values[head, "primary"], typed)
    for key, per_seed in draws.items():
        items = [per_seed[s] for s in SEEDS]
        points = [item[0]["estimate"] for item in items]
        valid = all(v is not None for v in points)
        ds, analysis, head, ref, cohort, metric = key
        rows.append(dict(dataset=ds, seed="four_seed_mean", analysis=analysis, head=head, reference=ref,
                         cohort=cohort, metric=metric, estimate=float(np.mean(points)) if valid else None,
                         seed_sd=float(np.std(points, ddof=1)) if valid else None,
                         n_by_seed=str([item[0]["n"] for item in items]),
                         **bounds(np.stack([item[1] for item in items]).mean(0))))
    return rows
