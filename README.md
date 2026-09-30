# Paired conditional evidence readouts

This is the minimal source release for the paper's matched readout experiment: fit equal-parameter multiplicative and additive span heads on frozen encoder states and category probabilities, then compare correction and existing-reference effects. It does **not** train the base encoder/category model or reproduce the whole original training campaign from raw corpora.

## Run

Python 3.11 is the tested environment. Install the two runtime dependencies in your chosen environment:

```bash
python3 -m pip install -r requirements.txt
bash run.sh pilot
```

For an existing environment, `PYTHON=/path/to/python bash run.sh pilot` selects its interpreter. The pilot check needs only a CPU and downloads nothing. Choose the appropriate PyTorch wheel for your machine; the original campaign used PyTorch 2.9.1 with CUDA 12.8 and NumPy 1.26.4.

For each dataset and each seed 17, 29, 43, 59, export the external feature banks described below, then run:

```bash
bash run.sh fit --train /external/train.npz --dev /external/dev.npz --out /external/readouts --device cpu
bash run.sh score --bank /external/test.npz --heads /external/readouts --out /external/test_scores.npz --device cpu
bash run.sh analyze --banks /external/test_scores_s17.npz /external/test_scores_s29.npz /external/test_scores_s43.npz /external/test_scores_s59.npz --out /external/paired_effects.csv
```

Use unique output paths for every dataset/seed. Existing outputs are refused. `--device cuda:0` also works when you have allocated a GPU. `analyze` accepts one or more corpora but requires all four seeds for every supplied corpus. Training writes the two heads and development history; scoring writes numeric output banks; analysis writes the estimate table. These are runtime artifacts outside the source release.

## External input connector

`bank.make_bank(...)` creates and validates a NumPy dictionary; `bank.save(path, dictionary)` writes the external NPZ consumed by the CLI. Loading always uses `allow_pickle=False`. Every bank contains all records in its partition, including normal or missing-reference records. Keep record order fixed. Bind train/dev/test to the SHA-256 of the same original fitted model with `bank.sha256(checkpoint_path)`.

Extract features with the original trained encoder and category predictor in evaluation mode, without gradients, with their original tokenizer, prefix, attention mask, precision and alignment. For the original TAMA `Cascade.forward`, the relevant outputs are:

```python
model.eval().requires_grad_(False)
with torch.no_grad():
    hidden, pooled, target_logits, type_logits, original_span_logits = model(input_ids, attention_mask)
    h = hidden.cpu().numpy()                       # float32, no mean pooling here
    p = type_logits.softmax(-1).cpu().numpy()
    p1 = target_logits.softmax(-1).cpu().numpy()
```

The external HateXplain/PLEAD `CategorySpan.forward` returns `(hidden, category_logits, original_span_logits)`; use `category_logits.softmax(-1)` for `p`. Concatenate batches in input-record order. Confirm that the original category probabilities and original span predictions reproduce your saved outputs before exporting features. This repository cannot attest to an external encoder or recover its tokenizer/labels from hidden states.

Call the connector from your existing data/model export code:

```python
from bank import make_bank, save, sha256

features = make_bank(
    dataset=dataset, split=split, seed=seed,
    model_sha256=sha256(original_checkpoint),
    ids=record_ids, groups=exact_text_group_ids,
    hidden=frozen_hidden, p=native_category_probabilities,
    gold_class=existing_category_indices,
    token_lengths=evaluation_token_counts,
    references=existing_reference_masks_by_post,
    span_gold=existing_subword_targets, span_mask=existing_supervision_mask,
    post_mask=original_post_subword_mask,
    # HateXplain: word_map=official_word_ids.
    # TAMA/PLEAD: projection=concatenated_character_overlap_matrices.
    # TAMA also requires p1=native_target_probabilities,
    # gold_target=existing_target_indices, span_valid=original_span_validity.
)
save(output_path, features)
```

The commented dataset-specific arguments must be supplied for the chosen corpus. Unsupported/missing fields fail validation. `references[i]` is a sequence of existing binary evaluation-token masks for post `i`: an empty sequence means unavailable, while a contained all-zero vector is a valid empty reference. Do not fill missing references with zeros. The connector derives primary/strict/union criteria and original type-error membership; you cannot supply a new error subset selected from the refitted heads.

| Input field | Shape and dtype | Meaning |
|---|---|---|
| `dataset`, `split`, `seed`, `model_sha256` | scalar strings; scalar integer seed | Corpus; train/dev/test; fitted-model identity |
| `ids`, `groups` | N strings | Opaque unique IDs; exact-text groups |
| `hidden` | float32[N,L,d] | Frozen per-subword encoder states, including padded positions |
| `p` | float32[N,C] | Original category softmax probabilities in the fixed axis below |
| `gold_class` | integer[N] | Existing category index; −1 only for TAMA non-TA records |
| `token_lengths` | N positive integers | Full evaluation-token counts, including tokens beyond truncation |
| `references` | N sequences of binary vectors | Existing masks, each of its post's full token length |
| `span_gold` | float32[N,L], binary | Original subword training targets |
| `span_mask`, `post_mask` | bool[N,L] | Supervised subwords; original-post nonstructural subwords |
| `word_map` (HateXplain) | integer[N,L] | Released token index per subword; −1 for special/padding |
| `projection` (TAMA/PLEAD) | float32[T,L] | Concatenated per-post character-overlap matrices; T=sum(token_lengths) |
| `p1` (TAMA) | float32[N,3] | Original target judgment probabilities |
| `gold_target`, `span_valid` (TAMA) | integer[N], bool[N] | Existing target class and original span-validity flag |

Serialized banks store `group` rather than the connector argument `groups`; cumulative `offsets` rather than `token_lengths`; flattened `individual_gold`, `individual_record` and `individual_offsets` rather than nested `references`. They additionally store `gold_primary`, `gold_strict`, `gold_union`, `span_applicable` and `primary_mask`. The validation code is the executable schema.

`bank.exact_group(original_text, dataset)` hashes exact original text for TAMA/PLEAD; for HateXplain pass the exact released token list. It performs no normalization. You can also retain your existing exact-text group identifiers. Matching the original numerical bootstrap draws requires preserving those identifiers and their sorting order, as well as record identities; changing group names preserves the resampling design but can change a finite set of draws. Train/dev exact-group overlap is rejected.

## Corpus semantics

The fixed category axes are defined in `bank.CLASSES`. TAMA uses its twelve types in the original order and target indices `non-abusive=0`, `targeted-abusive=1`, `unidentified-targets=2`. HateXplain uses `(hatespeech, normal, offensive)`; PLEAD uses `(comparison, derogation, threatening, hatecrime, nothate)`. Retain the original label conversion `animosity → derogation` for PLEAD.

- **TAMA:** Input is `[TGT] {existing_target} [/TGT] {original_post}`. Original length is 128, width 768. Evaluation tokens use `@[A-Za-z0-9_]+|\w+|[^\w\s]`. Subword targets indicate overlap with the existing valid character spans, not expanded whole-token spans. Every valid original-post subword is supervised, including negative labels on non-TA posts; the original invalid-span record has no span supervision. Supply exactly one reference for each valid gold-TA post, none otherwise. The native error set requires a correct native target judgment and a wrong native type. Preserve the externally supplied frozen record partitions; do not derive them from any relationship data.
- **HateXplain:** Use the post alone, length 128, width 768, strict-majority existing category labels and released token sequences. Remove exact-token overlaps from earlier partitions while retaining test membership. `word_map` comes from the original fast tokenizer's `word_ids()` with `is_split_into_words=True`. Subword targets copy the primary mask of their released token. Normal posts and posts without valid references have no span loss. Visible subword scores are accumulated in float64 and averaged per official token, then cast to float32, matching the source implementation. Uncovered tokens remain zero. Rationale-vector positions are not paired with category-annotator positions.
- **PLEAD:** Use the original single-post text alone, length 256, width 768, and authors' fixed partitions. Form one existing reference per annotator copy after unioning category-specific content slots across that copy's target rows. Target, protected-characteristic and stance slots are not evidence. Omit an invalid copy, retain valid siblings and explicit empty slots. Keep the authors' `annotation_fix`/`locate_span` preprocessing external; do not replace it with substring matching. Evaluation words are `\w+`. Subword targets overlap primary-positive evaluation words. Only valid positive posts supply span loss. Neither normal placeholders nor new judgments become evidence references.

For both character-aligned corpora, pass the original float32 projection weights or construct them with `bank.character_projection(evaluation_token_spans, post_relative_subword_spans)`. Structural/padding offsets must have zero length. The helper computes normalized intersection-character weights; it does not tokenize, locate evidence, add context or infer annotations. Preserve full references beyond encoder truncation; their uncovered projected predictions remain zero. The connector checks normalization and masks, but external character-offset correctness remains the caller's responsibility.

## Method and outputs

The two heads are `sigmoid(wᵀ[h ⊙ tanh(Wp)] + b)` and `sigmoid(wᵀtanh(h + Wp) + b)`. Each has `d*C+d+1` trainable parameters. Both start from identical parameter arrays and use the same frozen features, probabilities, supervision masks and `default_rng(seed+epoch)` minibatch order. Training uses batch 16, AdamW at .001 with weight decay .01 and gradient clip 1, for 30/8/16 epochs on TAMA/HateXplain/PLEAD. Each head selects its earliest strict maximum of development applicable-post native span F1. Base-model probabilities never update.

Scoring retains each head's own native prediction, all one-hot outputs, and a swap of the native wrong argmax probability with the existing reference-class probability on the original eligible errors. It preserves inactive inputs and the probability multiset, including ties. Stress enumerates only 12/2/4 evidence-bearing categories; external normal categories are excluded from the stress set. The decision threshold is always .5.

F1 is computed per reference, averaged within post, then across posts. Empty prediction with empty valid reference scores 1. Primary requires at least half the available references; strict requires greater than half; union requires any positive reference. With two references, primary is union. `individual_mean` averages separately scored references; it does not average binary masks. TAMA has only its original primary reference.

The output CSV includes each seed and `four_seed_mean`, absolute quality and within-post `reference_minus_primary` changes, for both heads and `additive_minus_multiplicative`. `g_native` is reference-one-hot minus native; `g_identity` is reference-one-hot minus predicted-one-hot; `hardening` is predicted-one-hot minus native; `g_swap` is swapped minus native. `stress_d` is native F1 minus the per-post minimum endpoint F1; `stress_r` is the endpoint range. Signed TP/FP/empty-reference contributions and fixed-prediction reference geometry are also retained.

For `analysis=reference_minus_primary`, `head=additive_minus_multiplicative`, `metric=g_swap`, the estimate is the correction-specific interaction: the reference-induced change in the swapped quality gap minus the reference-induced change in the native gap. The fixed type-error cohort is identical across heads/reference criteria within each seed. Missed-positive external posts receive one-hot sensitivity scores separately; swaps are not evaluated as a missed-abuse correction. Full-applicable native quality is reported separately from type-error quality.

All F1 values are fractions in [0,1]; gains, stress differences and interactions are signed fractions. Multiply by 100 for the paper's percentage points. CIs use 2,000 shared exact-group bootstrap draws from the full applicable frame, seed 20260923, with record-weighted subgroup ratios. Four-seed draws average the four seed-specific estimates at each shared draw before taking quantiles. Empty subgroup draws remain missing. Reported seed SD uses `ddof=1`. These intervals condition on fitted models and original error membership; they exclude retraining/annotation uncertainty.

## Scope and provenance

The release extracts the method from `B/code/b_interface.py`, `b_interface_analysis.py`, `b_reference_accounting.py`, `b_reference_pairing.py`, `b_missed_posts.py`, the extension's `b_analysis.py`, and `single_post_exact_group_audit.Frame`. It removes workspace paths, old-result checksums, historical run dependencies and campaign orchestration. It does not distribute or recreate the original base fits, pooled baselines, per-corpus raw-data adapters, PLEAD alignment-uniqueness appendix analysis, plots or full manuscript tables.

The external base fits used XLM-R-base revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`, float32/eager attention, and the original tokenizer. Obtain the model separately from [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base/tree/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089). Encoder extraction may use its original environment (Transformers 4.57.2, Tokenizers 0.22.1); these are not dependencies of this feature-bank method.

Dataset attribution: [TAMA](https://aclanthology.org/2026.acl-long.811/); [HateXplain](https://github.com/hate-alert/HateXplain/tree/01d742279dac941981f53806154481c0e15ee686), Mathew et al., AAAI 2021; [PLEAD](https://github.com/Ago3/PLEAD/tree/19ef1d0d3b015528ea25e9f91ba502040a6727f8), Calabrese, Ross and Lapata, TACL 2022, and its [official HF dataset](https://huggingface.co/datasets/agostina3/PLEAD/tree/43af9bcaf48f558cf90d026d5caea57222fe2da5). Obtain any external inputs under their respective terms. No dataset or upstream locator implementation is bundled, and this repository adds no project license.
