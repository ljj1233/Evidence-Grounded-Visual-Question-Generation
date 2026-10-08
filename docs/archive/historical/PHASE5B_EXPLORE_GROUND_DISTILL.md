# Phase 5B — Explore → Grounded Selection → Distillation

Development1000, greedy, one image → one question, current unified evaluator
(SeeQ `gpt-5.6-luna`, GPT-4o support, BLEU/CIDEr/Dup/Distinct-2). No
final_holdout, no new dataset, no metric change, no reference question / GT
answer / human label / object detector / scene graph / external teacher in any
selector or target. GPT-4o support is used for evaluation only. The selector
uses only the existing frozen UVFC scorer (`sum_j v_j u_j`) and the frozen Flat
verifier (`Qwen2.5-VL-7B-Instruct`). All runs were pinned to cores 0–7
(`taskset -c 0-7`, `OMP/MKL/OPENBLAS/NUMEXPR/RAYON_NUM_THREADS=8`,
`TOKENIZERS_PARALLELISM=false`); the distillation run used 4 DataLoader workers.

## 1. Direct-4 matched control (5B-A)

Same SFT checkpoint, N=4, T=0.8, top-p 0.9, seed 42, max_new_tokens 64, sdpa,
and the identical S0 selector as Phase 5A Explore-4; the only change is the
plain direct prompt instead of the exploration prompt.

| | Search | Coverage | Context | Spatial | Supported | Unsupported |
|---|---:|---:|---:|---:|---:|---:|
| Explore-4 + UVFC (S0) | 0.355 | 0.373 | 0.267 | 0.068 | 0.784 | 0.068 |
| Direct-4 + UVFC (S0) | 0.358 | 0.381 | 0.268 | 0.071 | 0.782 | 0.083 |

**Q1: Direct-4 + UVFC matches (or slightly exceeds) Explore-4 + UVFC on
Search/Coverage and Supported.** The exploration instruction is *not* the source
of the gain — generic best-of-4 sampling + UVFC selection explains it. The
exploration prompt cannot be claimed as the core innovation.

## 2. Selector ablation (5B-B)

All selectors use the identical Explore-4 candidate pool (3829 candidates /
1000 images) and the frozen verifier. `support_floor = min_j v_j` is the weakest
requirement-level support of the existing Flat verifier
(`presuppositions` / `reference_constraints` / `query_target`, mapped
supported/clear/answerable→1, uncertain/ambiguous→0.5,
unsupported/unanswerable→0). No new requirement extractor.

| | Search | Coverage | Context | Spatial | Supported | Unsupported | BLEU | CIDEr | Dup | Dist-2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT | 0.275 | 0.260 | 0.270 | 0.073 | 0.725 | 0.159 | 0.317 | 0.455 | 0.570 | 0.140 |
| Static UVFC | 0.332 | 0.332 | 0.255 | 0.071 | 0.814 | 0.094 | 0.326 | 0.519 | 0.499 | 0.174 |
| Direct-4 + UVFC S0 | 0.358 | 0.381 | 0.268 | 0.071 | 0.782 | 0.083 | 0.184 | 0.318 | 0.151 | 0.311 |
| Explore-4 + UVFC S0 | 0.355 | 0.373 | 0.267 | 0.068 | 0.784 | 0.068 | 0.183 | 0.290 | 0.142 | 0.308 |
| Explore-4 + SupportFirst S1 | 0.345 | 0.366 | 0.263 | 0.063 | 0.809 | 0.064 | 0.190 | 0.301 | 0.148 | 0.298 |
| Explore-4 + SupportTop2+UVFC S2 | 0.342 | 0.363 | 0.270 | 0.066 | 0.781 | 0.071 | 0.200 | 0.307 | 0.156 | 0.301 |
| Explore-Ground Distilled (5B-D) | 0.346 | 0.354 | 0.241 | 0.057 | 0.889 | 0.039 | 0.318 | 0.527 | 0.489 | 0.165 |
| Explore-Ground Distilled + UVFC (5B-E, step 133) | **0.366** | **0.371** | 0.245 | 0.062 | **0.917** | **0.032** | 0.302 | 0.516 | 0.478 | 0.165 |
| Prompt-CoT | 0.503 | 0.467 | 0.249 | 0.223 | 0.883 | 0.069 | 0.074 | 0.073 | 0.021 | 0.263 |
| GPT-5.6-Luna | 0.421 | 0.445 | 0.271 | 0.257 | 0.930 | 0.036 | 0.080 | 0.119 | 0.006 | 0.352 |

Selector audits (selected index distribution, violation/failed rate, support_floor, UVFC, change vs S0):

| Selector | idx dist (0/1/2/3) | viol | failed | mean floor | P10/P25/med floor | mean UVFC | changed vs S0 |
|---|---|---:|---:|---:|---:|---:|---:|
| S0 | 272/256/230/242 | 0.022 | 0.034 | 0.881 | 0.5/1.0/1.0 | 0.715 | 0.000 |
| S1 | 274/255/228/243 | 0.022 | 0.005 | 0.973 | 1.0/1.0/1.0 | 0.679 | 0.098 |
| S2 | 279/255/227/239 | 0.022 | 0.019 | 0.929 | 1.0/1.0/1.0 | 0.644 | 0.224 |
| Direct-4 S0 | 264/275/237/224 | 0.032 | 0.036 | 0.870 | 0.5/1.0/1.0 | 0.718 | 0.000 |

**Q2: Yes.** SupportFirst (S1) narrows the Supported gap (0.784→0.809) and cuts
the failed-parse rate (0.034→0.005) while retaining Coverage (0.373→0.366) and
Search (0.355→0.345). S2 is too conservative (Supported 0.781, UVFC 0.644).

## 3. Teacher selection (Section 6 gate)

Gate: Search ≥ 0.345, Coverage ≥ 0.360, Supported ≥ 0.800, Unsupported ≤ 0.080,
and Search/Coverage > Static.

| selector | Search | Coverage | Supported | Unsupported | passes |
|---|---:|---:|---:|---:|:--:|
| S0 (Explore-4 + UVFC) | 0.355 | 0.373 | 0.784 | 0.068 | no (Supported) |
| S1 (SupportFirst) | **0.345** | **0.366** | **0.809** | **0.064** | **yes** |
| S2 (SupportTop2) | 0.342 | 0.363 | 0.781 | 0.071 | no (Search, Supported) |

**Teacher = Explore-4 + SupportFirst (S1).** (S1's raw Search is 0.3452; a single
failed SeeQ row was retried to reach full 1000/1000 coverage.)

## 4. Teacher generation (5B-C)

Frozen manifest `outputs/phase4b_online_uvfc/data/train_10000.jsonl`
(SHA256 `c884ed36…a67adc2`, 10000 records, 9184 unique images) — not regenerated.
For each unique image: 4 exploration-prompt samples (T=0.8, seed 42), scored with
the frozen UVFC + Flat verifier, selected with S1.

- 35220 candidates / 9184 images (mean 3.83); S1 selected index 0/1/2/3 = 2505/2365/2387/1927.
- selected confirmed-violation rate 0.026, failed 0.003, mean support_floor 0.975, mean UVFC 0.689.
- 5444 unique selected questions.
- `outputs/phase5b_teacher_explore_ground/train_10000.jsonl` (full teacher artifact)
  and `student_train_10000.jsonl` (`{image, selected_question}` only).

## 5. Distillation (5B-D)

Original VQG SFT recipe (LoRA r16/α32/dropout 0.05, 7 modules, lr 2e-4, cosine,
batch 2×8=16, warmup 0.03, gradient checkpointing, flash-attention-2, seed 42),
init = Final SFT adapter, `num_train_epochs=1` (recipe otherwise unchanged; no
development1000 tuning). 625 steps, loss 0.660→0.589, peak 19.1 GB, 252 LoRA
modules, 43.6M trainable.

**Q3: Yes.** The distilled direct policy (greedy, Image → Question) retains the
teacher's Search/Coverage (0.346 / 0.354) and *improves* grounding over both the
teacher and Static UVFC: Supported **0.889** (vs teacher 0.809, Static 0.814),
Unsupported **0.039** (vs 0.064 / 0.094). It beats Static UVFC on all four
headline metrics. Greedy exact-match with the teacher is only 6.8%, i.e. the
student learned the teacher *distribution*, not verbatim targets.

Section 10 gate (Search>0.300, Coverage>0.300, Supported≥0.74, Unsupported≤0.14)
is satisfied, so 5B-E is triggered.

## 6. GRPO from the distilled init (5B-E)

Exact Static UVFC config (`configs/phase5b/explore_ground_uvfc.yaml`): same
manifest/ordering/seed/UVFC/anchor/optimizer/rollouts/decoding/LoRA; only the
init checkpoint changes (Final SFT → Explore-Ground Distilled).

The run self-terminated at **step 133/1000** under the unchanged Static guard
(`stop_reason = DEGENERATION`: duplicate_rate 0.344 − initial 0.125 > 0.2; KL max
0.003; malformed 0). The distilled init starts more duplicated than the SFT init
(0.125 vs 0.0625), so the SFT-calibrated duplicate guard fires earlier — the same
class of init-induced instability seen for the raw-base init in Phase 4B.2. The
guard was **not** overridden.

Evaluation of the reached step-133 checkpoint (development1000 greedy):

| | Search | Coverage | Context | Spatial | Supported | Unsupported | BLEU | CIDEr | Dup | Dist-2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Static UVFC | 0.332 | 0.332 | 0.255 | 0.071 | 0.814 | 0.094 | 0.326 | 0.519 | 0.499 | 0.174 |
| **Explore-Ground Distilled + UVFC** | **0.366** | **0.371** | 0.245 | 0.062 | **0.917** | **0.032** | 0.302 | 0.516 | 0.478 | 0.165 |

Deltas vs Static UVFC: Search **+0.034**, Coverage **+0.039**, Supported
**+0.103**, Unsupported **−0.062**. This meets the Section 11 ideal target
(Search ≥ 0.35, Coverage ≥ 0.36, Supported ≥ 0.80, Unsupported ≤ 0.09).

**Q4: Yes.** Exploration-aware initialization places the policy in a broader,
more grounded question distribution and subsequent UVFC optimization strengthens
it: the distilled-init run dominates Static UVFC on all four headline metrics.
Caveat: this is a step-133 checkpoint (guard stop), not a 1000-step model; the
init's higher duplication makes the frozen guard fire early, so the improvement
is achieved before the guard rather than at 1000 steps.

## 7. Qualitative cases

`image | SFT | Static | Teacher S1 | Distilled`:

- `100234` | What is the weather like? | Is this a harbor? | Is the sky clear? | Is there a boat in the water?
- `100990` | What is the man doing? | What is the giant object in the background? | Is the donut giant? | What is the man in the blue sweater doing?
- `102503` | What color is the bed? | What color is the bed? | How many pillows are on the bed? | What color is the chair?

The teacher frequently selects short yes/no probes ("Is the sky clear?", "Is the
donut giant?"); the distilled student produces grounded WH-questions with
specific visual evidence, consistent with its higher Supported and lower
Unsupported.

## 8. Supported-gap analysis

Static UVFC: Supported 0.814 / Unsupported 0.094. The exploration pool is much
less grounded (Explore-4 candidate-0 Supported 0.692 / Unsupported 0.152).
SupportFirst closes most of the gap by ranking on the weakest requirement
(mean support_floor 0.973; failed-parse rate 0.005) at a small coverage cost.
Distillation then *over*-corrects: Supported 0.889 / Unsupported 0.039, above
Static. The remaining gap is therefore not a verifier-accuracy problem; it is a
selection + training-distribution effect.

## 9. Compute / API cost

| stage | cost |
|---|---|
| Direct-4 pool (1000×4) | ~3.5 min GPU |
| Direct-4 UVFC+Flat scoring (3811) | ~35 min GPU |
| Explore-4 requirement-level scoring (3829) | ~35 min GPU |
| Teacher pool generation (9184×4) | ~33 min GPU |
| Teacher pool UVFC+Flat scoring (35220) | ~6 h GPU |
| Distillation (625 steps) | ~50 min GPU |
| 5B-E GRPO (guard stop at step 133) | ~51 min GPU |
| evaluations (6 runs, 1000/1000 SeeQ + support) | ~6 min API each |

## 10. Candidate count / inference overhead

The teacher/selector path needs, per image: 4 generations + 4 UVFC forwards
(8B) + 4 Flat-verifier forwards (7B) ≈ 9 model passes and an extra 7B model in
memory. The distilled student is a single greedy `Image → Question` pass with no
candidate pool, no verifier and no API call — a ~9× inference reduction at
higher Supported than the teacher.

## 11. Paper-level interpretation

- The Phase-5A coverage gain is **not** due to an explicit exploration
  instruction; it is generic best-of-N sampling + UVFC selection (Q1).
- A support-aware selector using only existing verifier information closes the
  grounding gap while retaining coverage (Q2).
- That inference-time behavior **does distill into a direct policy**, which
  matches Static UVFC coverage and exceeds it on grounding (Q3). Because Q3 is
  positive, the method is a **training method**, not merely inference-time
  search.
- Q4 is **positive**: distilled-init + Static UVFC dominates Static UVFC on all
  four headline metrics, so the full recipe is a training-method improvement,
  not merely inference-time search.
- The binding constraint is now the frozen degeneration guard calibrated on SFT:
  the distilled init is more duplicated, so the run self-stops at step 133. The
  improvement is real but is achieved before the guard, not at 1000 steps.

## 12. Recommendation

**Continue the training-method direction (Explore → Grounded Selection →
Distillation → UVFC), with two concrete fixes:**

1. **Drop the exploration prompt.** The matched Direct-4 + UVFC control equals
   or exceeds Explore-4 + UVFC (Q1), so the prompt is unnecessary; Direct-4 is
   cheaper and simpler. Re-run teacher generation with the direct prompt.
2. **Address the init-induced guard stop.** The distilled init trips the
   SFT-calibrated duplicate guard at step 133 (init dup 0.125). A guard expressed
   relative to the init's own distribution (as recommended for base init in
   Phase 4B) would let the 1000-step recipe be measured; the current numbers are
   a step-133 checkpoint.

Interim headline: **Explore-Ground Distilled + UVFC (step 133) is the best
tested local method on the four headline axes** — Search 0.366 / Coverage 0.371 /
Supported 0.917 / Unsupported 0.032, above Static UVFC on all four and within the
Section 11 ideal target, without exceeding Prompt-CoT's Search/Coverage.

No further phase was started.
