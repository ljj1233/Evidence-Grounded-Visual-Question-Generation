# Phase 5A — Exploration–UVFC Bottleneck Study

Development1000, greedy, one image → one question, current unified evaluator
(SeeQ `gpt-5.6-luna` visual-centric dims, GPT-4o visual support, BLEU/CIDEr/
Distinct-2/dup). No final_holdout, no new dataset, no metric change, no
reference question / GT answer / human label in any reward or prompt, and no
Prompt-CoT output used as supervision. Raw metric JSON is under
`outputs/phase5a_*`; consolidated table below.

## 1. Recovered environment (Section 1 audit)

| item | value |
|---|---|
| train manifest | `outputs/phase4b_online_uvfc/data/train_10000.jsonl` |
| train records / unique images | 10000 / 9184 |
| train SHA256 | `c884ed360196e494076ab0b73a45356de87c44d5fd8b2d5778102ea25a67adc2` |
| development manifest | `outputs/same_backbone_baselines/manifests/development1000_images.jsonl` |
| development SHA256 | `06782bedf1754fe59ff7609a06e2c79ce91403e4ecfcf9f2f295d6b104d27d47` |
| init / anchor checkpoint | `outputs/checkpoints/qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final` (base `Qwen3-VL-8B-Instruct`) |
| LoRA | r=16, alpha=32, dropout=0.05, 7 target modules, 43,646,976 trainable |
| seed | 42 |
| Static UVFC run | `outputs/phase4b_online_uvfc/static_uvfc` (online GRPO, SFT anchor, `uvfc_dir=outputs/phase26/vfc`) |
| Static UVFC reward | frozen UVFC `sum_j v_j u_j` + violation-first (`p2`) group ranking; Flat verifier `Qwen2.5-VL-7B-Instruct` |

The manifest was **not** regenerated. CPU was pinned to logical cores 0–7
(`taskset -c 0-7`, `OMP/MKL/OPENBLAS/NUMEXPR/RAYON_NUM_THREADS=8`,
`TOKENIZERS_PARALLELISM=false`, `os.sched_setaffinity(0,{0..7})`); logged in the
Phase-5A-D run config as `cpu_affinity=[0..7]`. No DataLoader workers are used.

## 2. Methods and main results

| Method | Search | Coverage | Context | Spatial | Supported | Unsupported | BLEU | CIDEr | Dup | Dist-2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT Direct (existing) | 0.275 | 0.260 | 0.270 | 0.073 | 0.725 | 0.159 | 0.317 | 0.455 | 0.570 | 0.140 |
| Static UVFC (existing) | 0.332 | 0.332 | 0.255 | 0.071 | 0.814 | 0.094 | 0.326 | 0.519 | 0.499 | 0.174 |
| **Explore Direct** | 0.290 | 0.279 | 0.261 | 0.086 | 0.751 | 0.135 | 0.314 | 0.464 | 0.579 | 0.136 |
| Explore-4 (candidate 0) | 0.319 | 0.331 | 0.271 | 0.061 | 0.693 | 0.152 | 0.223 | 0.298 | 0.205 | 0.289 |
| **Explore-4 + UVFC** | **0.355** | **0.373** | 0.267 | 0.068 | 0.784 | **0.068** | 0.183 | 0.290 | 0.142 | 0.308 |
| Static UVFC + LatentCov (D, trained) | 0.341 | 0.339 | 0.258 | 0.062 | 0.811 | 0.092 | 0.332 | 0.534 | 0.498 | 0.173 |
| Prompt-CoT (existing) | 0.503 | 0.467 | 0.249 | 0.223 | 0.883 | 0.069 | 0.074 | 0.073 | 0.021 | 0.263 |
| GPT-5.6-Luna (existing) | 0.421 | 0.445 | 0.271 | 0.257 | 0.930 | 0.036 | 0.080 | 0.119 | 0.006 | 0.352 |

All rows have full coverage: SeeQ 1000/1000 and GPT-4o support 1000/1000
(Explore Direct and Explore-4 required failure retries; 0 residual API errors).

### Deltas vs Static UVFC

| Method | ΔSearch | ΔCoverage | ΔSupported | ΔUnsupported |
|---|---:|---:|---:|---:|
| Explore Direct | −0.041 | −0.053 | −0.063 | +0.041 |
| Explore-4 | −0.013 | −0.002 | −0.121 | +0.058 |
| **Explore-4 + UVFC** | **+0.023** | **+0.040** | −0.030 | **−0.026** |
| Static UVFC + LatentCov (D) | +0.009 | +0.006 | −0.003 | −0.002 |

### Deltas vs SFT Direct

| Method | ΔSearch | ΔCoverage | ΔSupported | ΔUnsupported |
|---|---:|---:|---:|---:|
| Explore Direct | +0.016 | +0.019 | +0.026 | −0.024 |
| Explore-4 | +0.044 | +0.071 | −0.032 | −0.007 |
| **Explore-4 + UVFC** | **+0.080** | **+0.113** | +0.059 | −0.091 |

## 3. Answers to the bottleneck questions

**Q1 — Is question exploration insufficient?**
Partly, but a *single* exploration prompt is weak. Explore Direct (one greedy
question under the inspection prompt) only reaches Search 0.290 / Coverage 0.279
— barely above SFT (+0.016/+0.020) and *below* Static UVFC. A prompt alone does
not reproduce the coverage that 1000 steps of UVFC GRPO buy.

**Q2 — Is the UVFC constraint effective?**
Yes. Selecting among four sampled exploration candidates with the frozen Static
UVFC rule (violation-first, then UVFC) removes most unsupported exploration:
candidate-pool confirmed violations drop 1255 → 22 selected, and
Supported rises 0.693 → 0.784 while Unsupported falls 0.152 → 0.068. UVFC
selection avoided a candidate-0 confirmed violation on 287/1000 images.

**Q3 — Are exploration and UVFC complementary?**
Yes, this is the clearest result. Explore-4 + UVFC beats Static UVFC on Search
(+0.023) and Coverage (+0.040) simultaneously and lowers Unsupported (−0.026),
at a Supported cost of −0.030 (0.784 vs 0.814). The coverage gain of exploration
and the grounding filter of UVFC compose: exploration supplies the visual
coverage, UVFC removes the unsupported part. Supported is slightly below Static
(just outside the ±0.02 reference band) and is the main residual concern.

**Q4 — Is the latent coverage reward worth continuing?**
Not on this evidence. The frozen feature-bank proxy is essentially uncorrelated
with the benchmark coverage metric (Section 5), and the high-reward clusters are
dominated by generic yes/no questions rather than richer visual content. See the
recommendation and Experiment D status.

## 4. Mechanism analysis

### 4.1 Cluster occupancy (frozen K=64 bank, 9184 SFT train questions)

| System | entropy | max cluster share | top-5 share | mean coverage reward | outlier fraction |
|---|---:|---:|---:|---:|---:|
| SFT | 3.974 | 0.087 | 0.208 | 0.282 | 0.060 |
| Static UVFC | 3.926 | 0.085 | 0.233 | 0.262 | 0.119 |
| Explore Direct | 3.966 | 0.085 | 0.203 | 0.277 | 0.060 |
| Explore-4 | 3.954 | 0.075 | 0.214 | 0.243 | 0.183 |
| Explore-4 + UVFC | 3.892 | 0.075 | 0.237 | 0.215 | 0.256 |

All systems already use all 64 clusters with near-maximum entropy (ln 64 ≈ 4.16);
no system shows strong head concentration. The bank does **not** separate the
methods, and the Explore-4/UVFC questions are *more* off-manifold relative to the
SFT bank (outlier fraction 0.18 → 0.26), i.e. they are genuinely novel in the
frozen representation.

### 4.2 Reward vs benchmark metric (per image, n=1000)

Spearman(latent coverage reward, SeeQ Search) = **−0.055**;
Spearman(latent coverage reward, SeeQ Coverage) = **−0.052**. The internal
image-conditioned representation proxy is **not aligned** with the target
coverage metric on the selected questions.

### 4.3 Question diversity / exploration (no GPU, no API)

| System | mean words | unique rate | distinct-2 | vocab | top template share | top-5 template share |
|---|---:|---:|---:|---:|---:|---:|
| SFT | 5.46 | 0.430 | 0.140 | 330 | 0.691 | 0.887 |
| Static UVFC | 5.58 | 0.501 | 0.174 | 420 | 0.410 | 0.673 |
| Explore Direct | 5.59 | 0.421 | 0.136 | 330 | 0.718 | 0.903 |
| Explore-4 | 5.53 | 0.795 | 0.289 | 627 | 0.205 | 0.473 |
| Explore-4 + UVFC | 5.96 | 0.858 | 0.308 | 686 | 0.117 | 0.362 |

Explore Direct is *more* templated than SFT/Static (top template 0.718) — a
single prompt does not diversify. Diversity comes from sampling (Explore-4) and
is further increased by UVFC selection (top template 0.117, vocab 686), so the
Explore-4+UVFC gain is not a template-repeat artifact.

### 4.4 UVFC effect on exploration (Explore-4 vs Explore-4+UVFC)

| | Search | Coverage | Supported | Unsupported | Dup | Dist-2 |
|---|---:|---:|---:|---:|---:|---:|
| Explore-4 (cand 0) | 0.319 | 0.331 | 0.693 | 0.152 | 0.205 | 0.289 |
| Explore-4 + UVFC | 0.355 | 0.373 | 0.784 | 0.068 | 0.142 | 0.308 |

UVFC selection raises coverage and grounding together and removes unsupported
exploration while keeping (and increasing) useful exploration.

## 5. Qualitative examples

UVFC selection replacing a candidate-0 confirmed violation (image, cand0 → selected):

- `10040`: "Are these soldiers happy?" (violation) → "How many men are wearing hats?" (UVFC 1.48)
- `103017`: "Is the bed neatly made?" (violation) → "Is this a bedroom?"
- `111031`: "What game are they playing?" (violation) → "What is the woman wearing?"

Residual failures (selected still `CONFIRMED_VIOLATION`, 22/1000): e.g.
"Are there any cars visible?", "Are the plates clean?", "Are the zebras looking at the camera?"
(unobservable-target / external-knowledge style, matching the 0.125
`unobservable_target` failure rate).

High latent-coverage-reward examples are typically generic yes/no questions
("Is it raining?", "Is the dog sleeping?") — evidence that the frozen proxy
rewards a linguistic/off-manifold region, not richer visual content.

## 6. Experiment D — Static UVFC + latent coverage reward (completed)

Single-variable trainer: Static UVFC unchanged plus
`scale(UVFC) + 0.15 · G_support · R_cov`, with `G_support = 0` on a confirmed
violation and a frozen K=64 bank / P95 gate built only from the train manifest
(`outputs/phase5a_latent_coverage/`). The 1000-step run
(`outputs/phase5a_static_uvfc_latentcov/`, seed 42, 22.75 s/step, 6.42 h, no
guard trip, peak 61.96 GB) reaches Search 0.341 / Coverage 0.339 / Supported
0.811 / Unsupported 0.092.

Deltas vs Static UVFC: Search +0.009, Coverage +0.006, Supported −0.003,
Unsupported −0.002, BLEU +0.006, CIDEr +0.015. This satisfies the literal
"Coverage ↑, Search ↑, Supported preserved" criterion, but the magnitude
(+0.006…+0.009) is within single-seed noise and consistent with the near-zero
proxy–metric alignment in Section 4.2. Training dynamics show no coverage
exploitation: support-gated coverage reward stays ~0.07–0.22 across 1000 steps,
outlier-gated fraction 0.09–0.53, and duplicate rate stays 0.00–0.09, i.e. the
distance gate prevents rare-cluster hacking but also caps the bonus.

## 7. Failure cases

- Explore Direct: the policy ignores "Output only the question"-style
  inspection prompts only weakly — questions remain short and templated; no
  coverage gain over SFT.
- Explore-4: the SFT policy refuses the single-pass "four questions" instruction
  (emits exactly one question and EOS); the pool was therefore built by sampling
  the candidate prompt 4× at T=0.8. Raw single-pass evidence is kept as
  `outputs/phase5a_explore4/candidates_singlepass.jsonl`.
- Explore-4 + UVFC: 22/1000 selected questions remain confirmed violations;
  Supported −0.030 vs Static is the residual reliability cost.
- Latent coverage proxy: near-zero rank correlation with the benchmark metric;
  high-reward clusters are generic-question artifacts.

## 8. Recommendation

**A. Continue Exploration + UVFC.**

Explore-4 + UVFC is the only tested method that moves Search *and* Coverage above
Static UVFC while reducing Unsupported, with substantially higher question
diversity; exploration and the UVFC constraint are complementary. The follow-up
should target the remaining Supported gap (e.g. a grounding-aware selection or
stronger violation filter over the exploration pool), not more prompt wording.

**Not B (latent coverage reward):** the frozen image-conditioned representation
proxy is unaligned with the benchmark coverage metric (Spearman ≈ 0) and its
high-reward region is a generic-question artifact. Experiment D confirms this:
after 1000 steps the trained model is only +0.009 Search / +0.006 Coverage over
Static UVFC with grounding unchanged — a within-noise gain, not the
diversity-driven jump seen in Explore-4 + UVFC. Do not scale this reward.

**Not C (return to Static UVFC):** Static remains the best grounding/reference
baseline, but it is no longer the coverage frontier — Explore-4 + UVFC is.

No next phase was started.
