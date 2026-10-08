# QUESTION SPECIFICITY AUDIT v1 (Phase 2.2B)

> **No final specificity metric or reward is selected in Phase 2.2B.** This experiment
> evaluates whether hard-negative question-to-image retrieval and setwise applicability
> provide useful diagnostics for question-image specificity.

## Scope

- Primary systems: `mdpo`, `pdpo`, `random`, `vanilla` (1000 val_debug predictions each).
- Excluded: `sft_baseline` — **incomplete-training diagnostic; excluded from scientific comparison**.
- This phase studies **only** question-image specificity, deliberately separate from
  visual-answer reliance. The Phase 2.2A absolute `visual_dependence` rubric is not reused
  (it is saturated at ~95% = 5).
- Hard negatives per question: K = 9; candidate set = 1 source + 9 hard negatives.

## 1. Hard-negative construction

- Backbone: `openai/clip-vit-base-patch32` (same local checkpoint/protocol as Metric Bake-off v1).
- Image embeddings: L2-normalized, GPU-batched, cached on disk (`4000` question records; unique images encoded once).
- For each source image, all other val_debug images are ranked by image-image cosine similarity; the deterministic top-K (tie-break: ascending `image_id`) are kept (seed 42).
- Random negatives are not used in the primary experiment; the task is intentionally hard.
- Manifest: `hard_negative_manifest.jsonl`.

## 2. Question-to-Image Retrieval

CLIPScore is the **unchanged** Metric Bake-off v1 definition (`2.5 * max(cos, 0)`, question text as-is).
Each question is ranked against its own 10-image candidate set. Chance Recall@1 = 10%.

### Table 1 — Hard-negative Question-to-Image Retrieval

| System | R@1 [95% CI] | R@5 | MRR [95% CI] | Mean source rank | Median source rank | Source CLIPScore | Mean hard-neg CLIPScore | Max hard-neg CLIPScore | Gap (source − mean neg) | Gap (source − max neg) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mdpo | 0.358 [0.328, 0.387] | 0.852 | 0.558 [0.536, 0.578] | 2.975 | 2.0 | 0.642 | 0.577 | 0.656 | 0.065 | -0.014 |
| pdpo | 0.354 [0.322, 0.384] | 0.853 | 0.556 [0.534, 0.578] | 2.947 | 2.0 | 0.650 | 0.583 | 0.662 | 0.066 | -0.013 |
| random | 0.352 [0.323, 0.383] | 0.858 | 0.556 [0.534, 0.578] | 2.935 | 2.0 | 0.650 | 0.583 | 0.662 | 0.066 | -0.013 |
| vanilla | 0.353 [0.322, 0.382] | 0.841 | 0.553 [0.529, 0.574] | 3.008 | 2.0 | 0.646 | 0.581 | 0.661 | 0.065 | -0.015 |

Hard-negative score diagnostics (per question, averaged per system): number of hard negatives scoring above the source, and number within a small CLIPScore margin of the source.

| System | mean # neg > source | mean # neg within 0.005 | within 0.01 | within 0.02 | within 0.05 |
|---|---:|---:|---:|---:|---:|
| mdpo | 1.975 | 2.213 | 2.480 | 2.950 | 4.411 |
| pdpo | 1.947 | 2.192 | 2.445 | 2.943 | 4.377 |
| random | 1.935 | 2.181 | 2.435 | 2.931 | 4.371 |
| vanilla | 2.008 | 2.247 | 2.487 | 2.993 | 4.409 |

**Portability / rank distribution (retrieval).** Fraction of questions with source rank 1 / ≤2 / ≤5, and the full rank distribution 1–10.

| System | frac rank=1 | frac rank≤2 | frac rank≤5 | rank 1..10 distribution |
|---|---:|---:|---:|---|
| mdpo | 0.358 | 0.551 | 0.852 | 0.358/0.193/0.132/0.088/0.081/0.052/0.035/0.036/0.017/0.008 |
| pdpo | 0.354 | 0.550 | 0.853 | 0.354/0.196/0.129/0.103/0.071/0.057/0.042/0.027/0.015/0.006 |
| random | 0.352 | 0.552 | 0.858 | 0.352/0.200/0.131/0.103/0.072/0.053/0.041/0.025/0.017/0.006 |
| vanilla | 0.353 | 0.545 | 0.841 | 0.353/0.192/0.126/0.100/0.070/0.057/0.045/0.034/0.016/0.007 |

_The two gap columns are diagnostics only and are NOT declared a final metric._

## 3. GPT-4o Setwise Specificity

Setwise applicability prompt/rubric (prompt version `specificity_setwise_v1`): the question plus 10
**randomized-order** candidate images are shown; the source identity is hidden. The judge returns JSON
only: `applicable_images` (indices whose image provides sufficient visible evidence for the question
to be naturally asked), `best_matching_image` (which image most specifically motivates this exact
question, or null), `unique_best_match` (true only if exactly one image best motivates it), and a short
`reason`. The judge is explicitly told not to judge mere object-word overlap.

### Table 2 — GPT-4o Setwise Specificity

| System | n | Source Applicable [95% CI] | Source Unique [95% CI] | Source Best-Match [95% CI] | Unique Source Best-Match [95% CI] | Portability mean [95% CI] ↓ | Portability median | Portability p25 | Portability p75 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mdpo | 200 | 0.835 [0.780, 0.885] | 0.415 [0.355, 0.480] | 0.570 [0.505, 0.640] | 0.570 [0.505, 0.640] | 0.202 [0.168, 0.237] | 0.111 | 0.000 | 0.333 |
| pdpo | 200 | 0.870 [0.825, 0.915] | 0.355 [0.290, 0.420] | 0.565 [0.495, 0.635] | 0.565 [0.495, 0.635] | 0.213 [0.179, 0.248] | 0.111 | 0.000 | 0.333 |
| random | 200 | 0.840 [0.790, 0.890] | 0.395 [0.330, 0.460] | 0.560 [0.490, 0.625] | 0.560 [0.490, 0.625] | 0.216 [0.182, 0.252] | 0.111 | 0.000 | 0.333 |
| vanilla | 200 | 0.835 [0.785, 0.885] | 0.365 [0.300, 0.430] | 0.580 [0.510, 0.650] | 0.580 [0.510, 0.650] | 0.221 [0.187, 0.259] | 0.111 | 0.000 | 0.333 |

Portability = (# applicable hard-negative images) / 9. Lower = more source-image-specific.

| System | portability histogram (0/9 … 9/9) |
|---|---|
| mdpo | 0.415/0.205/0.100/0.080/0.070/0.025/0.030/0.030/0.030/0.015 |
| pdpo | 0.355/0.205/0.150/0.085/0.040/0.065/0.055/0.025/0.015/0.005 |
| random | 0.395/0.180/0.110/0.090/0.055/0.060/0.040/0.050/0.015/0.005 |
| vanilla | 0.365/0.195/0.125/0.100/0.045/0.045/0.065/0.040/0.015/0.005 |

_All metrics in Table 2 are diagnostic statistics in this phase and are NOT a final proposed metric._

_Observation: in this run `unique_best_match` was true for every set where `best_matching_image` was the source, so the Source Best-Match and Unique Source Best-Match columns coincide; the judge did not, however, always pick the source as best._

## 4. Specificity + Existing Visual Support

Source-image `visual_support` is reused from Phase 2.2A (`gpt4o_judge.jsonl`); **no new API calls**.
Quadrants cross strong specificity (unique source best-match) with full support (support = 2). The four
conceptual cases — (1) generic but supported, (2) source-specific and supported, (3) source-specific but
unsupported, (4) generic and unsupported — are kept separate and never collapsed into one score.

### Table 3 — Specificity + Existing Visual Support

| System | Existing Support mean ↑ | specific & supported | specific & unsupported | generic & supported | generic & unsupported |
|---|---:|---:|---:|---:|---:|
| mdpo | 1.920 | 0.550 | 0.020 | 0.395 | 0.035 |
| pdpo | 1.940 | 0.545 | 0.020 | 0.410 | 0.025 |
| random | 1.940 | 0.545 | 0.015 | 0.410 | 0.030 |
| vanilla | 1.935 | 0.555 | 0.025 | 0.395 | 0.025 |

## 5. Paired System Comparisons

Paired bootstrap by `image_id` (resamples = 1000).
Retrieval and GPT setwise differences are reported separately and never merged.

### Table 4 — Paired System Comparisons

| Comparison | CLIP R@1 Δ [95% CI] | CLIP MRR Δ [95% CI] | CLIP source-rank Δ [95% CI] | GPT source-applicable Δ | GPT unique-source-best Δ | GPT portability Δ |
|---|---|---|---|---|---|---|
| pdpo vs vanilla | 0.001 [-0.010, 0.014] | 0.003 [-0.005, 0.013] | -0.061 [-0.125, -0.005] | 0.035 [-0.010, 0.085] | -0.015 [-0.070, 0.040] | -0.008 [-0.027, 0.012] |
| mdpo vs vanilla | 0.005 [-0.021, 0.030] | 0.005 [-0.012, 0.022] | -0.033 [-0.132, 0.068] | 0.000 [-0.050, 0.050] | -0.010 [-0.080, 0.060] | -0.019 [-0.046, 0.010] |
| random vs vanilla | -0.001 [-0.013, 0.012] | 0.003 [-0.005, 0.013] | -0.073 [-0.132, -0.023] | 0.005 [-0.040, 0.050] | -0.020 [-0.075, 0.035] | -0.004 [-0.025, 0.016] |
| pdpo vs random | 0.002 [-0.007, 0.010] | -0.000 [-0.006, 0.006] | 0.012 [-0.019, 0.047] | 0.030 [-0.015, 0.080] | 0.005 [-0.050, 0.060] | -0.003 [-0.021, 0.014] |
| mdpo vs pdpo | 0.004 [-0.022, 0.029] | 0.001 [-0.016, 0.019] | 0.028 [-0.074, 0.133] | -0.035 [-0.090, 0.015] | 0.005 [-0.065, 0.080] | -0.011 [-0.040, 0.016] |

_A difference is not claimed significant absent a CI excluding 0; tiny differences are not over-read._

_Only two paired retrieval differences have a 95% CI excluding 0 (pdpo vs vanilla and random vs vanilla on CLIP source rank, both small negative mean differences of ~0.06–0.07 ranks). Every GPT setwise paired difference includes 0. No system is declared better than another, and no system is selected as the final method here._

## 6. Relation to Existing Automatic Metrics

Existing BLEU-4 / CIDEr / Q-BLEU4 / Distinct-2 / duplicate-rate values are reused, not recomputed.

| System | BLEU-4 | CIDEr | Q-BLEU4 | Distinct-2 | Dup rate | CLIP R@1 | GPT source unique | Portability ↓ | Existing support ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mdpo | 0.1860 | 0.2911 | 0.1802 | 0.2068 | 0.2500 | 0.358 | 0.415 | 0.202 | 1.920 |
| pdpo | 0.2209 | 0.3636 | 0.2152 | 0.2166 | 0.2780 | 0.354 | 0.355 | 0.213 | 1.940 |
| random | 0.2213 | 0.3682 | 0.2155 | 0.2144 | 0.2840 | 0.352 | 0.395 | 0.216 | 1.940 |
| vanilla | 0.2187 | 0.3562 | 0.2130 | 0.2097 | 0.2910 | 0.353 | 0.365 | 0.221 | 1.935 |

_Systems with near-identical CIDEr/Q-BLEU4 can still be compared on retrieval/setwise specificity;
no correlation claim is forced where the systems do not differ._

## 7. Example Analysis

Deterministic selection criteria (no manual cherry-picking); up to 20 per group (actual counts below); saved in `qualitative_examples.jsonl`.

- **A. Low specificity / high support** — support = 2 and portability ≥ 0.5; sorted by highest portability, then worst rank.
- **B. High specificity / high support** — support = 2, unique source best-match, portability = 0; sorted by best rank.
- **C. High specificity / low support** — support < 2, unique source best-match, portability ≤ 1/9; sorted by lowest support.
- **D. Retrieval failure** — CLIP source rank > 1 but GPT best-matching image = source; sorted by worst rank.

| Group | count |
|---|---:|
| A. low specificity / high support | 20 |
| B. high specificity / high support | 20 |
| C. high specificity / low support | 16 |
| D. retrieval failure | 20 |

Representative examples:

**A_low_specificity_high_support**
- `pdpo` / image 211282: “What type of pizza is shown in the image?” (rank=8, portability=0.556, support=2, reason: Image 8 shows a variety of pizzas, making it most relevant for asking about pizza types.)
- `vanilla` / image 494634: “What is the cat doing?” (rank=8, portability=0.556, support=2, reason: Image 9 shows a cat sleeping, which directly answers the question about what the cat is doing.)
- `mdpo` / image 327306: “What is the skateboarder doing?” (rank=7, portability=0.556, support=2, reason: Image 9 shows a skateboarder in mid-air, clearly performing a trick.)

**B_high_specificity_high_support**
- `mdpo` / image 102356: “What is the color of the motorcycle's sidecar?” (rank=1, portability=0.000, support=2, reason: Image 5 shows a motorcycle with a visible sidecar, allowing the question about its color.)
- `mdpo` / image 116750: “What is written on the side of the blue container?” (rank=1, portability=0.000, support=2, reason: Image 4 shows a blue container with text on its side.)
- `mdpo` / image 117466: “What is the child in the purple shirt holding?” (rank=1, portability=0.000, support=2, reason: Image 1 shows a child in a purple shirt holding something.)

**C_high_specificity_low_support**
- `mdpo` / image 566038: “What type of bird is seen in the image?” (rank=1, portability=0.000, support=0, reason: Image 10 shows a bird, which directly relates to the question.)
- `pdpo` / image 566038: “What type of bird is this?” (rank=1, portability=0.000, support=0, reason: Image 2 shows a bird, which directly relates to the question.)
- `random` / image 566038: “What type of bird is this?” (rank=1, portability=0.000, support=0, reason: Image 8 shows a bird, making the question applicable.)

**D_retrieval_failure**
- `mdpo` / image 124803: “What is the baby playing with?” (rank=2, portability=0.000, support=2, reason: Image 2 shows a baby playing with a toy activity center.)
- `mdpo` / image 134213: “What is the name of the museum?” (rank=2, portability=0.000, support=2, reason: Image 5 shows a museum sign with the name 'Seeley Stable Museum'.)
- `mdpo` / image 243955: “What is the weather like on the beach?” (rank=2, portability=0.444, support=2, reason: Image 7 shows people on a beach with umbrellas, suggesting a focus on weather conditions.)

## 8. Runtime / Throughput / Hardware

### Table 5 — Runtime / throughput

| Quantity | Value |
|---|---|
| GPU model | NVIDIA A800-SXM4-80GB |
| CLIP device | cuda |
| Requested / selected batch size | 256 / 256 |
| OOM batch halvings | 0 |
| Peak GPU memory (MB) | 1357.5 |
| Unique images encoded (once) | 1000 |
| Total image-embedding time (s) | 9.688 |
| Total text-embedding time (s) | 0.315 |
| Retrieval scoring time (s) | 0.286 |
| Questions/sec | 6654.463 |
| GPT concurrency | 32 |
| GPT sets judged | 800 |
| GPT failures | 0 |
| GPT cache hits | 0 |

## 9. Cache policy

- Image embeddings: disk cache keyed by backbone hash under `outputs/question_specificity_v1/cache/`; each unique image encoded once, reused across systems.
- Text embeddings: in-memory dedup by exact question string.
- GPT judgments: deterministic cache key = sha256(question | ordered candidate image ids | judge model | prompt version); incremental JSONL append; successful cached requests are never repeated; failed items logged separately.
- Candidate-image display order is randomized per (seed, system, image_id) and stored, so recovery of the hidden source index is exact.

## 10. Limitations

- Hard negatives are CLIP-image-similarity neighbours; they need not be semantically identical, so some negatives may be easy or accidentally fully applicable.
- The GPT setwise judge is an automated anchor, not human annotation, and may be imperfect on fine visual relations.
- The 200-questions-per-system subset shares source images across systems (paired design) but is a subset; estimates carry sampling uncertainty (bootstrap CIs reported).
- CLIPScore is known to be only weakly sensitive to compositional/relational content.
- `sft_baseline` is excluded (incomplete-training diagnostic); no claim is made about a properly trained SFT model.

## Outputs

`outputs/question_specificity_v1/`: `hard_negative_manifest.jsonl`, `clip_retrieval_per_question.jsonl`,
`clip_retrieval_summary.json`, `gpt4o_setwise_judgments.jsonl`, `gpt4o_setwise_summary.json`,
`paired_system_analysis.json`, `qualitative_examples.jsonl`, `runtime_report.json`.

## Closing statement

No final specificity metric or reward is selected in Phase 2.2B.
This experiment evaluates whether hard-negative question-to-image retrieval and setwise applicability
provide useful diagnostics for question-image specificity.
