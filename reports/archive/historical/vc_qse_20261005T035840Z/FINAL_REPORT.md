# FINAL REPORT — PART A archive + PART B VC-QSE

Run: `reports/vc_qse_20261005T035840Z/` · seed 42 · GPU 0 (A800-80GB)
git `9dea27f09aa5c21f4fa53f730923ac1df64169eb` (working tree dirty; **no tag**)

## 1. Soft-evidence v2 archive path

`archives/full_ours_soft_evidence_v2_20261005/`

## 2. Archive file list

- `METHOD_STATUS.md`, `MANIFEST.json`, `COMMANDS.md`, `RESULTS_SUMMARY.md`,
  `FAILURE_ANALYSIS.md`, `GIT_STATE.md`
- `GIT_STATUS.porcelain.txt` (370), `GIT_DIFF.patch` (4235),
  `GIT_DIFF_CACHED.patch` (12), `GIT_UNTRACKED.txt` (329)
- `code/` — 19 source copies: `src/vvqg/evidence/{model,model_v2,assignment,__init__}.py`,
  `scripts/vvqg/ec_evidence_{train,generate}.py`,
  `scripts/vvqg/ec_slot_assignment_{train,generate,smoke}.py`,
  `scripts/vvqg/ec_build_slot_packets.py`,
  `scripts/vvqg/ec_e2_{correspondence,decoder_use,analyze,encoder_audit,c0_control}.py`,
  `scripts/vvqg/ec_phase1_v2_diagnostics.py`, `scripts/vvqg/ec_v2_{compare,assignment_cases}.py`,
  `tests/test_slot_assignment_v2.py`
- `reports/` — 3 archived report trees (67 files copied); 4 oversized checkpoint
  files recorded by size+sha256 but **not** copied.
- MISSING at requested path (recorded in MANIFEST, actually present at
  `scripts/vvqg/` and archived from there): the five `scripts/vqg/ec_*` paths.

## 3. v2 final failure conclusion

`ARCHIVED_FAILED_AS_MAIN_METHOD`; `not_a_published_baseline: true`;
`current_main_method: false`. Slot collapse fixed numerically (cosine 1.000 →
−0.25) but image correspondence FAIL (R@1 0.0052 vs chance 0.0039), decoder-use
FAIL (residual/input 4.5e-4), 195/256 images generate the same question under all
slot conditions, Coverage 1.937→1.951 with 95% CI [−0.024, +0.053]. RL never
authorized or run. Not visual grounding; slot-cosine decrease is not success
evidence.

## 4. VC-QSE research hypothesis

Automatically discovering multiple askable visual concepts from the image and
generating one question per concept (before any selection) increases question
space Coverage over Direct-4 without lowering Supported. No latent slots, no
Hungarian assignment, no v2 evidence branch, no RL.

## 5. Concept proposer implementation

`scripts/vvqg/vc_concept_propose.py` + `src/vvqg/vc_qse/concepts.py`:
frozen `Qwen3-VL-8B-Instruct`, **image-only**, greedy, 512 tokens, strict JSON
schema, raw responses cached, parse failures recorded (never dropped), multiple
concepts retained per image, fallback flagged. Controls: `blank` (grey image),
`shuffled` (seed-42 pixel permutation).

## 6. Concept schema

```json
{"objects":[{"text":"...","confidence":0.0}],
 "attributes":[{"text":"...","confidence":0.0}],
 "relations":[{"text":"...","confidence":0.0}],
 "scene":[{"text":"...","confidence":0.0}]}
```
Caps objects/attributes/relations=4, scene=2; text ≤8 words; dedupe; confidence
kept; no questions/answers/explanations.

## 7. Concept smoke result (64 images)

**PASS.** parse 1.000 (≥0.95); 13.70 concepts/image (≥2); not a fixed template
(`tree` 14.1%); blank inventory empty (change 1.0); shuffled change 1.0; 0
question/answer leakage; fallbacks flagged. Details: `CONCEPT_SMOKE.md`.

## 8. C0 and VC-QSE commands

See `RUN_COMMANDS.md`. Core:

```bash
# concepts
.venv/bin/python scripts/vvqg/vc_concept_propose.py --manifest <devN_manifest.jsonl> --outdir <out> --condition original --batch-size 8
# VC candidates
.venv/bin/python scripts/vvqg/vc_generate.py --concepts <out>/concepts_original.jsonl --outdir <vc> --batch-size 16
# C0 Direct-4
.venv/bin/python scripts/vvqg/phase5b_direct4_generate.py --outdir <c0> --manifest <devN_manifest.jsonl> --batch-size 16 --num-samples 4
# local support + adherence + selection
.venv/bin/python scripts/vvqg/vc_support_verify.py --candidates <candidates.jsonl> --out <support.jsonl> --batch-size 24
.venv/bin/python scripts/vvqg/vc_concept_adherence.py --candidates <vc_candidates.jsonl> --out <adherence.jsonl> --batch-size 16
.venv/bin/python scripts/vvqg/vc_pilot.py --c0-support <...> --vc-support <...> --vc-adherence <...> --outdir <selected>
```

## 9. 64-image pilot result

C0 supported 0.953 / unsupported 0.031; VC supported 0.984 / unsupported 0.000;
fallback 0.031 vs 0.000; duplicate 0.063 vs 0.031; mean words 5.56 vs 5.55.
`PILOT64.md`.

## 10. 256-image pilot result

See `PILOT256.md`; gate JSON `pilot256/gate.json`.

## 11. Coverage / Supported / Unsupported table

| metric | C0_DIRECT4 | VC-QSE_PILOT | Δ |
|---|---:|---:|---:|
| Coverage (0–5) | 1.6797 | 1.5430 | −0.1367 |
| Supported | 0.7578 | 0.8164 | +0.0586 |
| Unsupported | 0.0820 | 0.0664 | −0.0156 |
| Uncertain | 0.1602 | 0.1172 | −0.0430 |

## 12. Paired CI

mean Δ coverage −0.1367, bootstrap 95% CI **[−0.293, +0.012]** (10000 resamples,
seed 42); 47 up / 63 down / 146 tied.

## 13. Fallback rate

0.082 (21/256). Non-fallback Δ −0.1362; fallback Δ −0.1429.

## 14. object/attribute/relation/scene stratification

object n=89 Δ −0.2022; attribute n=41 Δ −0.3415; relation n=43 Δ −0.1628;
scene n=62 Δ **+0.1129**. Only scene is positive.

## 15. Failure cases (36 shown; 156 recorded)

```
128324 d=-4 c0[5/supp] "How many elephants are in the image?" || vc[1/supp] "Is the grass green?" <scene:savanna>
150675 d=-4 c0[5/unce] "Is this a bedroom?" || vc[1/supp] "What time is it?" <attribute:wall clock>
9077 d=-3 c0[4/supp] "Is this indoors?" || vc[1/supp] "Is this indoors?" <scene:indoor market>
22257 d=-3 c0[4/supp] "How many giraffes are there?" || vc[1/unce] "Is there a tree in the background?" <scene:savanna>
31569 d=-3 c0[4/unce] "Are the animals in their natural habitat?" || vc[1/supp] "Is the elephant standing on grass?" <object:elephant>
117201 d=-3 c0[4/supp] "How many people are standing?" || vc[1/supp] "What color is the snow?" <attribute:white snow>
143572 d=-3 c0[4/supp] "Are there a lot of people in the stands?" || vc[1/supp] "What sport is being played?" <scene:tennis match>
153865 d=-3 c0[4/unce] "Is this a kite festival?" || vc[1/supp] "What are the colors on the kite?" <object:kite shaped like shark>
2315 d=-2 c0[5/supp] "How many trees can you see?" || vc[3/unsu] "Are the elephants wild?" <relation:elephants near river>
14921 d=-2 c0[3/supp] "How many pictures are on the wall?" || vc[1/supp] "What is the color of the towel?" <scene:bathroom>
20487 d=-2 c0[3/unce] "What is written in red?" || vc[1/supp] "How many people are in this picture?" <object:two people>
21310 d=-2 c0[3/supp] "How many children are wearing tennis shoes?" || vc[1/supp] "Is the middle girl a blonde?" <object:three children>
21327 d=-2 c0[3/supp] "What is the name of the beer?" || vc[1/supp] "What brand is the beer?" <object:beer glass>
24247 d=-2 c0[3/supp] "Is the woman taking a picture of the dog?" || vc[1/supp] "What is the woman doing?" <object:person>
25853 d=-2 c0[3/supp] "What is leaning against the bench?" || vc[1/supp] "What color is the bike?" <object:bicycle>
27226 d=-2 c0[3/supp] "How many trains are there?" || vc[1/supp] "What color is the train?" <object:red train>
28508 d=-2 c0[3/supp] "How many animals are there?" || vc[1/unsu] "Is this an adult giraffe?" <relation:giraffe standing>
33345 d=-2 c0[3/supp] "How many lamps are there?" || vc[1/supp] "What color is the television?" <relation:television on table>
34180 d=-2 c0[3/unsu] "What is the man in the back doing?" || vc[1/supp] "Is the person in the banana costume smiling?" <object:person with banana costume>
48196 d=-2 c0[3/supp] "What show is advertised on the bus?" || vc[1/supp] "What is the name of the movie advertised on the bus?" <relation:bus on road>
57283 d=-2 c0[3/supp] "What is in the girls hair?" || vc[1/supp] "Is the child sitting indoors?" <scene:indoor>
63516 d=-2 c0[3/supp] "How many horses are in the picture?" || vc[1/supp] "What type of animal is this?" <object:horse>
67422 d=-2 c0[3/supp] "What number is on the shirt?" || vc[1/supp] "What number is on the shirt?" <fallback>
70280 d=-2 c0[3/supp] "How many birds are on the right?" || vc[1/unsu] "Is this a beach?" <scene:beach>
76521 d=-2 c0[3/supp] "What number is the man on the left?" || vc[1/supp] "How many people are on the lift?" <scene:ski resort>
79392 d=-2 c0[3/unce] "What type of trees are in the foreground?" || vc[1/supp] "Is it a clear day?" <attribute:blue sky>
79975 d=-2 c0[3/unce] "Are the cones for traffic?" || vc[1/supp] "What is the man riding?" <relation:skateboarder riding skateboard>
87483 d=-2 c0[3/supp] "Is there a sheep in the background?" || vc[1/supp] "Is there green grass in the picture?" <attribute:green grass>
93040 d=-2 c0[3/supp] "Are the cows resting?" || vc[1/supp] "What color is the grass?" <attribute:green grass>
95595 d=-2 c0[3/unce] "What is this person holding?" || vc[1/supp] "Is this a kitchen?" <attribute:kitchen>
106392 d=-2 c0[3/supp] "What color is the countertop?" || vc[1/supp] "What color are the cabinets?" <attribute:wooden cabinets>
111031 d=-2 c0[3/supp] "How many people are shown?" || vc[1/unce] "What is the woman holding?" <relation:woman holding wii remote>
122047 d=-2 c0[3/unce] "What are the white objects?" || vc[1/unsu] "What color is the bowl?" <attribute:white bowl>
124157 d=-2 c0[3/supp] "Are the bananas ripe?" || vc[1/supp] "What color is the fruit?" <attribute:yellow>
127496 d=-2 c0[3/supp] "What room is this?" || vc[1/supp] "What color is the toilet?" <object:toilet>
133867 d=-2 c0[3/supp] "What is on the elephants?" || vc[1/supp] "What is on the sign?" <attribute:red sign>
```
Full list: `pilot256/selected/failure_cases_merged.jsonl`.
Concept-adherence=no examples (dropped by VC_SELECTOR): 636 `toilet`|"What is the
reflection?"; 636 `toilet against stone wall`|"Is there water in the toilet?";
1903 `pickle on hot dog`|"Is the hot dog in a paper wrapper?".

## 16. 256 gate

**FAIL** — criterion 1 (Coverage > C0) and criterion 5 fail; 2,3,4,6,7,8 pass.

## 17. Train VC-QSE

**NOT ALLOWED** (256 gate FAIL).

## 18. Generate train concept data

**NOT ALLOWED** (256 gate FAIL).

## 19. RL

**NOT ALLOWED** (and not run at any point).

## 20. Checkpoints / seed / GPU / git

- seed 42; GPU 0 (A800-80GB); git `9dea27f09aa5c21f4fa53f730923ac1df64169eb`.
- SFT generator: `outputs/checkpoints/qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final`,
  adapter sha256 `5fb66c36a2255fb3a9011db198fd55a8be8b9cbb26eef611727a816bc7934e0e`.
- concept proposer / adherence: `/hpc2hdd/home/fye374/models/Qwen3-VL-8B-Instruct`,
  config sha256 `5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661`.
- Flat verifier: `/hpc2hdd/home/fye374/models/Qwen2.5-VL-7B-Instruct`,
  config sha256 `77d9ec7321cc572e3579e2c84799c9cadaded63c49ce93b101733349fc330c43`.
- Evaluator: SeeQ `gpt-5.6-luna` (variant, single pass), support `gpt-4o`
  (single pass, 478 requests + 34 cache hits).

## 21. New / modified files

New: `src/vvqg/vc_qse/{__init__,concepts,selector}.py`;
`scripts/vvqg/vc_{concept_propose,concept_smoke,generate,support_verify,concept_adherence,pilot,analyze256}.py`;
`tests/test_vc_qse.py`; `archives/full_ours_soft_evidence_v2_20261005/**`;
`reports/vc_qse_20261005T035840Z/**`. **No existing core file modified; no old
code/checkpoint/report deleted.**

## 22. Copyable summary (≤12 lines)

```
PART A: archives/full_ours_soft_evidence_v2_20261005/ — v2 ARCHIVED_FAILED_AS_MAIN_METHOD
  (slot collapse fixed; correspondence FAIL; decoder-use FAIL; Coverage CI [-0.024,+0.053]; no RL).
PART B: VC-QSE (image-only Qwen3-VL concepts -> concept-conditioned questions -> VC_SELECTOR).
  Concept smoke 64: PASS (parse 1.00, 13.7 concepts/img, controls change 1.0, no leakage).
  64 pilot: local Flat support 0.984 vs C0 0.953; no external metric.
  256 pilot (single-pass SeeQ+GPT-4o): Coverage 1.543 vs C0 1.680, delta -0.137, CI [-0.293,+0.012].
    Supported 0.816 vs 0.758; Unsupported 0.066 vs 0.082; fallback 0.082; dup 0.102 vs 0.090.
    Strata: object -0.202 (89), attribute -0.341 (41), relation -0.163 (43), scene +0.113 (62).
  256 GATE = FAIL (Coverage below C0). => no train pool, no VC-QSE training, no RL.
  seed 42, GPU 0, git 9dea27f0. Failures: pilot256/selected/failure_cases_merged.jsonl (156).
```
