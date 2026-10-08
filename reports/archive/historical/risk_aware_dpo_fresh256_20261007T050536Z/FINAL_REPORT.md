# FINAL REPORT — Risk-aware DPO-v2 (fresh-256) — DATA_INSUFFICIENT

## Headline

- **STATUS: DATA_INSUFFICIENT**
- The frozen S/U/X pair rules do not yield the pre-registered minimum of 100 `S>U` and 100 `U>X` pairs.
  Per the explicit instruction "不得自动降低阈值", training was **not started** and no thresholds were changed.
- The fresh 256-image pilot manifest was still produced (for transparency); it is unused because no
  DPO-v2 model was trained.
- `FINAL_HOLDOUT_ACCESSED: NO`.

## FACT

Data construction (`PAIR_COUNTS.json`, `DATA_STATS.json`), on the existing instruction-matched aligned pool
(`outputs/ec_qse_aligned_direct4/dev/candidates_scored.jsonl`):

- Class definitions: S = supported AND support_floor ≥ 0.75 AND no uncertain/unsupported requirement;
  U = overall uncertain; X = unsupported OR CONFIRMED_VIOLATION.
- Class image counts — train: S 578, U 67, X 578; all 1000 images: S 956, U 147, X 750.
- Pair counts (train): **S>X 313, S>U 38, U>X 43**. All-images: S>X 391, S>U 78, U>X 62.
- Filter reasons: 404 pairs removed by the UVFC guard (chosen UVFC < image candidate median − 0.10).
- Fresh pilot: 256 images selected from 294 remaining after excluding v1 train (578) and v1 pilot (128);
  overlap = 0 with train, 0 with old pilot, 0 with final_holdout.

Applying the UVFC guard as a candidate filter (instead of after hash-pick) lowers the counts further
(train S>U 25, U>X 31). No reasonable interpretation reaches 100/100.

## INTERPRETATION

- The aligned candidate pool contains very few `uncertain` and unsupported low-support candidates, so the
  `U>X` (and `S>U`) pair types are intrinsically scarce; the UVFC tie-break guard removes many more.
- The frozen data-sufficiency guard therefore blocks Risk-aware DPO-v2 before training. Lowering the
  support_floor threshold, dropping the UVFC guard, or reducing the 100-pair minimum would all be
  automatic threshold changes, which were explicitly forbidden.

## UNKNOWN

- Whether a larger candidate pool (more images / higher-K) would supply ≥100 `U>X` pairs.
- Whether Risk-aware DPO-v2 would reduce Unsupported or preserve the DPO-v1 Supported gain; untested
  because training did not run.

## Next action

Stop. No training, no expansion, no threshold change. To proceed the user must explicitly relax the
frozen count guard (or authorize a larger candidate pool); neither was done here.
