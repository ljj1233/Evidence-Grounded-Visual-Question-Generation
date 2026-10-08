# Same-type pair feasibility / shortcut audit (CPU only, no training, no API)

Uses the existing formal candidate pool + frozen Flat labels. No relabeling, no
training, no modification of the production pairs. Diagnostic only.

## Pair feasibility

| quantity | value |
|---|---:|
| original pairs (images) | 6633 |
| same-image, same-broad-type pairs constructible | **3734** |
| fraction of original images | **56.3%** |
| type balance | yes/no 51.5%, what 37.9%, count 6.5%, color 4.1%, other 0% |
| mean token length chosen / rejected | 5.79 / 5.48 (diff +0.32) |

Selection rule (deterministic): per image, within the first type in the fixed
order (yes/no, count, color, what, other) that has both a Supported and an
Unsupported candidate, chosen = max-UVFC supported, rejected = min-UVFC
unsupported; one pair per image. No dev-evaluation information is used.

## Text-only shortcut (image-grouped 5-fold NB bag-of-words pairwise ranking)

| pair set | mean accuracy | folds |
|---|---:|---|
| original (production) pairs | **0.706** | 0.689 / 0.708 / 0.714 / 0.707 / 0.711 |
| same-type candidate pairs | **0.629** | 0.581 / 0.633 / 0.645 / 0.640 / 0.646 |

Type-matching removes part of the shortcut (Δ ≈ −0.077), but a text-only
classifier still separates chosen from rejected well above chance (0.629) even
when the broad question type is identical. So the pair set encodes a **lexical /
phrasing** preference in addition to the type shift.

## Strongest within-type lexical features (weighted NB log-ratio)

chosen-leaning: `on` (+428), `there` (+337), `in` (+186) …
rejected-leaning: `is` (−309), `doing` (−153), `what` (−141), `this` (−117), `man` (−112) …

Interpretation: chosen ("Are there … on/in …") vs rejected ("Is this …? What is
the man doing?") phrasing.

## Conclusion (feasibility)

- A same-type training set is feasible for **56%** of images, but is heavily
  yes/no + what weighted (89.4%) and thin on count/color.
- Same-type matching only partially removes the text-only shortcut (0.706 → 0.629).
- The same-type set was **not** trained.
