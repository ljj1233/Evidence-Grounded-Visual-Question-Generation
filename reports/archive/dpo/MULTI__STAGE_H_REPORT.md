# STAGE H — Multi-positive feasibility audit (no training, no API)

Starting set M = 2733 (TREATMENT-M images). For each image: Pair-1 fixed as in
TREATMENT-M; search for an additional broad type t2 ≠ type(Pair-1 chosen) that has
both a Flat-Supported and a Flat-Unsupported candidate. t2 chosen deterministically
from seed 42 + image_id over the sorted eligible types.

## Feasibility

| quantity | value |
|---|---:|
| M | 2733 |
| **R** (images with ≥1 eligible second type) | **164** (6.0% of M) |
| additional eligible types per image | 0: 2569, 1: 164 |
| supported candidates of chosen t2 | 164 (exactly 1 per R image) |
| unsupported candidates of chosen t2 | 164 (exactly 1 per R image) |
| Pair-1 type histogram (R) | yes/no 132, count 19, what 9, color 4 |
| Pair-2 type histogram | what 128, color 17, yes/no 14, count 5 |
| joint transition (top) | yes/no→what 116, yes/no→color 11, count→yes/no 10, count→what 9 |

M images mostly have 4 candidates (2457/2733), yet only 164 contain a second type with
both labels; when they do, there is exactly one supported and one unsupported candidate
of that type, so q2+ (max Base length-normalized log-prob) and q2- (max TF-IDF to q2+)
are forced — the selection rules are satisfied trivially.

## Pair-2 statistics (R=164)

| quantity | value |
|---|---:|
| mean abs length gap (chosen/rejected) | 1.531 |
| mean TF-IDF cosine | 0.163 |
| mean q2+ Base length-normalized log-prob | −1.079 |
| NB text-only probe | 0.585 |
| TF-IDF+LR text-only probe | 0.561 |

Pair-2 pairs are near chance on both text-only probes (0.585 / 0.561), so the
multi-positive pair carries almost no lexical shortcut — a positive property.

## Usability

R = 164 images → **~21 optimizer steps** at batch = 8 images. This is not a usable
sample for the controlled SINGLE-R vs MULTI-R comparison; per the pre-registered
condition ("only execute STAGE I if … Stage H produces a usable R"), **STAGE I and
STAGE J were not run**. The bottleneck is the 4-candidate pool: it rarely supplies a
second same-type Supported/Unsupported pair per image.
