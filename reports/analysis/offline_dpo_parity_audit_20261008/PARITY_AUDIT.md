# PART A — Flat evaluator / generation parity audit (no API)

## Verdict

**No evaluation bug. The premise mixes two different labelers.**

- Formal dev1000 "Full-DPO Supported = **0.988**" is **GPT-4o** support
  (`988 supported / 6 uncertain / 6 unsupported`, n=1000).
- DEV256 "final Flat-supported = **0.885**" is the **local Flat verifier**
  (`StructuredFlatVerifier` + frozen `visual_support_prompts.json`).
- On the *same* DEV256 final predictions: **GPT-4o = 0.984**, **Flat = 0.884**.

The two numbers are not comparable, so "0.988 vs 0.885" is not a contradiction.

The user's arithmetic (0.988 ⇒ ~12 non-supported on 1000; 0.885/256 ⇒ ~29) would
only conflict if both were Flat. They are not. There is **no old Flat label set for
dev1000** (the formal run scored dev with SeeQ + GPT-4o only; Flat was used on the
training candidate pool, not on dev generations).

## 2×2 parity table (DEV256)

| generation | scorer | final supported | base supported |
|---|---|---:|---:|
| old (formal dev1000 questions) | current Flat | 0.888 | 0.777 |
| new (trajectory regenerated) | current Flat | 0.884 | 0.749 |
| old (formal dev1000 questions) | GPT-4o | 0.988 | 0.938 |
| new (trajectory regenerated) | GPT-4o | 0.984 | 0.844 |

Old-vs-new Flat label agreement: final **0.996**, base **0.968**.
There is no separate "old scorer" binary: both runs instantiate the same
`StructuredFlatVerifier` with the same frozen prompt; the rescore confirms it.

## Generation parity

| tag | exact-question mismatches / 256 | GPT-4o label agreement |
|---|---:|---:|
| base | 15 (5.9%) | 0.961 |
| final (DPO) | 12 (4.7%) | 0.992 |

Greedy generation is **not bit-reproducible across batch composition**: the formal
dev1000 run batched 1000 images and the trajectory run batched the 256 subset, so
padding/numerics differ slightly and ~5% of questions flip. This is the only
"discrepancy" found, and it does not affect the support conclusion (0.888 vs 0.884).

## Flat vs GPT-4o agreement (same DEV256 questions, existing labels)

| checkpoint | n | agreement | Cohen κ | disagreement | count | color | what | yes/no |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 247 | 0.688 | 0.120 | 0.312 | 0.192 | 0.250 | 0.400 | 0.418 |
| step1000 | 252 | 0.787 | −0.011 | 0.213 | 0.230 | 0.200 | 0.280 | 0.300 |
| step2000 | 246 | 0.755 | 0.058 | 0.245 | 0.270 | 0.200 | 0.208 | 0.000 |
| step3000 | 243 | 0.774 | 0.067 | 0.226 | 0.244 | 0.170 | 0.500 | 0.000 |
| final | 251 | 0.876 | 0.037 | 0.124 | 0.175 | 0.214 | — | 0.095 |

Final confusion (Flat rows × GPT columns): Flat-supported → GPT {supported 219,
uncertain 1, unsupported 2}; Flat-unsupported → GPT {supported 28, unsupported 1}.
Flat is systematically **stricter** (28 GPT-supported cases called Flat-unsupported);
κ is near 0, so "Supported" is **not** a judge-independent quantity here.

## Implication

Parts B–E can proceed: there is no unresolved evaluation bug. But any claim must name
the labeler (GPT-4o vs Flat), and the two support numbers must never be compared
across labelers.
