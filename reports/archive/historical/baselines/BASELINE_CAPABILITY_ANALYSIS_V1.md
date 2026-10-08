# Baseline Capability Analysis v1

SFT improves reference overlap on this development set: corpus BLEU-4 rises from 0.089809 to 0.317225, CIDEr from 0.126362 to 0.454971, and the documented Q-BLEU variant from 0.087013 to 0.310714. These are profile measurements, not a claim that one metric alone defines question quality.

SFT worsens greedy output diversity: Distinct-2 falls from 0.276117 to 0.139524 and normalized excess duplicate rate rises from 0.121 to 0.570. In the 4-sample pools, the exact rollout unique rate (mean unique questions per image divided by 4) is 0.76625 for Zero-shot and 0.95575 for SFT; the normalized values are 0.76600 and 0.95550. Greedy unique-question rates are 0.879 and 0.430; those greedy rates are lexical corpus statistics, while within-image rollout uniqueness is only informative for sampled pools.

Visual-centric capability is evaluated with the released SeeQ rubric using the documented gpt-5.6-luna judge substitution. Paper-exact GPT-5.4/five-question SeeQ is not claimed; the local four-dimension scores use the frozen one-question-per-image project protocol.

The independent support judge produced 1,000 valid labels for each system; API and parser failures are zero after retry. Zero-shot supported/unsupported/uncertain rates are 0.634/0.230/0.136; SFT rates are 0.725/0.159/0.116. Failure counts and the six failure-type rates are in the summary.

Dominant failure types are reported separately in the summary; this phase records the observed profile and does not infer causes.

VIC is a mechanism diagnostic expected to inform the visual-information axis, especially image-vs-no-image contribution; it is not a master score or reward weight. Violation-aware feedback is theoretically expected to affect Supported/Unsupported/Uncertain rates and their six failure categories. These are protocol expectations, not observed method effects.
