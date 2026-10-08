# Required source dependency

Copied `reports/published_baseline_intervention_20261004T032525Z/scripts/build_transforms.py` because `src/vvqg/evidence_constrained/counterfactual.py` dynamically imports it at module import time. This is a  source dependency required for test collection, not an experiment cache or generated dataset. Source code remains byte-identical. First collection failure is preserved in PYTEST_FIRST_COLLECTION.txt.
