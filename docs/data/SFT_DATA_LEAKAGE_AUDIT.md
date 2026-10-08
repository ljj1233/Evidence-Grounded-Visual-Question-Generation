# SFT data leakage audit

This is the required Phase 2.4 delivery name for the complete audit in [SFT_LEAKAGE_AUDIT_V2.md](SFT_LEAKAGE_AUDIT_V2.md). The decision is `CLEAN_WITH_TEXT_OVERLAP_CAVEAT`: canonical COCO image IDs are disjoint across all audited splits, records contain only `image` and `question`, the loader/loss use question tokens only, and final-holdout artifacts contain no predictions, rewards, or judge labels. VQAv2 question/template overlap is reported separately and is not image leakage.
