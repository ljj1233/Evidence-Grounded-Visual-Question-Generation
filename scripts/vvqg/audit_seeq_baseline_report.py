#!/usr/bin/env python3
"""SeeQ-style baseline audit: synthesis artifacts (pipeline maps, fidelity, decision).

Pairs with audit_seeq_baseline.py (data funnels).  Reads the computed data JSONs
and writes the interpretive artifacts into outputs/seeq_baseline_audit/.
Audit-only; reads/writes no model or generation artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/seeq_baseline_audit"
OFFICIAL_COMMIT = "18e795293808c5cf9f69f1e980577ab40b3faa95"


def load(name: str) -> dict:
    return json.loads((OUT / name).read_text())


def write(name: str, payload: dict) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    proposal = load("proposal_diversity.json")
    rewrite = load("rewrite_audit.json")
    funnel = load("uniqueness_funnel.json")
    diversity = load("diversity_funnel.json")
    filt = load("filter_audit.json")
    retention = load("retention_audit.json")
    budget = load("training_budget_audit.json")
    fmt = load("training_format_audit.json")

    # ── official pipeline ────────────────────────────────────────────────────
    official = {
        "commit": OFFICIAL_COMMIT,
        "repo": "https://github.com/tianyi-lab/SeeQ",
        "inputs": "external unlabeled image JSON; image-only, no human questions (README; load_seed_data postfix='sat')",
        "stages": [
            {"stage": "Seed proposal", "evidence": "OFFICIAL_CODE gen/scripts/generation_pipeline.py:1045-1072",
             "purpose": "propose seed questions from the image",
             "inputs": "image", "model_role": "generation model (MODEL_TYPE/MODEL_PATH)",
             "prompt": "genqa_prompt_v1 + booster and genqa_prompt_v2 + booster (evol_utils.py:55-59)",
             "sampling": "do_sample=True", "decision_rule": "is_multi_question reject; simplify; answer; self_judge",
             "output": "seed (Simplified, Answer, filter_met)", "retained_metadata": "dict_seed v1/v2"},
            {"stage": "Seed diversity mechanism", "evidence": "OFFICIAL_CODE generation_pipeline.py:1046-1072; evol_utils.py:55",
             "purpose": "10 seed draws per image", "model_role": "generation model",
             "prompt": "5 boosters x 2 prompt versions", "decision_rule": "n/a", "output": "up to 10 seeds",
             "retained_metadata": "booster/promptver"},
            {"stage": "Simplify", "evidence": "OFFICIAL_CODE selfjudge_utils.py:441-452",
             "purpose": "normalize before answer/score", "model_role": "filter/rewrite model",
             "sampling": "do_sample=True", "decision_rule": "reject 'What effect'/'What happens'; dedup dict_ans",
             "output": "simplified question"},
            {"stage": "Answer generation", "evidence": "OFFICIAL_CODE selfjudge_utils.py:454-455",
             "purpose": "answer the simplified question", "model_role": "generation model",
             "prompt": "gene_answer()", "output": "answer", "decision_rule": "n/a"},
            {"stage": "Standard evolution (rewrite)", "evidence": "OFFICIAL_CODE generation_pipeline.py:171-354; evol_utils.py:143-200",
             "purpose": "make the seed harder", "model_role": "generation model",
             "prompt": "generate_instructions_v2: add_object_v1_specific, add_task_v1_random, add_task_v1_specific, add_comparison_v1, add_multihop_v1",
             "decision_rule": "multi_question; simplify; dedup; lexical_overlap>=0.82 reject; self_judge; difficulty improvement",
             "output": "rewritten QA + failure reason"},
            {"stage": "Difficulty-improvement gate", "evidence": "OFFICIAL_CODE selfjudge_utils.py:937-969; generation_pipeline.py:318,760",
             "purpose": "accept only harder rewrites",
             "decision_rule": "reasoning_delta>=0.5 OR (reasoning_delta>=0 and perception_delta>=1.0) OR both>0",
             "output": "accept / difficulty_not_improved_enough"},
            {"stage": "Compound / multi-question rejection", "evidence": "OFFICIAL_CODE generation_pipeline.py:68-77; selfjudge_utils.py:786-802",
             "purpose": "reject concatenated questions", "decision_rule": "is_multi_question and _is_compound_question",
             "output": "reject multi_question"},
            {"stage": "Lexical-overlap filter", "evidence": "OFFICIAL_CODE retrieval_utils.py:143-148; generation_pipeline.py:272",
             "purpose": "reject paraphrases", "decision_rule": "lexical_overlap_ratio(rewrite,seed)>=0.82 reject",
             "output": "lexical_overlap_too_high"},
            {"stage": "Deduplication", "evidence": "OFFICIAL_CODE generation_pipeline.py:140,235,288,681",
             "purpose": "no repeated simplified question", "decision_rule": "dict_ans / dict_qs_met membership",
             "output": "duplicate_simplified_question"},
            {"stage": "Model-following gate", "evidence": "OFFICIAL_CODE selfjudge_utils.py:993-1007; generation_pipeline.py:781",
             "purpose": "rewrite must obey the added constraint", "decision_rule": "has_required_model_following(last constraint)",
             "output": "model_following_failed"},
            {"stage": "Structured rewrite validation", "evidence": "OFFICIAL_CODE retrieval_utils.py:215-277",
             "purpose": "require evidence/steps/intermediate inference",
             "decision_rule": "validate_rag_question(require_structured=True): >=2 visible_evidence, >=2 reasoning_steps, nonempty intermediate_inference, syntactic, overlap<0.82, no generic flags, specificity>=2.0",
             "output": "insufficient_visible_evidence / insufficient_reasoning_steps / missing_intermediate_inference / generic_shallow_pattern / low_specificity"},
            {"stage": "RAG image-fact cache", "evidence": "OFFICIAL_CODE retrieval_utils.py:504-548",
             "purpose": "ground 'harder' rewrites in extracted image facts",
             "decision_rule": "md5(image_path)-keyed JSON cache; feeds add_multihop_rag_facts_v1 (broader_question_prompts)"},
            {"stage": "RAG sample bank", "evidence": "OFFICIAL_CODE retrieval_utils.py:339-481,649-704",
             "purpose": "harder-than-examples via quality-weighted sampling; bootstraps across rounds",
             "decision_rule": "sample_from_bank(k=5, quality-weighted); try_add_to_bank novelty (concept-tag Jaccard<0.6) + difficulty gate"},
            {"stage": "Stage-2 evolution", "evidence": "OFFICIAL_CODE generation_pipeline.py:1145-1188; retrieval_utils.py:707-734",
             "purpose": "second evolution on top-8 stage-1 outputs", "decision_rule": "rank_seed_pairs_for_stage2(top_k=8)",
             "output": "stage-2 standard + RAG rewrites"},
            {"stage": "Retention", "evidence": "UNKNOWN",
             "purpose": "turn accepted rewrites into the training set",
             "decision_rule": "UNKNOWN (no retention code released)"},
            {"stage": "Training formatting", "evidence": "OFFICIAL_CODE gen/create_training/utils.py:6-53",
             "purpose": "conversation formats", "decision_rule": "random instruction per task type",
             "output": "IQ (image->Q), IQA (image->Q+A), VQG (image+A->Q), VQA (image+Q->A)"},
            {"stage": "Training objective", "evidence": "PAPER",
             "purpose": "train questioner + answerer on retained synthetic QA",
             "decision_rule": "UNKNOWN trainer/optimizer/epochs (not released)"},
            {"stage": "Round progression", "evidence": "README; PAPER",
             "purpose": "M_t bootstraps harder over runs via the growing sample bank",
             "decision_rule": "NUM_TURN within a run; bank persists across runs"},
            {"stage": "Evaluation", "evidence": "OFFICIAL_CODE eval/question/valid_difficulty_eval.py; question_gene_embedding_diversity.py",
             "purpose": "4 visual-centric dimensions (0-5) + embedding diversity",
             "decision_rule": "GPT judge (JUDGE_MODEL default gpt-5-mini); Qwen3-Embedding-4B mean pairwise cosine distance"},
        ],
        "unknowns": ["retention/selection code", "training loop M0->M1->M2", "optimizer/epochs/LoRA-vs-full-FT",
                      "official judge model at paper time", "exact NUM_TURN/round count"],
    }
    write("official_pipeline.json", official)

    # ── local pipeline ───────────────────────────────────────────────────────
    local = {
        "official_commit_claimed": OFFICIAL_COMMIT,
        "adaptation_name": "SeeQ-style Self-Evolution Adaptation",
        "inputs": "9,184 unlabeled VQAv2 train2014 images (image-only, no human question/answer)",
        "stages": [
            {"stage": "Schedule", "evidence": "src/vvqg/baselines/seeq_style/pipeline.py:204-213",
             "detail": "2 turns x 2 candidates = 4 proposals/image; task=TASKS[(i*4+turn*2+cand+seed+round)%17]"},
            {"stage": "Proposal", "evidence": "pipeline.py:196; vllm_backend.py:128-135",
             "prompt": "single PROPOSAL_PROMPT + ' Task family: {task}'",
             "sampling": "greedy (temperature=0.0)", "output": "JSON 'Question'"},
            {"stage": "Answer", "evidence": "pipeline.py:197", "model_role": "proposer model",
             "prompt": "identical ANSWER_PROMPT", "detail": "answers the raw proposal, not a simplified seed; greedy"},
            {"stage": "Rewrite", "evidence": "pipeline.py:198", "model_role": "reference model (M0)",
             "prompt": "single REWRITE_PROMPT; free-form 'Rewritten question'; greedy",
             "detail": "no boosters, no RAG, no structured evidence/steps/inference"},
            {"stage": "Filter", "evidence": "pipeline.py:88-116,199",
             "prompt": "FILTER_PROMPT; reference model (M0); greedy",
             "decision_rule": "hard gate only: Factuality.count==0 and Ambiguity.count==0 and Validity.answerable"},
            {"stage": "Difficulty improvement", "evidence": "absent in _accepted", "decision_rule": "MISSING"},
            {"stage": "Compound/multi rejection", "evidence": "absent", "decision_rule": "MISSING"},
            {"stage": "Lexical-overlap filter", "evidence": "absent", "decision_rule": "MISSING"},
            {"stage": "Deduplication", "evidence": "absent in _accepted", "decision_rule": "MISSING"},
            {"stage": "Model-following gate", "evidence": "absent", "decision_rule": "MISSING"},
            {"stage": "Structured rewrite validation", "evidence": "absent", "decision_rule": "MISSING"},
            {"stage": "RAG fact cache / sample bank", "evidence": "pipeline.py:269 fact_cache_used/sample_bank_used always empty; rag_used=false",
             "decision_rule": "MISSING (scaffold-only)"},
            {"stage": "Retention", "evidence": "pipeline.py:53-74; scripts/vvqg/prepare_seeq_round.py",
             "decision_rule": "deterministic image round-robin, exactly 10,000"},
            {"stage": "Training formatting", "evidence": "src/vvqg/baselines/seeq_style/training.py:11-20",
             "output": "qg (image->question+' '+answer) and qa (image+question->answer)"},
            {"stage": "Training", "evidence": "round1_m1/training_args.json; round2_m2/training_args.json",
             "detail": "LoRA r16/a32/drop0.05, 1 epoch, bs2, no accum, lr2e-5, cosine, warmup1000, 10,000 steps"},
            {"stage": "Round progression", "evidence": "round2_generation_m1/generation_manifest.json",
             "detail": "R1 M0 proposer/M0 reference; R2 M1 proposer/M0 reference"},
            {"stage": "Evaluation", "evidence": "evaluation/summary.json",
             "detail": "dev1000 greedy + sampled-4; BLEU-4/CIDEr/Q-BLEU-4v/Distinct-2/dup; GPT-5.6-luna 4 dims; GPT-4o support; Qwen3-Embedding-4B diversity"},
        ],
    }
    write("local_pipeline.json", local)

    # ── fidelity matrix ──────────────────────────────────────────────────────
    def row(stage, official, local, status, severity, source):
        return {"stage": stage, "official_seeq": official, "local_implementation": local,
                "match_status": status, "severity": severity, "evidence_source": source}

    matrix = [
        row("Proposal", "10 seed draws/image (5 boosters x 2 prompts), do_sample=True",
            "4 draws/image from ONE prompt + task label, greedy",
            "ADAPTED", "HIGH", "OFFICIAL_CODE"),
        row("Intent conditioning", "booster/prompt-version diversity; task list only in evolution",
            "17 task-definition names appended as 'Task family'; no booster/prompt version",
            "ADAPTED", "MEDIUM", "OFFICIAL_CODE"),
        row("Answer generation", "gene_answer() on simplified seed (generation model)",
            "identical ANSWER_PROMPT on raw proposal (proposer model)",
            "FUNCTIONALLY_EQUIVALENT", "LOW", "OFFICIAL_CODE"),
        row("Rewrite", "multi-strategy instructions (object/task/area/comparison/multihop) + RAG; structured JSON",
            "single free-form REWRITE_PROMPT; 'Rewritten question' only",
            "ADAPTED", "HIGH", "OFFICIAL_CODE"),
        row("Simplify", "simplify_prompt() called before answer/score (mandatory in seed + evolution)",
            "simplify_prompt defined but never called",
            "MISSING", "MEDIUM", "OFFICIAL_CODE"),
        row("Difficulty comparison", "has_difficulty_improved(seed,rewrite) required for acceptance",
            "absent; judge soft scores ignored",
            "MISSING", "CRITICAL", "OFFICIAL_CODE"),
        row("Validity filtering", "self_judge hard block via is_valid_hard_constraints (self_judge_v2/ori path)",
            "hard gate Factuality/Ambiguity/Validity inline",
            "FUNCTIONALLY_EQUIVALENT", "LOW", "OFFICIAL_CODE"),
        row("Compound rejection", "is_multi_question + _is_compound_question",
            "absent",
            "MISSING", "MEDIUM", "OFFICIAL_CODE"),
        row("Lexical/model-following filter", "lexical_overlap>=0.82 reject; has_required_model_following",
            "absent",
            "MISSING", "HIGH", "OFFICIAL_CODE"),
        row("Structured rewrite validation", "validate_rag_question(require_structured=True)",
            "absent",
            "MISSING", "HIGH", "OFFICIAL_CODE"),
        row("Deduplication", "per-run dict_ans/dict_qs_met; bank novelty",
            "absent",
            "MISSING", "HIGH", "OFFICIAL_CODE"),
        row("RAG sample bank", "quality-weighted sample_from_bank + novelty try_add_to_bank",
            "empty scaffold; rag_used=false",
            "MISSING", "HIGH", "OFFICIAL_CODE"),
        row("Fact cache", "md5-keyed image-fact JSON feeding RAG instructions",
            "empty scaffold; not used",
            "MISSING", "MEDIUM", "OFFICIAL_CODE"),
        row("Retention", "UNKNOWN (unreleased)",
            "deterministic image round-robin to exactly 10,000; no dedup",
            "UNKNOWN", "MEDIUM", "UNKNOWN"),
        row("Training formatting", "IQ/IQA/VQG/VQA with random instruction per example",
            "qg (~unstructured IQA) + qa (~VQA); IQ and VQG absent",
            "ADAPTED", "HIGH", "OFFICIAL_CODE"),
        row("Training objective", "train questioner + answerer (trainer unreleased)",
            "LoRA SFT QG+QA, 1 epoch, bs2, lr2e-5",
            "ADAPTED", "MEDIUM", "PAPER"),
        row("Round progression", "M_t proposer; persistent growing bank across runs",
            "R1 M0->M1; R2 M1->M2; frozen M0 reference; no bank",
            "ADAPTED", "MEDIUM", "PAPER"),
        row("Evaluation - dataset", "model's own generated question sets",
            "development1000 val2014 sample",
            "ADAPTED", "LOW", "OFFICIAL_CODE"),
        row("Evaluation - judge", "GPT judge (default gpt-5-mini; 4 dims 0-5)",
            "gpt-5.6-luna variant, 4 dims 0-5, same rubric family",
            "FUNCTIONALLY_EQUIVALENT", "LOW", "OFFICIAL_CODE"),
        row("Evaluation - diversity", "Qwen3-Embedding-4B mean pairwise cosine distance",
            "same Qwen3-Embedding-4B mean pairwise cosine distance",
            "EXACT", "LOW", "OFFICIAL_CODE"),
    ]
    write("fidelity_matrix.json", {"rows": matrix,
          "counts": {"EXACT": sum(r["match_status"] == "EXACT" for r in matrix),
                     "FUNCTIONALLY_EQUIVALENT": sum(r["match_status"] == "FUNCTIONALLY_EQUIVALENT" for r in matrix),
                     "ADAPTED": sum(r["match_status"] == "ADAPTED" for r in matrix),
                     "MISSING": sum(r["match_status"] == "MISSING" for r in matrix),
                     "UNKNOWN": sum(r["match_status"] == "UNKNOWN" for r in matrix)}})

    # ── RAG / fact cache ─────────────────────────────────────────────────────
    write("rag_factcache_audit.json", {
        "official": {
            "fact_cache": "get_image_facts extracts {objects,attributes,spatial_relations}; md5(image_path) JSON cache; feeds broader_question_prompts",
            "sample_bank": "JSONL bank; curate (dedup + quality), quality-weighted sample_from_bank(k=5); try_add_to_bank adds novel accepted questions; bootstraps difficulty ceiling across rounds",
            "role": "generation context, not acceptance correctness; drives the 'harder' direction",
        },
        "local": {"fact_cache": "EMPTY", "sample_bank": "EMPTY", "rag_used": False},
        "classification": "MISSING",
        "consequence": "changes the generation context (no grounding facts, no harder exemplars); does not by itself alter the filter acceptance rule but removes the main documented driver of difficulty growth",
        "severity": "HIGH",
        "evidence": "OFFICIAL_CODE retrieval_utils.py; local runtime.json/rag_used=false",
    })

    # ── evaluation protocol ──────────────────────────────────────────────────
    write("evaluation_protocol_audit.json", {
        "official": {
            "dataset": "model-generated question sets from the chosen input images",
            "questions_per_image": "all generated (seed + evolution)",
            "judge": "GPT judge, JUDGE_MODEL default gpt-5-mini, 4 dims scored 0-5 then /5",
            "diversity": "Qwen3-Embedding-4B mean pairwise cosine distance per image",
        },
        "local": {
            "dataset": "development1000 (val2014 sample), disjoint from train/calibration/holdout",
            "greedy": "1 question/image, max_new_tokens=32",
            "sampled": "4 questions/image, temperature=0.8, top_p=0.9, max_new_tokens=32",
            "judge": "gpt-5.6-luna variant (VARIANT_DOCUMENTED, not official GPT-5.4/mini)",
            "embedding": "Qwen3-Embedding-4B",
            "normalization": "M1/M2 emit 'question? answer'; predictions truncated at first '?' uniformly; M0 unchanged",
        },
        "fidelity": {"rubric": "FUNCTIONALLY_EQUIVALENT", "dataset": "ADAPTED",
                     "sampling": "ADAPTED", "judge": "FUNCTIONALLY_EQUIVALENT (variant model)",
                     "diversity": "EXACT"},
        "caveat": "absolute official/local numbers are not comparable because judge version and dataset differ; only within-local M0/M1/M2 trends are interpretable",
    })

    # ── collapse attribution ─────────────────────────────────────────────────
    r1p = proposal["R1"]["question_part"]
    r2p = proposal["R2"]["question_part"]
    write("collapse_attribution.json", {
        "earliest_substantial_contraction": "PROPOSAL",
        "sources": {
            "A_proposal_collapse": {
                "classification": "STRONG_EVIDENCE",
                "numbers": {
                    "R1_proposal_unique_rate": r1p["unique_rate"], "R1_proposal_distinct_2": r1p["distinct_2"],
                    "R1_top_share": round(r1p["top_question"][1] / r1p["n"], 4),
                    "R2_proposal_unique_rate": r2p["unique_rate"], "R2_proposal_distinct_2": r2p["distinct_2"],
                    "R2_top_share": round(r2p["top_question"][1] / r2p["n"], 4),
                    "R1_within_image_one_unique_share": 5820 / 9184, "R2_within_image_one_unique_share": 7707 / 9184,
                    "M0_sampled4_unique_rate": funnel["models"]["M0"]["sampled4"]["unique_rate"],
                },
                "mechanism": "greedy decoding + a single proposal prompt + no boosters: 4 candidates/image are near-identical; already collapsed before any rewrite/filter",
            },
            "B_rewrite_homogenization": {
                "classification": "MODERATE_EVIDENCE",
                "numbers": {"R1_exact_copy_rate": rewrite["R1"]["proposal_to_rewrite_exact_copy_rate"],
                            "R2_exact_copy_rate": rewrite["R2"]["proposal_to_rewrite_exact_copy_rate"],
                            "R1_rewrite_unique": rewrite["R1"]["rewrite_unique"], "R1_proposal_unique": rewrite["R1"]["proposal_unique"],
                            "R2_rewrite_unique": rewrite["R2"]["rewrite_unique"], "R2_proposal_unique": rewrite["R2"]["proposal_unique"]},
                "mechanism": "rewrite is largely an identity map (66-82% exact copies) and adds no new diversity; it fails to increase difficulty",
            },
            "C_filter_homogenization": {
                "classification": "MODERATE_EVIDENCE",
                "numbers": {"R1_local_accepted": filt["rounds"]["R1"]["replayed_official_deterministic"]["local_accepted"],
                            "R1_surviving_deterministic": filt["rounds"]["R1"]["replayed_official_deterministic"]["local_accepted_surviving_official_deterministic_filters"],
                            "R2_local_accepted": filt["rounds"]["R2"]["replayed_official_deterministic"]["local_accepted"],
                            "R2_surviving_deterministic": filt["rounds"]["R2"]["replayed_official_deterministic"]["local_accepted_surviving_official_deterministic_filters"]},
                "mechanism": "the hard-only gate admits generic/duplicate questions that official lexical, dedup, difficulty and model-following filters would remove",
            },
            "D_retention_homogenization": {
                "classification": "MODERATE_EVIDENCE",
                "numbers": {"R1_retained_unique": funnel["R1"]["retained"]["unique_normalized"],
                            "R1_retained_n": funnel["R1"]["retained"]["n"],
                            "R2_retained_unique": funnel["R2"]["retained"]["unique_normalized"],
                            "R1_top_share": round(retention["R1"]["top_retained_questions"][0][1] / 10000, 4),
                            "R2_top_share": round(retention["R2"]["top_retained_questions"][0][1] / 10000, 4)},
                "mechanism": "round-robin selects from an already-collapsed set and duplicates it into training (10k records, ~1.6-1.9k unique)",
            },
            "E_training_overfitting": {
                "classification": "MODERATE_EVIDENCE",
                "numbers": {"effective_epochs_R1": budget["R1"]["effective_epochs"], "effective_epochs_R2": budget["R2"]["effective_epochs"],
                            "unique_question_exposures_R1": budget["R1"]["unique_question_exposures"],
                            "unique_question_exposures_R2": budget["R2"]["unique_question_exposures"],
                            "R1_final_loss": budget["R1"]["final_loss"], "R2_initial_loss": budget["R2"]["initial_loss"],
                            "R2_final_loss": budget["R2"]["final_loss"]},
                "mechanism": "1 epoch, but each unique question is seen 5-6x/epoch; M2 starts at loss 0.19 (M1 already fits its own collapsed output) and drops to 0.054, reinforcing the mode",
            },
            "F_evaluation_artifact": {
                "classification": "WEAK_EVIDENCE",
                "numbers": {"collapse_in_greedy": True, "collapse_in_sampled": True,
                            "normalization_uniform": True},
                "mechanism": "collapse is visible under both greedy and sampled decoding, so it is not a greedy-only or truncation artifact",
            },
        },
        "notes": ["causality is not inferred from temporal order; each source is supported by a measured contraction",
                  "the dominant contraction is the proposal stage and is an implementation (decoding/prompt) divergence, not expected SeeQ behavior"],
    })

    # ── divergences ──────────────────────────────────────────────────────────
    divergences = [
        {"id": "D1", "stage": "filter", "divergence": "difficulty-improvement acceptance missing",
         "severity": "CRITICAL", "official_evidence": "selfjudge_utils.has_difficulty_improved; generation_pipeline.py:318,760",
         "impact": "the advertised self-evolution-to-harder mechanism is not enforced; local accepts on static validity only"},
        {"id": "D2", "stage": "proposal", "divergence": "greedy decoding + single prompt vs do_sample + 10 seed draws",
         "severity": "HIGH", "official_evidence": "generation_pipeline.py:1046-1072; evol_utils.py:55-59",
         "impact": "proposal distribution is already collapsed (unique 5-8%, within-image 63-84% one unique)"},
        {"id": "D3", "stage": "filter", "divergence": "lexical-overlap + dedup + compound + model-following filters missing",
         "severity": "HIGH", "official_evidence": "retrieval_utils.lexical_overlap_ratio; selfjudge_utils._is_compound_question; generation_pipeline dedup",
         "impact": "offline replay of these deterministic rules keeps only 1287/23618 (R1, -94.6%) and 1026/25789 (R2, -96.0%) of the local accepted rows"},
        {"id": "D4", "stage": "rewrite", "divergence": "free-form rewrite without structured evidence/steps/intermediate inference",
         "severity": "HIGH", "official_evidence": "retrieval_utils.validate_rag_question(require_structured=True); harder_question_prompts/broader_question_prompts",
         "impact": "no grounding requirement and no difficulty guarantee; 66-82% of rewrites are exact copies of the proposal"},
        {"id": "D5", "stage": "RAG", "divergence": "fact cache and sample bank absent",
         "severity": "HIGH", "official_evidence": "retrieval_utils.get_image_facts/load_sample_bank/try_add_to_bank; generation_pipeline RAG stage",
         "impact": "removes the documented driver of difficulty growth and exemplar diversity"},
        {"id": "D6", "stage": "training format", "divergence": "no image->question-only (official IQ) format; qg target concatenates answer",
         "severity": "MEDIUM", "official_evidence": "create_training/utils.py IQ/IQA/VQG/VQA",
         "impact": "the model is never trained on the evaluation-time image->question format; it always emits 'question answer'"},
        {"id": "D7", "stage": "proposal", "divergence": "no simplify step before answer/score",
         "severity": "MEDIUM", "official_evidence": "selfjudge_utils.simplify_prompt used in seed and evolution",
         "impact": "answers/scores are computed on unnormalized questions"},
        {"id": "D8", "stage": "retention", "divergence": "no dedup; round-robin project rule",
         "severity": "MEDIUM", "official_evidence": "retention UNKNOWN",
         "impact": "10k training records contain only ~1.6-1.9k unique questions"},
        {"id": "D9", "stage": "provenance", "divergence": "R1 rewrite schema {'text'} vs R2 {'Rewritten question'}; both rounds greedy; local SeeQ code untracked in git",
         "severity": "LOW", "official_evidence": "n/a",
         "impact": "handled by the question-extraction fallback; a version drift between rounds"},
    ]
    write("divergences.json", {"high_or_critical": [d for d in divergences if d["severity"] in ("CRITICAL", "HIGH")],
                               "all": divergences,
                               "counts": {s: sum(1 for d in divergences if d["severity"] == s)
                                          for s in ("CRITICAL", "HIGH", "MEDIUM", "LOW")}})

    # ── decision ─────────────────────────────────────────────────────────────
    write("decision.json", {
        "decision": "SEEQ_STYLE_VALID_WITH_MAJOR_CAVEATS",
        "exact_reproduction": False,
        "core_self_evolution_loop_preserved": "PARTIAL",
        "high_severity_divergences": 5,
        "rerun_required": False,
        "paper_safe_name": "SeeQ-style Self-Evolution Adaptation (same backbone)",
        "diversity_collapse_first_substantial_at": "PROPOSAL",
        "reasoning": (
            "The loop skeleton (propose -> answer -> rewrite -> filter -> retain -> train, iterated, with "
            "R2 generated by the M1 proposer against a frozen M0 reference) is preserved and the model roles "
            "are correct. However the acceptance gate omits official difficulty-improvement/model-following "
            "filtering, the rewrite is unstructured and mostly identity, the RAG fact/sample-bank machinery is "
            "absent, and proposal generation is greedy with a single prompt. The baseline therefore represents "
            "self-evolution but is not a fair direct comparison to SeeQ on its headline 'harder questions' claim. "
            "The observed diversity collapse starts at the proposal stage and is an implementation divergence "
            "(greedy + single prompt), not expected SeeQ behavior. No confirmed implementation error invalidates "
            "the trained weights, so a rerun is not mandatory; the result can remain as an auxiliary baseline "
            "with explicit caveats."),
        "not_invalid_because": "no wrong model loaded, no stale adapter in the valid R2, retention/training executed as specified, evaluation normalization applied uniformly",
        "recommended_actions": [
            "keep the result as an auxiliary same-backbone baseline labelled SeeQ-style adaptation",
            "do not claim difficulty improvement or fair reimplementation of SeeQ's filter",
            "if a main-table comparison is required, regenerate with sampling + official filter criteria (minimal plan in docs/reproducibility/SEEQ_BASELINE_FINAL_AUDIT.md)",
        ],
    })

    print("synthesis artifacts written")


if __name__ == "__main__":
    main()
