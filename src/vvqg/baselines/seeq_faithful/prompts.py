"""Verbatim SeeQ prompts traced to the paper and the official repository.

Sources
-------
PAPER: "Self-Evolving Visual Questioner" (arXiv:2606.13929v1), Appendix B.
OFFICIAL_CODE: github.com/tianyi-lab/SeeQ @ 18e795293808c5cf9f69f1e980577ab40b3faa95
    utils/gen/evol_utils.py
    utils/gen/selfjudge_utils.py
    gen/scripts/generation_pipeline.py

The paper names five visual intents but ships no per-intent template (Appendix B
contains a single Question Proposal prompt, Fig. 2).  The per-intent condition
lines below are therefore constructed minimally and marked
LOCAL_CONTROLLED_ADAPTATION; every other string is copied verbatim.
"""
from __future__ import annotations

SEEQ_COMMIT = "18e795293808c5cf9f69f1e980577ab40b3faa95"

# ---------------------------------------------------------------------------
# PAPER Fig. 2 == OFFICIAL_CODE evol_utils.genqa_prompt_v1 / gen_format
# ---------------------------------------------------------------------------
QUESTION_CONSTRAINTS = (
    "The question must rely on visual information such that its answer would change or be impossible to determine without the image. "
    "The answer should be deterministic and answerable with a few sentences. "
    "You should generate only ONE focused question, not multiple sub-questions. "
    "IMPORTANT: Do NOT create compound or multi-part questions by joining sub-questions with 'and' or 'what'. "
    "A question like 'How does X affect Y, and what in the image shows Z?' is FORBIDDEN — it is two easy questions concatenated, not a harder one. "
    "Instead, make the SINGLE question harder by requiring deeper reasoning: "
    "multi-hop inference, counterfactual/causal reasoning, cross-region comparison, or quantitative/ordinal judgment. "
    "The answer must be DIFFERENT from the original answer. "
)

GEN_FORMAT = "Return result in a json format: {'Question': 'xxx', 'Thinkings': 'Your selection process'}\n"

GENQA_PROMPT_V1 = (
    "List n1 topics that you can answer questions about. Choose a topic uniformly from this list, and state it. "
    "Then write 60 subtopics about the chosen topic. Then choose a subtopic uniformly from this list, and state it. "
    "Then write a question that is not about the subtopic, but can only be answered with expertise in the subtopic. "
    "Then write the answer. Both the question and answer should be long. "
    "The name of the subtopic should not appear in the question, and none of the words in subtopic should be reused in the question. "
    + QUESTION_CONSTRAINTS + GEN_FORMAT
)

# The paper's Fig. 2 proposal prompt, used verbatim.
BASE_PROPOSAL_PROMPT = GENQA_PROMPT_V1

# ---------------------------------------------------------------------------
# Five visual intents named in PAPER 3.1.  Template wording is LOCAL (not shipped).
# ---------------------------------------------------------------------------
INTENTS = {
    "direct_recognition": (
        "direct recognition — ask the model to identify, name, or read one specific "
        "visible object, attribute, or action in the image."
    ),
    "comparison": (
        "comparison — ask the model to compare two or more visible objects, regions, "
        "or attributes in the image."
    ),
    "spatial_relations": (
        "spatial relations — ask about the position, layout, orientation, or spatial "
        "relation of visible elements in the image."
    ),
    "scene_understanding": (
        "scene understanding — ask about the scene, activity, context, or situation "
        "that the visible evidence suggests."
    ),
    "grounded_reasoning": (
        "grounded reasoning — ask a question whose answer requires combining multiple "
        "pieces of visual evidence or inference grounded in the image."
    ),
}
INTENT_NAMES = tuple(INTENTS)

# ---------------------------------------------------------------------------
# OFFICIAL_CODE evol_utils.py rewrite condition + format
# ---------------------------------------------------------------------------
QUESTION_CONDITION = (
    "You are a Question generator. Your objective is to rewrite a given Q&A into a genuinely harder version that require more visual evidence or reasoning steps. "
    "Make it harder by increasing the DEPTH of reasoning required — for example: "
    "(1) Evidence selection: Which visible evidence best supports that this area is used for work?"
    "(2) Object-role comparison: Which object is more central to the workspace function, the laptop or the apple?"
    "(3) Spatial relation: What does the chair’s placement relative to the desk suggest about the room’s use?"
    "(4) Scene-function inference: What visible objects indicate that this is a study/work area?"
    "(5) Multi-object grounding: How do the laptop, chair, and notebook together suggest the purpose of this space?"
    "Do NOT make it harder by simply appending more sub-questions or extra clauses. "
    "Do NOT generate a question that asks the same things as the original or is just a paraphrase. "
    "The question should not become broader, more subjective, or more generic."
    "The added constraint must change the type or depth of reasoning required, not just expand the question's length. "
    "Do NOT generate questions using the template 'How does X affect/impact/influence Y' — "
    "these produce world-knowledge questions that do not require looking at the image. "
)

REWRITE_FORMAT = (
    "Return the result in a JSON format: {'Rewritten question': 'xxx', 'Added information': 'xxxx', "
    "'Modification': '<what modification is done>'}\n"
)

# OFFICIAL_CODE per-strategy rewrite instructions (evol_utils.py), composed with
# the shared question_condition exactly as the repository does.
ADD_COMPARISON_V1 = QUESTION_CONDITION + """Rewrite the given Q&A so that the question requires CROSS-OBJECT or CROSS-REGION COMPARISON.
The rewritten question must ask the model to compare two or more distinct objects, regions, or attributes visible in the image and make a judgment (e.g., which is larger, which has more X, how do they differ in Y).
Do NOT just append a second question. Fuse the comparison into one focused question.
## Constraints
- The question must name or reference at least two distinct entities to compare.
- The answer must require identifying both entities and evaluating their difference/similarity.
- The new answer must differ from the original answer.
- Ensure all generated data is consistent with the image content.
- Double-check provided descriptions against the image content.
- """ + QUESTION_CONSTRAINTS + REWRITE_FORMAT

ADD_MULTIHOP_V1 = QUESTION_CONDITION + """Rewrite the given Q&A so that the question requires MULTI-HOP INFERENCE.
The rewritten question must require the model to first infer an intermediate visual fact, and then use that intermediate fact to answer the final question. The final answer cannot be obtained without completing the intermediate step.
Do NOT just append a second question. Design one question whose answer requires two or more inference hops.
## Constraints
- The question should not give away the intermediate fact directly.
- The answer must demonstrate the multi-hop chain.
- The new answer must differ from the original answer.
- Ensure all generated data is consistent with the image content.
- Double-check provided descriptions against the image content.
- """ + QUESTION_CONSTRAINTS + REWRITE_FORMAT

ADD_AREA_V1_RANDOM = QUESTION_CONDITION + """You will be given a list of positions in the image. You SHOULD complicate the given Q&A using the following method, but not limited to it: In the rewritten problem, make sure the question and its reasoning steps relies on the information from the selected area of the image, while avoiding making the problem unnecessarily lengthy. If a problem can be solved in just a few steps, rewrite the problem by adding new constraints and requirements to increase the number of reasoning steps.
## Constraints
- Achieve solving steps and answers related to the question.
- Ensure all generated data is consistent with the image content.
- Double-check provided descriptions against the image content.
- """ + QUESTION_CONSTRAINTS + REWRITE_FORMAT

ADD_TASK_V1_RANDOM = QUESTION_CONDITION + """You will be given a list of tasks that test different capabilities. You SHOULD complicate the given Q&A using the following method, but not limited to it: In the rewritten problem, make sure the question and its reasoning steps test one task capability in the list, while avoiding making the problem unnecessarily lengthy. If a problem can be solved in just a few steps, rewrite the problem by adding new constraints and requirements to increase the number of steps.
## Constraints
- Achieve solving steps and answers related to the question.
- Ensure all generated data is consistent with the image content.
- Double-check provided descriptions against the image content.
- """ + QUESTION_CONSTRAINTS + REWRITE_FORMAT

# OFFICIAL_CODE config/task_definition.json task names (used by add_task_v1_random).
TASKS = (
    "Object Detection", "Object Classification", "Attribute Recognition",
    "Pattern Recognition", "Spatial Reasoning", "Event Understanding",
    "Action Prediction", "Fact Recall", "Implicit Relation Inference",
    "Commonsense Reasoning", "Logical Reasoning", "Causal Reasoning",
    "Multi-Step Reasoning", "Abductive Reasoning", "Analogical Reasoning",
    "Constraint Satisfaction", "Social Reasoning",
)

# LOCAL_CONTROLLED_ADAPTATION: deterministic intent -> official rewrite strategy.
INTENT_REWRITE = {
    "direct_recognition": QUESTION_CONDITION + REWRITE_FORMAT,
    "comparison": ADD_COMPARISON_V1,
    "spatial_relations": ADD_AREA_V1_RANDOM,
    "scene_understanding": ADD_TASK_V1_RANDOM,
    "grounded_reasoning": ADD_MULTIHOP_V1,
}

# ---------------------------------------------------------------------------
# OFFICIAL_CODE selfjudge_utils.py — shared answer/simplify prompts
# ---------------------------------------------------------------------------
ANSWER_PROMPT = (
    "You are a helpful assistant that helps generating the answer of the question "
    "based on the given image. Directly output the final answer. "
)

SIMPLIFY_PROMPT = """You are a careful question normalizer.
    Rewrite the question into a shorter, cleaner form while preserving the same reasoning structure.
    Rules:
    - Keep it as ONE syntactic question ending with '?'.
    - Preserve all compared entities, visible evidence anchors, spatial relations, and counterfactual conditions.
    - Preserve whether the question is asking for comparison, causal explanation, multi-hop inference, or conditional reasoning.
    - Remove only incidental wording, not the core reasoning requirements.
    - Do NOT turn the question into a generic template like "What happens", "What effect", "What impact", "Which is more important", "How does X affect Y", "How does X impact Y", or "How does X influence Y".
    - Do NOT broaden the question into mood/usefulness/aesthetic/comfort commentary.
    Only output the rewritten question. Do not add any explanation or formatting.
    """

# --- validity: OFFICIAL_CODE selfjudge_utils.hard_constraints() -------------
HARD_CONSTRAINTS_PROMPT = """
    You are a strict JSON evaluation engine. Your output MUST be a SINGLE, minified JSON object.
    DO NOT include any text, prose, explanations, or conversational preamble before or after the JSON.
    Your entire output must start with { and end with }.

    --- INPUTS TO ANALYZE ---
    - generated_question (Q)

    --- REQUIRED OUTPUT JSON SCHEMA (Fill this structure) ---
    {
    "hard_constraints": {
        "Factuality": { "count": 0, "evidence": [] },
        "Ambiguity":  { "count": 0, "evidence": [] },
        "Validity": {
        "answerable": true,
        "missing_info": [],
        "invalid_category": "",
        "evidence": []
        }
    }
    }

    --- GENERAL RULES ---
    1.  **Evidence Objects:** All `evidence` lists must contain objects with this exact format:
        {
        "span_text":"<verbatim cue>",
        "char_span":[start,end],
        "type":"<label>",
        "why":"<one-sentence rationale tied to this cue>"
        }
    2.  **Verbatim Spans:** `span_text` MUST be the minimal verbatim substring from 'Q' or 'C'.
    3.  **Accurate Spans:** `char_span` MUST match the `span_text` location exactly.
    4.  **Empty Evidence:** If no items are found for a category, use `count: 0` and `evidence: []`.

    --- DETAILED FILLING RULES (BY KEY) ---

    1.  hard_constraints.Factuality
        Include **unsupported or false assumptions** in Q:
        - presupposed facts not supported by the image
        - over-specific details
        - **references to nonexistent objects**
        type ∈ {"presupposition","contradiction","unsupported_detail","nonexistent_object"}
        - span_text: The exact problematic phrase.
        - why: Say why it's likely hallucinated or unsupported by C.

    2.  hard_constraints.Ambiguity
        Include vague or unclear references.
        type ∈ {"vague_pronoun","underspecified","polysemy","ellipsis","scope_ambiguity"}
        - span_text: The ambiguous token/phrase.
        - why: State what is missing.

    3.  hard_constraints.Validity
        A question is INVALID only if the required information cannot be inferred from the image in principle. Set `"answerable": false` and assign an `"invalid_category"` when Q requires:
        - **nonvisual** → sound, inner material, internal design, ...
        - **hidden** → behind/inside/occluded elements, ...
        - **external** → historical facts, artist name, cultural meaning, rituals, ...
        Add each problematic phrase under evidence with type="unanswerable_cue".
        List missing high-level info types in `missing_info`.

    FINAL REMINDER: Output JSON only. Start with {. End with }. No extra text.
    """

# --- perception: OFFICIAL_CODE selfjudge_utils.soft_constraints_extracted() --
SOFT_CONSTRAINTS_EXTRACTED_PROMPT = """
    You are a strict JSON evaluation engine. Your task is to categorize specific entities by their perception difficulty based on the provided image and question.

    --- OUTPUT FORMAT ---
    - Your output MUST be a SINGLE, minified JSON object.
    - NO text, markdown blocks, or explanations.
    - Start with { and end with }.

    --- DEFINITIONS ---
    - EASY: Entity is prominent, unobstructed, and immediately identifiable.
    - HARD: Entity is tiny, occluded, blurry, low-contrast, or requires deep zooming.

    --- REQUIRED JSON SCHEMA ---
    {
      "soft_constraints": {
        "perception_difficulty": {
          "image": {
            "easy_count": 0,
            "hard_count": 0,
            "categorized_entities": [
              { "entity": "name", "difficulty": "easy|hard", "reason": "short text" }
            ]
          }
        }
      }
    }

    --- INPUTS TO ANALYZE ---
    - Shared Entities (Question + Image): [INSERT_SHARED_ENTITIES]
    - Image (U): [VISUAL_INPUT]

    FINAL INSTRUCTION: Analyze every provided entity. Ensure the sum of easy_count and hard_count equals the total number of unique entities analyzed. Output JSON only.
    """

# --- reasoning: OFFICIAL_CODE selfjudge_utils.prompt_reasoning_steps_and_count()
REASONING_STEPS_PROMPT = """
    You are a step-by-step reasoning engine.

    TASK 1 — Extract Reasoning Steps
    Given a QUESTION and an IMAGE, infer the reasoning steps required to answer the question.

    A reasoning step = a single cognitive operation a human must perform
    (identify objects, compare quantities, infer cause/effect, interpret relations, integrate cues).

    Forbidden:
    - Do NOT include steps that confirm, verify, or double-check previous results.
    - Do NOT include meta steps (e.g., "look at the image", "analyze the question").
    - Do NOT include the final answer.

    TASK 2 — Count Steps
    After generating the reasoning steps, count how many steps there are.

    FINAL OUTPUT FORMAT (STRICT JSON, NO EXTRA TEXT):
    {
        "soft_constraints": {
            "reasoning_difficulty": {
                "count": <number_of_steps>,
                "evidence": ["step1", "step2", ...]
            }
        }
    }

    Where:
    - "count" = number of reasoning steps
    - "evidence" = the list of reasoning steps (in order)

    Do NOT output anything except this JSON.
    """

# OFFICIAL_CODE text-only entity extraction prompt (generation_pipeline gen_and_judge / self_judge).
TEXT_ENTITY_PROMPT = (
    "Extract ALL noun or noun-phrase entities mentioned in Q, whether abstract or concrete "
    "and saved as text entity. "
)
