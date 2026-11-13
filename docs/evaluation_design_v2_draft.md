# Evaluation specification v2 (probe specification, revision 3): evidence-based modality necessity

> 状态更新（2026-11-13）：本文是早期评估设计草案，保留作设计历史。当前的定义与论证见 [necessity_formal_argument.md](necessity_formal_argument.md)（第9节为实现约定），当前实现为 `src/egvqg/necessity_ensemble.py`。两者与本文不一致处以前者为准；`evaluation_protocol_v1.0.md` 仍为冻结协议。

Status: specification for review, 2026-11-12. Revision 3 is authorized for an eight-item first probe; execution state is recorded in PROGRESS.md. It does not modify `docs/evaluation_protocol_v1.0.md` or any historical run; if adopted it becomes a new protocol version with its own runs. Author: Claude (design only; no review labels entered). Revision 3 fixes dependency cancellation, unresolved integration and paired-control expectations before execution.

Task: given text T, image I and question Q, decide whether a complete answer to Q requires information from both T and I, and show the evidence.

## 1. Summary of the method

- Necessity is a property of **the question over all legitimate solution paths**, not of one reference solution and not of one solver's behaviour.
- A modality M is judged per answer component, and each verdict needs positive evidence:
  - **Not necessary (0):** a complete solution path that uses only the retained input.
  - **Necessary (1):** a witness of two coherent, answerable completions of M, both compatible with the retained input and the fixed Q, that give different answers.
  - **Unresolved (null):** neither is available, or the evidence was rejected.
- The reference solution verifies the answer and gives an initial evidence map. It is a starting point for the search, not the object being judged.
- Code validates structure only. Whether a premise is supported, a completion is admissible or a derivation is correct is a semantic check made by a named reviewer.
- Isolated answering is used only where the analysis is unresolved. TI/T/I/0 scoring and BDR are optional secondary diagnostics.

## 2. Definitions

"Retained input without M" is Q, the other modality and permitted knowledge K.

### 2.1 Answer components

A component is a **minimal independently assessable requirement** of Q: one requested value, choice, direction or (only if Q asks) explanation.

- Splitting test: if a response could be right on one part and wrong on another, they are separate components.
- A conjunction, a list, or "X and also Y" is always split, however the sentence is phrased.
- A part computed from another part is still its own component. Recompute its actual dependency: arithmetic cancellation, symmetry or a constant function can remove an input dependency; never inherit dependencies mechanically.
- The decomposition is part of the reviewed reference. Bundling independent answers into one component is a decomposition error, not integration.

### 2.2 Validity (V)

V=1: with T, I and K, every component has exactly one supported value (up to equivalent wording or units) and Q's presuppositions hold.
V=0: false presupposition, unsupported by the material, or two reasonable readings of Q give different answers.
V=null: unreadable evidence, or the answer depends on a scientific assumption that neither material nor Q fixes.
Necessity is evaluated only when V=1.

### 2.3 Permitted knowledge (K-v1)

Allowed:
- K1. Arithmetic, algebra, ratios, percentages, standard unit conversions.
- K2. Secondary-school subject concepts and standard formulas (definitions of concentration, density, kinetic energy; how dominance works; food-chain roles), as in v1.0 §3.
- K3. Notation and structural reading conventions universal in the subject (an uppercase allele letter denotes the dominant allele; rows and columns of a Punnett square are parental gametes; table headers name their columns).
- K4. Language and everyday world knowledge needed to read Q (object names, what a task instruction asks).

Not allowed (material-specific), by the instance-variation test "would another instance of the same kind of material plausibly state a different value?":
- N1. Readings, counts, labels and values shown in this material.
- N2. Which trait an allele maps to; species- or place-specific facts.
- N3. Experimental conditions: durations, stated equality of containers, batteries, amounts.
- N4. Encoding units and scales of a diagram (one ball = one particle).
- N5. Real sizes inferred from unscaled drawings.

Every record names the K version it used. A different K version is a different evaluation.

### 2.4 Completions and the completion space

A completion of M is a full, concrete content for the removed modality. The space of admissible completions is:

- (a) Same genre and layout as the actual M: a Punnett square stays a Punnett square for the same gene, a table keeps its variables, a hint stays a hint about the same setup.
- (b) Internally coherent. Coordinated changes inside M are allowed and expected when consistency needs them (cells of a Punnett square together with its row and column labels; a total together with its parts).
- (c) Compatible with the retained input: it contradicts nothing in Q, in the other modality, or in K, including what the other modality says about M ("both samples have the same number of particles" constrains every image completion).
- (d) Answerable: under the completion, Q is valid with a unique complete answer.

Making M blank, unreadable, self-contradictory or insufficient is not a completion and is never a witness. Q, the other modality and K are never changed.

### 2.5 Necessity, integration, combined conclusions

For component c and modality M:
- dep_M(c)=1 iff two admissible completions of M give different values for c.
- dep_M(c)=0 iff every admissible completion gives the same value for c.

Then:
- D_M = 1 if dep_M(c)=1 for some c; 0 if dep_M(c)=0 for all c; otherwise null.
- S = OR3 over AND3(dep_T(c), dep_I(c)), evaluated component by component when both modalities are required. S=0 only when every component is confirmed not jointly dependent; an unresolved potentially joint component keeps S=null. If both-required=0, S is not applicable. Thus absence of a confirmed joint component never establishes stitching.
- Both required = V AND D_T AND D_I. Integrated = both required AND S. Stitched = both required AND NOT S. (Three-valued: any 0 gives 0, all 1 gives 1, otherwise null.)

## 3. Evidence

### 3.1 Reference solution: answer verification and initial evidence map

- Gives the components and their values, and a list of supporting facts.
- A fact may have **several supported sources** (for example a volume stated in T and also printed in I). Sources are listed, not forced to one.
- It is reviewed before use. It tells the analysis where to look first; a verdict never rests on "the reference used this fact".

### 3.2 Sufficiency path (evidence for dep_M(c)=0)

An ordered list of premises reaching the value of c, each with a locator in the retained input (Q span, other-modality quote or location) or a named K item. It need not resemble the reference solution.

If the path has to reason over something only M would fix, it must include a **case analysis** with its space stated:
- `algebraic`: the quantity cancels for every value of a named parameter.
- `enumerated_exhaustive`: the listed cases are all admissible completions on that dimension, with the reason the list is complete.
- `enumerated_partial`: some cases checked. This does not support 0.

An invariance claim is only as strong as the space it states. The space is recorded and reviewed with the path.

### 3.3 Witness (evidence for dep_M(c)=1)

Two completions A and B of M (A may be the actual content), each with: its content, why it is compatible with the retained input, and the resulting value of every component. They must satisfy §2.4 and differ on c. A witness is independent of any particular solution path: if two admissible completions give different answers, no path from the retained input can fix the answer.

### 3.4 What neither form establishes

- No path found: not necessity.
- A fact absent from the retained input: not necessity, unless a witness shows the answer moves.
- A witness rejected in review: not sufficiency. The verdict becomes null unless the reviewer supplies opposite evidence of their own.
- A model search is bounded. Verdicts are evidence-supported assessments conditional on K-v1, the stated completion space and the reviewer's checks; they are not formal proofs.

### 3.5 Shortcut labels

Shortcuts are tied to a **decisive material-specific fact**: a fact of type N1–N5 that the material presents and on which the answer depends (varying it would change the answer if nothing else fixed it). Labels may overlap.

| Label | Condition |
|---|---|
| `image_only` | D_T=0 confirmed |
| `text_only` | D_I=0 confirmed |
| `question_leakage` | A decisive fact is stated or logically entailed by Q, so the modality that carries it is not needed. The record names the fact, the Q span and the modality made redundant |
| `common_knowledge` | A decisive fact the material presents is in fact fixed by K, so neither modality is needed for it. The record names the fact and the K item |
| `independent_stitching` | Both required and S=0 |

Not shortcuts by themselves: a path that uses arithmetic or a formula, Q naming the objects or the task, or a fact that turned out not to matter (invariance). An invariance case is `image_only` or `text_only` with no leakage or knowledge label.

## 4. Pipeline

| Stage | Purpose | Sees | Output | Model call |
|---|---|---|---|---|
| A. Bind | Fix identity of T, I, Q | T, I, Q | hashes, K version | no |
| B. Reference | Components, values, evidence map, stated assumptions | T, I, Q | reference candidate | 1 (0 when a reviewed reference is supplied) |
| C. Reference review | Verify answer, decomposition and image readings; set V | T, I, Q, candidate | accept / corrected / unresolved / invalid, with provenance | reviewer |
| D. Modality analysis | For one removed modality: per component, a sufficiency path or a witness; decisive facts | T, I, Q, reviewed reference | proposed dep values with evidence | 1 per modality (2 per Q) |
| E. Structural validation | Fields, bindings, statuses, conflicts | record | structural status, flags | code |
| F. Targeted isolated answering | Only for a component and modality left null: look for a path the analysis missed | Q and the retained modality only; never reference, labels or analysis | answer or refusal with premises and missing facts | 1, then 1 repeat of D with that record |
| G. Semantic review | Check each evidence item | T, I, Q, evidence items | supported / not_supported / uncertain per item, with provenance | reviewer |
| H. Verdict | Apply §5 | record | V, D_T, D_I, S, shortcuts, status | code |

Notes:
- Stage D runs once per modality. Each call carries one question (is this modality needed?) and a short output; the historical single necessity call produced about 1,000 output tokens and was truncated once at 2,048.
- Stage F is not a verdict. A correct blind answer is a candidate path whose premises go back through D and G; a blind answer that assumed a fact, a wrong answer and a refusal each only nominate something to test.
- Optional secondary diagnostics, never prerequisites: full TI/T/I/0 answering and scoring, historical BDR (V AND C_TI AND NOT C_T AND NOT C_I, refusals unresolved), and the relevance label R from `relatedness-v1`. They describe solver behaviour and relevance beside the necessity verdict.
- Stages B–D never show a solver anything; stage F never shows the solver the reference.
- Calls per Q: 2 with a supplied reviewed reference, 3 otherwise, plus 2 per unresolved modality in a question, at most one follow-up round.

## 5. Decision rules

### 5.1 Structural validation (code, stage E)

Code checks only that the record is well formed:
1. Question, reference, analysis and review are bound to the same Q hash and K version.
2. The reference has a review decision with provenance; components have unique ids.
3. Every component has, for every modality, exactly one of: a path, a witness, or an unresolved reason.
4. A path has at least one premise; every premise has a source type and a locator; no source lies in the removed modality; any case analysis has a `space` and a `closure` value.
5. A witness has two completions, each with content, a compatibility note and a value for every component, and the two values for the cited component differ as strings.
6. Each shortcut label names its decisive fact and locator.
7. Conflicts are flagged: a path and a witness for the same component and modality; a witness for M while a stage-F answer without M is recorded correct with no assumed premise.

Passing these checks certifies nothing about truth. A structurally complete record with unreviewed evidence has status `proposed`.

### 5.2 Semantic review (reviewer, stage G)

For each evidence item the reviewer records `supported`, `not_supported` or `uncertain`, with source `human` or `user_delegated_ai`:
- Path: each locator really contains the premise; each K premise is in K-v1; the derivation reaches the stated value; any case analysis covers the stated space and the space is the right one.
- Witness: each completion meets §2.4 (a)–(d); each stated value is correct under its completion.
- Decisive fact: it is of type N1–N5, the answer depends on it, and it is where the label says.
- Decomposition: components are minimal (§2.1).

### 5.3 Verdict (code, stage H)

```
if reference review != verified:  V = 0 (invalid) or null (unresolved); necessity not evaluated
for each modality M, component c:
    evidence = the path or witness for (M, c)
    if structural check fails or a conflict is flagged:      dep = null   (reason: structural | conflict)
    elif evidence is unreviewed:                             dep = proposed value, status "proposed"
    elif review == supported:                                dep = 0 for a path, 1 for a witness, status "confirmed"
    elif review == not_supported:                            dep = null   (reason: evidence_rejected)
    else:                                                    dep = null   (reason: review_uncertain)
    a path whose case analysis is enumerated_partial never yields 0
D_M, S, both_required, integrated: as in §2.5, over confirmed values only
shortcuts: as in §3.5, only from confirmed evidence
```

Reported results use confirmed values. Proposed values are a screen and are reported separately, never mixed into the same count. Every planned Q stays in the denominator; null is reported as unresolved with the `[pass/N, (pass+unresolved)/N]` bounds; no new headline metric is added.

## 6. Worked examples

All three use development materials whose images were inspected for this draft.

### 6.1 Necessary: koi eye colour (`scienceqa_40`)

- T: "…The allele for red eyes (E) is dominant over the allele for black eyes (e). This Punnett square shows a cross between two koi fish."
- I: top gametes e, e; left gametes E, E; all four cells Ee.
- Q: 根据图中所示的杂交，子代锦鲤表现为什么眼色？
- Component c1: offspring eye colour = red.

| Modality | Completion A (actual) | Completion B | Compatibility | c1 under A / B |
|---|---|---|---|---|
| T | E is the red-eye allele, dominant | "The allele for black eyes (E) is dominant over the allele for red eyes (e)" | I shows only letters; Q names no colour; uppercase = dominant kept (K3) | red / black |
| I | gametes e,e × E,E; cells all Ee | gametes e,e × e,e; cells all ee (labels and cells changed together) | T says only that a cross between two koi is shown | red / black |

Result: dep_T(c1)=1, dep_I(c1)=1, so D_T=1, D_I=1, S=1. Both required and integrated. No shortcut label.

### 6.2 Unnecessary: guppy spotted-tail probability (`scienceqa_1992`)

- T: "…The allele for a spotted tail (I) is dominant over the allele for an unspotted tail (i). This Punnett square shows a cross between two guppies."
- I: top gametes i, i; left gametes I, i; cells Ii, Ii, ii, ii.
- Q: 根据图中所示的杂交，子代出现斑点尾的概率是多少？
- Component c1: probability = 1/2.

Text, sufficiency path with case analysis:
1. Two cells are Ii and two are ii (I, inner cells).
2. Under complete dominance Ii shows the trait of I and ii the trait of i, so the two genotypes show different traits (K2, K3).
3. Space: which of the two tail traits allele I carries; `enumerated_exhaustive` because Q names a two-state trait and the genre is single-gene complete dominance. Case "I = spotted": Ii spotted, 2 of 4. Case "I = unspotted": ii spotted, 2 of 4.
4. Value 1/2 in every case.

Image, witness: A actual (I,i × i,i) gives 1/2; B gametes I,I × i,i with all cells Ii gives 1. Both compatible with T.

Result: dep_T(c1)=0, dep_I(c1)=1, so D_T=0, D_I=1. Not both required. Shortcut `image_only`; no leakage and no common-knowledge label, because no decisive fact was supplied or substituted. The 0 holds within the stated space; a reviewer who holds that incomplete dominance is an admissible completion for this genre would mark the case analysis `not_supported`, and the verdict would become null.

### 6.3 Unresolved: gas pressure ratio (`scienceqa_19`)

- T: identical closed rigid containers; each ball is one particle; same number of particles.
- I: masses 28 u and 44 u; average particle speed 1,300 m/s for both.
- Q: 样品B与样品A的气体压强之比是多少？

The ratio 11:7 follows only if both samples share the same relation between mean speed and mean-square speed (for example equilibrium distributions). Neither T, I nor Q states that, and it is not in K-v1. Reference review: `unresolved`, reason `scientific_assumption`. V=null; no modality analysis is run and no necessity label is issued. The item stays in the denominator as unresolved. (The same material supports a valid joint question, the mass-density ratio, recorded separately by Codex as a witness question.)

## 7. Minimal JSON record

Example 6.2, abridged to the fields the rules use:

```json
{
  "question_sha256": "…", "knowledge_version": "K-v1",
  "reference": {
    "review": {"decision": "accept_candidate", "source": "user_delegated_ai", "reviewer": "…"},
    "components": [{"id": "c1", "asks": "probability of a spotted tail", "value": "1/2"}],
    "evidence_map": [
      {"id": "f1", "fact": "cells are Ii, Ii, ii, ii", "sources": [{"in": "I", "locator": "four inner cells"}]},
      {"id": "f2", "fact": "I (spotted) is dominant over i (unspotted)", "sources": [{"in": "T", "locator": "The allele for a spotted tail (I) is dominant…"}]}
    ],
    "assumptions": []
  },
  "analysis": {
    "T": {"c1": {
      "proposed": 0,
      "path": {
        "premises": [
          {"statement": "two cells Ii, two cells ii", "source": "I", "locator": "inner cells"},
          {"statement": "Ii and ii show different traits under complete dominance", "source": "K", "locator": "K2,K3"}
        ],
        "case_analysis": {"space": "which tail trait allele I carries", "closure": "enumerated_exhaustive",
          "closure_reason": "two-state trait named in Q; single-gene complete dominance",
          "cases": [{"case": "I = spotted", "value": "1/2"}, {"case": "I = unspotted", "value": "1/2"}]}
      },
      "review": {"result": "supported", "source": "user_delegated_ai", "reviewer": "…"}
    }},
    "I": {"c1": {
      "proposed": 1,
      "witness": {
        "completion_a": {"content": "actual: gametes I,i × i,i; cells Ii,Ii,ii,ii", "compatible_because": "actual", "values": {"c1": "1/2"}},
        "completion_b": {"content": "gametes I,I × i,i; cells Ii,Ii,Ii,Ii", "compatible_because": "T states only that a cross is shown", "values": {"c1": "1"}}
      },
      "review": {"result": "supported", "source": "user_delegated_ai", "reviewer": "…"}
    }}
  },
  "isolated_answers": [],
  "structural": {"status": "pass", "flags": []},
  "verdict": {"validity": 1, "text_necessary": 0, "image_necessary": 1, "integration": "not_applicable",
              "both_required": 0, "integrated": 0,
              "shortcuts": [{"label": "image_only"}], "status": "confirmed"}
}
```

A leakage label would read `{"label": "question_leakage", "decisive_fact": "E (red) is dominant", "question_span": "已知红眼等位基因E对黑眼等位基因e为显性", "redundant_modality": "T"}`.

## 8. Prompt templates

`score-v1` and `relatedness-v1` are unchanged and optional. Outputs use the JSON fields of §7.

### `answer-v2` (stage B; sees T, I, Q)

```
为当前问题建立可核验的参考解答。不沿用数据集原题答案。
文本：<T>$T</T>
图像通过图像输入提供。
问题：<Q>$Q</Q>
1. 把Q的要求拆成最小的、可以单独判对错的答案要点。并列的多个所求必须分开；Q没有要求解释时不列解释。
2. 给出每个要点的值。
3. 列出支持这些值的事实。同一事实在多处都有依据时，把每一处来源都列出：T（原文引文）、I（图中位置和读数）、Q（题干片段）、K（通常中学学科知识或运算）。
   换一份同类材料就可能不同的事实（读数、某等位基因对应的性状、图例编码单位、实验时长等）不得标为K。
4. 列出为得到唯一答案所采用的解释或科学假设；不同的合理解释会得到不同答案时，status写ambiguous。
5. 图像读数不确定时写uncertain，不猜补。
只输出JSON：{"status":"answerable或ambiguous或unsupported或unreadable",
 "components":[{"id":"c1","asks":"所求","value":"值或null"}],
 "evidence_map":[{"id":"f1","fact":"事实","sources":[{"in":"T或I或Q或K","locator":"引文/位置/片段/知识内容","reading":"clear或uncertain"}]}],
 "assumptions":[],"acceptable_answers":[],"issues":[]}
```

### `modality-necessity-v2` (stage D; one call per removed modality; sees T, I, Q, reviewed reference)

```
审核：完整回答当前问题是否必须使用$REMOVED的信息。保留输入是：问题Q、$RETAINED、通常学科知识。
结论只能来自下面两种证据之一，不凭印象回答“需要”或“不需要”。参考解答只用于核对答案和提示可能相关的事实；其他合理解法同样算数。
文本：<T>$T</T>
图像通过图像输入提供。
问题：<Q>$Q</Q>
已审核的参考解答：$REFERENCE
$ISOLATED_RECORD
对每个答案要点分别给出下列之一：
A. 不必要的证据：只用保留输入得到该要点的完整解法。每条前提写明来源和定位，来源不得是$REMOVED。
   如果解法需要对只有$REMOVED才能确定的内容分情况讨论，写出讨论的范围space、范围是否穷尽closure（algebraic、enumerated_exhaustive、enumerated_partial）及理由，并逐一给出每种情况下的值。
B. 必要的证据：$REMOVED的两个完整内容（其中一个可以是实际内容）。每个内容必须：与实际内容同类同结构；自身一致，需要时可协调修改多处；不与Q、$RETAINED和通常学科知识矛盾；使Q有唯一完整答案。分别写出两种内容下各要点的值，至少该要点的值不同。把$REMOVED变成空白、不可读、自相矛盾或信息不足，不算证据。
C. 两者都给不出时写unresolved及原因。没有找到解法不等于必要；某事实缺失也不等于必要。
另外列出决定性材料事实：材料中给出、答案确实依赖的特定事实。若它已被Q直接说出，或其实属于通常学科知识，写明该事实及其在Q中的片段或对应知识；普通运算、公式、对象名称和任务指令不算。
每项用简短、可核对的条目，不写长篇推理。
只输出JSON：{"removed":"T或I","components":{"c1":{"proposed":1或0或null,
  "path":null或{"premises":[{"statement":"前提","source":"Q或T或I或K","locator":"定位"}],"case_analysis":null或{"space":"范围","closure":"…","closure_reason":"理由","cases":[{"case":"情况","value":"值"}]}},
  "witness":null或{"completion_a":{"content":"内容","compatible_because":"依据","values":{"c1":"值"}},"completion_b":{同上}},
  "unresolved_reason":null}},
 "decisive_facts":[{"fact":"事实","carried_by":"T或I","supplied_by":"none或question或general_knowledge","locator":"Q片段或知识内容"}]}
```

`$ISOLATED_RECORD` is empty in the first pass. In a targeted repeat it holds the stage-F record, introduced as "缺$REMOVED时的一次隔离作答，仅作线索；逐条核对其前提是否确实来自保留输入，答对、答错、拒答本身都不是结论".

### `solve-v2` (stage F only; isolated)

```
请回答下面的问题。
可以使用通常学科知识和本次实际提供的材料；不要假设未提供的特定事实。
能够确定时作答；无法确定时明确表示无法确定，并指出缺少的具体事实。
可见材料：
$VISIBLE_MATERIALS
问题：
<Q>$Q</Q>
列出实际用到的前提及来源：given_text、given_image、question、general_knowledge；自行设定而未被提供的前提，来源必须写assumed。
只输出JSON：{"answer":null或"答案","abstain":true或false,"premises":[{"statement":"前提","source":"来源","locator":"定位或null"}],"missing_facts":[],"reason_brief":"简短依据"}
```

## 9. First probe: eight controlled items

Purpose: test whether stages D, E, G and H produce correct verdicts with valid evidence on cases whose answer is known by construction. It tests the evaluation method only.

Items are hand-written on development materials (none in the formal 30). Each comes with a reviewed reference (components and values), so stage B is not called.

| ID | Material | Q | Expected | What it tests |
|---|---|---|---|---|
| E1 | koi `40` | 根据图中所示的杂交，子代锦鲤表现为什么眼色？ | D_T=1, D_I=1, S=1 | Witness on both sides, with coordinated image changes |
| E2 | koi `40` | 已知红眼等位基因E对黑眼等位基因e为显性。根据图中所示的杂交，子代锦鲤表现为什么眼色？ | D_T=0, D_I=1; leakage of "E = red, dominant" | Pair of E1: only D_T may differ; leakage names the decisive fact |
| E3 | guppy `1992` | 根据图中所示的杂交，子代出现斑点尾的概率是多少？ | D_T=0 by exhaustive case analysis, D_I=1; no leakage or knowledge label | Absent fact that does not move the answer; stated space |
| E4 | solution `192` | 图中 Solution A 的模型共表示多少个溶质粒子？ | D_T=1 (each ball = 1 particle vs 2 gives 9 vs 18), D_I=1, S=1 | Same legend fact as a comparison question, but decisive here; K-v1 item N4 |
| E5 | battery `15694` | 关机冷却期间，两块电池的温度各下降了多少摄氏度？ | Two components (13, 10); D_T=0 by a path from the table, D_I=1 | Irrelevant stated condition; splitting into components |
| E6 | battery `15694` | 关机冷却期间，两块电池的平均降温速率相差多少℃/分钟？ | D_T=1 (5 min vs 10 min gives 0.6 vs 0.3), D_I=1, S=1 | Pair of E5: the duration becomes decisive |
| E7 | koi `40` | 哪种眼色的等位基因是显性的，图中左上格子代的基因型是什么？ | c1 depends on T only, c2 on I only; D_T=1, D_I=1, S=0 | Bundled phrasing must not turn stitching into integration |
| E8 | gas `19` | 样品B与样品A的气体压强之比是多少？ | V=null; no necessity labels; zero calls | Unresolved validity propagates and spends nothing |

Not covered here and left for a second probe: a common-knowledge case, a text-only-sufficient question, a false-presupposition question, and scripted "refusal despite enough information" and "correct guess" records for stage F. E3, E4, E7 and E8 are the items most likely to expose a weakness; they are included on purpose.

**Expected labels.** The table above is the author's (Claude's) construction, not a review. Before any run, a reviewer checks the eight expected verdicts and their written witnesses or paths against T and I and seals them. Provenance is recorded as `human` (you or a colleague; about 20 minutes for eight items) or `user_delegated_ai` (a delegated reviewer who did not write them). Only the first supports a claim of human-checked expected labels.

**Procedure.**
1. Seal expected labels and the supplied references.
2. Stage D: two calls for each of E1–E7.
3. Stage E, then stage F only for a component and modality left null.
4. Stage G: the reviewer checks every evidence item, without the expected labels in view.
5. Stage H, then compare with the sealed labels.

**Call budget.** 14 planned calls (7 items × 2). Targeted follow-up costs 2 calls per unresolved modality in a question (one isolated answer and one reanalysis covering that question’s components). Hard cap 24 HTTP requests for the whole probe, cumulative across resumes. At the observed 3.3–4.0 s per request for short calls and about 8.6 s for analysis-sized calls, model time is roughly 2–4 minutes. No full TI/T/I/0 answering, scoring or BDR is run.

**Results that reveal a defect.**
1. A confirmed verdict that contradicts a sealed expected label.
2. A proposed witness the reviewer rejects because a completion is incoherent, incompatible with the retained input, unanswerable, or does not change the answer.
3. A proposed path the reviewer rejects because a premise is not where it claims to be, is not in K-v1, or its case analysis is not exhaustive for the stated space.
4. E2 failing to retain E1’s validity, image necessity and answer; or failing to reduce text necessity and add the decisive leakage label. Both-required, integrated and integration applicability legitimately change.
5. E7 reported as integrated.
6. A leakage or common-knowledge label on E3, E5 or any item where nothing decisive was supplied.
7. Any call issued for E8, or a necessity label on it.
8. More than two component-modality pairs left null among E1–E7: too timid to be useful.

Disagreements are classified on the evidence as evaluator error, expected-label error or definitional boundary. Expected-label corrections are logged as amendments beside the sealed file.

**What would justify the next step.** No defect of types 1–7. The next step is then the second probe (the uncovered cases above), followed by the questions already generated and delegated-reviewed in earlier runs; no new generation is needed for either.

## 10. Engineering changes and limits

Changes needed to execute this specification (not started):
1. `prompts/answer-v2.txt`, `modality-necessity-v2.txt`, `solve-v2.txt`.
2. `configs/evaluation_v2.json` with a new `protocol_version` and `knowledge_version`.
3. Imported cases may carry a reviewed reference with provenance; the reference call is then skipped.
4. `experiments.py`: two analysis steps (one per modality) storing parsed output under `record['analysis']`; a targeted step that runs `solve-v2` and one repeat analysis only for null component-modality pairs.
5. `metrics.py`: `validate_structure(record)` and `decide_necessity(record)` as pure functions implementing §5.1 and §5.3.
6. Evidence review sheet: export evidence items, import per-item review results with provenance (extension of the existing review import).
7. Tests: the §5 rules as a truth table, including "complete JSON without review stays proposed", "rejected witness becomes null, not 0", "partial enumeration never yields 0", bundled components, and blind-input isolation for `solve-v2`.

Limits:
- Admissibility and the choice of completion space are judgments. The specification makes them explicit and reviewable; it does not remove them.
- A 0 verdict by case analysis holds only within its stated space; a 1 verdict holds only if both completions are accepted as admissible.
- The analysis model sees the reviewed reference and may anchor on it. The instruction that other solution paths count, the targeted isolated answer and the reviewer are the counterweights.
- All model stages share one model's vision; only the reviewer's check of image readings is independent.
- Image completions are described, not rendered. Sibling ScienceQA items with the same text and a different image could later serve as real image completions.
- "Confirmed" means confirmed by the named reviewer source. With delegated AI review it is not a human gold standard.

Open choices:
1. K3 treats "uppercase letter = dominant allele" as general knowledge. If you prefer it to be material-specific, more text completions become admissible (for example "e is dominant"); the expected verdicts of this probe do not change.
2. N4 treats diagram encoding units as material-specific (as v1.0 does). E4 depends on this.
3. An explicitly requested explanation is an answer requirement; unrequested explanations are not added. Multiple legitimate phrasings are evaluated by semantic equivalence, not exact string identity.
4. Who reviews the eight expected labels and the evidence items.


### First-probe operational commitments

K-v1-probe1 explicitly scopes genetic completions to a single-gene, two-phenotype complete-dominance model; uppercase denotes dominance in this declared notation. This is a conditional evaluation space, not a claim about all genetic phenomena. Encoding scales remain material-specific. Expected labels and references are reviewed by Codex under the user’s prior delegation, with prior exposure stated; no independent-human or blind-review claim is made. E8 has supplied validity=null and spends zero analysis calls. Every question remains in N=8. Code checks strings only as a structural screen; semantic equivalence and admissibility require evidence review. First-round results will separate model proposals, rejected evidence and final reviewed judgments. No generation, BDR or semantic resampling is run.
