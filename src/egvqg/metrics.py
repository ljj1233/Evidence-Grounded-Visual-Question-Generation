"""Fixed-denominator metrics: validity, tri-state BDR, condition coverage and no-material accuracy."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

from .evaluation import bdr_state
from .records import (
    CONDITIONS,
    AnswerStatus,
    BDRState,
    QuestionStatus,
    ScoreStatus,
    ValidityStatus,
)


def _index_latest(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        index[str(row[key])] = row
    return index


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _aggregate(
    slots: list[Mapping[str, Any]],
    *,
    questions: Mapping[str, Mapping[str, Any]],
    validity: Mapping[str, Mapping[str, Any]],
    answers: Mapping[str, Mapping[str, Any]],
    responses: Mapping[tuple[str, str], Mapping[str, Any]],
    scores: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    n_planned = len(slots)
    n_valid = 0
    n_invalid = 0
    n_unresolved = 0
    n_pending = 0
    n_invalid_confirmed = 0
    n_generation_failure = 0
    n_technical_failure = 0

    n_pass = 0
    n_bdr_fail = 0
    n_bdr_unresolved = 0
    bdr_reasons: Counter[str] = Counter()

    n_generated = 0
    condition_generated = {condition: 0 for condition in CONDITIONS}
    condition_resolved = {condition: 0 for condition in CONDITIONS}
    condition_correct = {condition: 0 for condition in CONDITIONS}

    n_valid_verified = 0
    n_no_material_eligible = 0
    n_no_material_correct = 0
    n_no_material_unresolved = 0

    for slot in slots:
        question_id = str(slot["question_id"])
        question = questions.get(question_id)
        validity_row = validity.get(question_id)
        answer_row = answers.get(question_id)

        if question is None:
            validity_state = "pending"
            n_pending += 1
        elif question["status"] != QuestionStatus.OK:
            validity_state = ValidityStatus.INVALID
            n_invalid += 1
            if question["status"] == QuestionStatus.GENERATION_FAILURE:
                n_generation_failure += 1
            elif question["status"] == QuestionStatus.TECHNICAL_FAILURE:
                n_technical_failure += 1
        elif validity_row is None:
            validity_state = ValidityStatus.UNRESOLVED
            n_unresolved += 1
        elif validity_row["status"] == ValidityStatus.VALID:
            validity_state = ValidityStatus.VALID
            n_valid += 1
        elif validity_row["status"] == ValidityStatus.INVALID:
            validity_state = ValidityStatus.INVALID
            n_invalid += 1
            n_invalid_confirmed += 1
        else:
            validity_state = ValidityStatus.UNRESOLVED
            n_unresolved += 1

        if question is not None and question["status"] == QuestionStatus.OK:
            n_generated += 1

        scores_by_condition: dict[str, bool | None] = {}
        for condition in CONDITIONS:
            response = responses.get((question_id, condition))
            score = scores.get(response["response_id"]) if response else None
            value: bool | None = None
            if score is not None and score["status"] == ScoreStatus.SCORED and isinstance(score.get("correct"), bool):
                value = score["correct"]
            scores_by_condition[condition] = value

            if question is not None and question["status"] == QuestionStatus.OK:
                condition_generated[condition] += 1
                if value is not None:
                    condition_resolved[condition] += 1
                    if value:
                        condition_correct[condition] += 1

        bdr, reason = bdr_state(question, validity_row, scores_by_condition)
        bdr_reasons[reason] += 1
        if bdr == BDRState.PASS:
            n_pass += 1
        elif bdr == BDRState.FAIL:
            n_bdr_fail += 1
        else:
            n_bdr_unresolved += 1

        if validity_state == ValidityStatus.VALID and answer_row and answer_row["status"] == AnswerStatus.VERIFIED:
            n_valid_verified += 1
            if scores_by_condition["0"] is not None:
                n_no_material_eligible += 1
                if scores_by_condition["0"]:
                    n_no_material_correct += 1
            else:
                n_no_material_unresolved += 1

    conditions = {}
    for condition in CONDITIONS:
        conditions[condition] = {
            "n_generated": condition_generated[condition],
            "n_resolved": condition_resolved[condition],
            "n_correct": condition_correct[condition],
            "coverage_of_planned": _rate(condition_resolved[condition], n_planned),
            "coverage_of_generated": _rate(condition_resolved[condition], condition_generated[condition]),
            "accuracy_on_resolved": _rate(condition_correct[condition], condition_resolved[condition]),
        }

    return {
        "n_planned": n_planned,
        "n_generated": n_generated,
        "validity": {
            "n_valid": n_valid,
            "n_invalid": n_invalid,
            "n_unresolved": n_unresolved,
            "n_pending": n_pending,
            "n_adjudicated": n_valid + n_invalid,
            "n_invalid_confirmed": n_invalid_confirmed,
            "n_generation_failure": n_generation_failure,
            "n_technical_failure": n_technical_failure,
            "valid_rate": _rate(n_valid, n_planned),
            "valid_rate_upper_with_unresolved": _rate(n_valid + n_unresolved + n_pending, n_planned),
        },
        "bdr": {
            "n_pass": n_pass,
            "n_fail": n_bdr_fail,
            "n_unresolved": n_bdr_unresolved,
            "n_resolved": n_pass + n_bdr_fail,
            "bdr": _rate(n_pass, n_planned),
            "bdr_upper_with_unresolved": _rate(n_pass + n_bdr_unresolved, n_planned),
            "coverage_of_planned": _rate(n_pass + n_bdr_fail, n_planned),
            "reasons": dict(sorted(bdr_reasons.items())),
        },
        "no_material": {
            "n_valid_verified": n_valid_verified,
            "n_eligible": n_no_material_eligible,
            "n_correct": n_no_material_correct,
            "n_unresolved": n_no_material_unresolved,
            "accuracy": _rate(n_no_material_correct, n_no_material_eligible),
            "denominator_share_of_planned": _rate(n_no_material_eligible, n_planned),
            "coverage_of_valid_verified": _rate(n_no_material_eligible, n_valid_verified),
        },
        "conditions": conditions,
    }


def compute_metrics(
    plan: list[Mapping[str, Any]],
    questions: Iterable[Mapping[str, Any]],
    validity: Iterable[Mapping[str, Any]],
    answers: Iterable[Mapping[str, Any]],
    responses: Iterable[Mapping[str, Any]],
    scores: Iterable[Mapping[str, Any]],
    *,
    run_id: str,
    protocol_version: str,
    evidence_class: str,
) -> dict[str, Any]:
    questions_by_id = _index_latest(questions, "question_id")
    validity_by_question = _index_latest(validity, "question_id")
    answers_by_question = _index_latest(answers, "question_id")
    responses_by_question_condition = {
        (str(row["question_id"]), str(row["condition"])): row for row in responses
    }
    scores_by_response = _index_latest(scores, "response_id")

    overall = _aggregate(
        plan,
        questions=questions_by_id,
        validity=validity_by_question,
        answers=answers_by_question,
        responses=responses_by_question_condition,
        scores=scores_by_response,
    )

    methods: list[str] = []
    for slot in plan:
        method = str(slot["method"])
        if method not in methods:
            methods.append(method)
    by_method = {
        method: _aggregate(
            [slot for slot in plan if slot["method"] == method],
            questions=questions_by_id,
            validity=validity_by_question,
            answers=answers_by_question,
            responses=responses_by_question_condition,
            scores=scores_by_response,
        )
        for method in methods
    }

    synthetic = "synthetic" in evidence_class.lower() or evidence_class.lower().startswith("mock")
    return {
        "run_id": run_id,
        "protocol_version": protocol_version,
        "evidence_class": evidence_class,
        "synthetic": synthetic,
        "warning": (
            "SYNTHETIC FIXTURE — model-scoring integration; source Q/A are mock, not research findings."
            if evidence_class == "synthetic_fixture_model_scoring"
            else
            "SYNTHETIC MOCK — engineering fixture only; not a model result or research finding."
            if synthetic
            else "Metrics are derived from the records in this run."
        ),
        **overall,
        "by_method": by_method,
    }


def _fraction(numerator: int, denominator: int) -> str:
    if not denominator:
        return "0/0"
    return f"{numerator}/{denominator}"


def _percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def render_summary(metrics: Mapping[str, Any]) -> str:
    validity = metrics["validity"]
    bdr = metrics["bdr"]
    no_material = metrics["no_material"]
    n = metrics["n_planned"]
    lines = [
        "# 问题生成与评估汇总",
        "",
        f"> **{metrics['warning']}**",
        "",
        f"- run_id: `{metrics['run_id']}`",
        f"- protocol_version: `{metrics['protocol_version']}`",
        f"- evidence_class: `{metrics['evidence_class']}`",
        f"- 预先安排的生成尝试数 N: **{n}**",
        "",
        "## 总体指标",
        "",
        "| 指标 | 数值 | 未决/边界 |",
        "|---|---:|---|",
        (
            f"| 有效问题率 | {_fraction(validity['n_valid'], n)} "
            f"({_percent(validity['valid_rate'])}) "
            f"| 未决 {validity['n_unresolved']}；上界 "
            f"{_percent(validity['valid_rate_upper_with_unresolved'])} |"
        ),
        (
            f"| BDR | {_fraction(bdr['n_pass'], n)} ({_percent(bdr['bdr'])}) "
            f"| 未决 {bdr['n_unresolved']}；上界 "
            f"{_percent(bdr['bdr_upper_with_unresolved'])} |"
        ),
        (
            f"| 无材料作答正确率 | {_fraction(no_material['n_correct'], no_material['n_eligible'])} "
            f"({_percent(no_material['accuracy'])}) "
            f"| 有效且可评分分母占 N 的 {_percent(no_material['denominator_share_of_planned'])}；"
            f"未决 {no_material['n_unresolved']} |"
        ),
        "",
        "无效/无有效题分解："
        f"确认无效 {validity['n_invalid_confirmed']}，"
        f"生成失败 {validity['n_generation_failure']}，"
        f"技术失败 {validity['n_technical_failure']}；"
        f"有效 {validity['n_valid']}，未决 {validity['n_unresolved']}，"
        f"待完成 {validity['n_pending']}。",
        "",
        "## 分方法",
        "",
        "| 方法 | N | 有效 | 无效/无有效 | 未决 | BDR | 无材料正确率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in metrics["by_method"].items():
        method_validity = row["validity"]
        method_bdr = row["bdr"]
        method_no_material = row["no_material"]
        lines.append(
            f"| {method} | {row['n_planned']} "
            f"| {_fraction(method_validity['n_valid'], row['n_planned'])} "
            f"| {method_validity['n_invalid']} "
            f"| {method_validity['n_unresolved']} "
            f"| {_fraction(method_bdr['n_pass'], row['n_planned'])} "
            f"| {_fraction(method_no_material['n_correct'], method_no_material['n_eligible'])} "
            f"({_percent(method_no_material['accuracy'])}) |"
        )

    lines.extend(
        [
            "",
            "## 四条件评分覆盖",
            "",
            "| 条件 | 已评分/已生成 | 覆盖率（已生成） | 覆盖率（全部 N） | 已评分正确率 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for condition in CONDITIONS:
        row = metrics["conditions"][condition]
        lines.append(
            f"| {condition} | {row['n_resolved']}/{row['n_generated']} "
            f"| {_percent(row['coverage_of_generated'])} "
            f"| {_percent(row['coverage_of_planned'])} "
            f"| {_percent(row['accuracy_on_resolved'])} |"
        )

    lines.extend(
        [
            "",
            "未决与失败均保留在固定分母中；未决值不按答错处理。",
            "本汇总只验证记录、隔离、分母与恢复逻辑，不能作为方法有效性或真实模型表现的证据。",
            "",
        ]
    )
    return "\n".join(lines)


# The new protocol keeps every metric on N. Legacy compute_metrics above retains
# its historical denominators so existing runs are never silently reinterpreted.
def _three_and(*values):
    if any(v == 0 for v in values):
        return 0
    return 1 if all(v == 1 for v in values) else None


def _three_not(value):
    return None if value is None else 1 - value


def _label_value(item):
    if not isinstance(item, dict) or item.get('status') != 'decided':
        return None
    value = item.get('value')
    if type(value) not in (int, bool) or value not in (0, 1):
        raise ValueError('decided label must be 0 or 1')
    return int(value)


def record_states(record, source='automatic'):
    """Three-valued states; reject stale answer/score bindings before aggregating."""
    from .records import canonical_json, sha256_text
    q = record['question']; ref = record['reference']
    absent = not q.get('normalized_text') and record['generation']['status'] in (
        'generation_failure', 'technical_failure', 'no_question')
    if q.get('normalized_text'):
        if q['sha256'] != sha256_text(q['normalized_text']):
            raise ValueError('question hash mismatch')
        if ref.get('question_sha256') != q['sha256'] or ref.get('question_id') != q['question_id']:
            raise ValueError('reference is bound to another question')
    labels = record['labels'][source]
    v = 0 if absent else _label_value(labels['validity'])
    acceptable_statuses = ('verified', 'verified_automatic') if source == 'automatic' else ('verified',)
    verified = ref['status'] in acceptable_statuses
    reference_sha = sha256_text(canonical_json({k: ref[k] for k in ('verified_answer', 'acceptable_answers', 'rubric')}))
    scores = {}
    for c in CONDITIONS:
        r = record['responses'][c]
        score = None
        if r:
            if r['question_id'] != q['question_id'] or r['question_sha256'] != q['sha256']:
                raise ValueError('response is bound to another question')
            if verified and r['status'] == 'answered' and not r.get('abstain') and r.get('score_status') == 'scored':
                if r.get('score_reference_sha256') != reference_sha:
                    raise ValueError('score reference changed; re-score affected responses')
                value = r.get('correct')
                if type(value) not in (int, bool) or value not in (0, 1):
                    raise ValueError('scored response must have 0/1 correctness')
                score = int(value)
        scores[c] = score
    b = _three_and(v, scores['TI'], _three_not(scores['T']), _three_not(scores['I']))
    shortcuts = {k: _three_and(v, _label_value(item)) for k, item in record['shortcuts'][source].items()}
    shortcut_values = list(shortcuts.values())
    any_shortcut = 1 if 1 in shortcut_values else (0 if all(x == 0 for x in shortcut_values) else None)
    return {'validity': v, 'bdr': b, 'conditions': {c: _three_and(v, scores[c]) for c in CONDITIONS},
            'shortcuts': shortcuts, 'any_shortcut': any_shortcut,
            'labels': {k: (0 if absent or item.get('status')=='not_applicable' else _label_value(item)) for k, item in labels.items()}}


def _fixed_summary(values, n, *, n_not_applicable=0):
    values = list(values)
    if len(values) != n:
        raise ValueError('metric lost planned attempts')
    passed, failed = values.count(1), values.count(0)
    unresolved = values.count(None)
    return {'n_planned': n, 'n_pass': passed, 'n_fail': failed - n_not_applicable,
            'n_not_applicable': n_not_applicable, 'n_unresolved': unresolved,
            'rate': _rate(passed, n), 'unresolved_rate': _rate(unresolved, n),
            'lower': _rate(passed, n), 'upper': _rate(passed + unresolved, n)}


def compute_record_metrics(rows, *, source='automatic'):
    n = len(rows); states = [record_states(r, source) for r in rows]
    absent = sum(not r['question']['normalized_text'] and r['generation']['status'] in (
        'generation_failure', 'technical_failure', 'no_question') for r in rows)
    result = {key: _fixed_summary((s[key] for s in states), n) for key in ('validity', 'bdr', 'any_shortcut')}
    result['n_planned'] = n
    for group in ('conditions', 'shortcuts', 'labels'):
        keys = CONDITIONS if group == 'conditions' else (
            ('text_only', 'image_only', 'common_knowledge', 'question_leakage', 'independent_stitching') if group == 'shortcuts'
            else ('validity', 'relatedness', 'text_necessary', 'image_necessary', 'integration'))
        result[group] = {k: _fixed_summary((s[group][k] for s in states), n,
            n_not_applicable=sum((not r['question']['normalized_text'] and r['generation']['status'] in (
                'generation_failure','technical_failure','no_question')) or r['labels'][source][k].get('status')=='not_applicable'
                for r in rows) if group == 'labels' and k != 'validity' else 0) for k in keys}
    # Disjoint cross-tab cells retain every row, including invalid and pending.
    cross = Counter()
    for s in states:
        l = s['labels']
        row = f"V={s['validity']} R={l['relatedness']}"
        if l['text_necessary'] == 0 or l['image_necessary'] == 0:
            column = 'necessity_fail'
        elif l['text_necessary'] == l['image_necessary'] == 1:
            column = 'both_necessary_integrated' if l['integration'] == 1 else (
                'both_necessary_stitched' if l['integration'] == 0 else 'unresolved')
        else:
            column = 'unresolved'
        cross[(row, column)] += 1
    result['cross_table'] = [{'row': r, 'column': c, 'n': count, 'share': _rate(count, n)}
                             for (r, c), count in sorted(cross.items())]
    core = Counter({k:0 for k in ('joint_integrated','single_modality','knowledge_or_leakage','stitched','unresolved')})
    for state in states:
        l=state['labels']; shortcuts=state['shortcuts']
        if state['validity']!=1 or l['relatedness']!=1:
            continue
        if shortcuts['independent_stitching']==1:
            key='stitched'
        elif shortcuts['common_knowledge']==1 or shortcuts['question_leakage']==1:
            key='knowledge_or_leakage'
        elif l['text_necessary']==0 or l['image_necessary']==0:
            key='single_modality'
        elif l['text_necessary']==l['image_necessary']==l['integration']==1:
            key='joint_integrated'
        else:
            key='unresolved'
        core[key]+=1
    result['core_cross_table']={'validity':1,'relatedness':1,**dict(core),'row_total':sum(core.values()),'denominator':n}
    return result


def summarize_records(records_path, *, output_dir=None):
    """Offline entry point for python -m egvqg records-metrics --records ..."""
    from pathlib import Path
    from .records import ExperimentStore, read_jsonl, write_json
    records_path = Path(records_path); rows = read_jsonl(records_path)
    ids = [r['attempt_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate attempts in records.jsonl')
    if (records_path.parent / 'manifest.json').exists() and (records_path.parent / 'plan.jsonl').exists():
        store = ExperimentStore(records_path.parent)
        if set(ids) != store.ids:
            raise ValueError('record/plan mismatch; N must not shrink')
    versions = {r['protocol_version'] for r in rows}
    if len(versions) > 1:
        raise ValueError('cannot pool protocol versions')
    metrics = {'protocol_version': next(iter(versions), 'three-phase-prompt-v1'), 'n_planned': len(rows),
               'warning': 'Automatic results are provisional. Human and adjudicated tables never inherit automatic labels. All rates use planned N; bounds express unresolved labels, not confidence intervals.'}
    for source in ('automatic', 'human', 'adjudicated'):
        metrics[source] = compute_record_metrics(rows, source=source)
        metrics[source]['by_method'] = {m: compute_record_metrics([r for r in rows if r['method'] == m], source=source)
                                         for m in sorted({r['method'] for r in rows})}
    out = Path(output_dir) if output_dir else records_path.parent
    out.mkdir(parents=True, exist_ok=True); write_json(out / 'metrics.json', metrics)
    lines = ['# 固定分母实验汇总', '', metrics['warning'], '', f'预安排名额 N={len(rows)}。', '']
    for source in ('automatic', 'human', 'adjudicated'):
        groups = [('all', metrics[source]), *metrics[source]['by_method'].items()]
        lines += [f'## {source}', '', '| 方法 | 指标 | 通过/N | 未决 | 不适用 | 下界 | 上界 |',
                  '|---|---|---:|---:|---:|---:|---:|']
        for method, stats in groups:
            items = [(k, stats[k]) for k in ('validity', 'bdr', 'any_shortcut')]
            items += [(f'{group}.{key}', item) for group in ('conditions', 'labels', 'shortcuts') for key, item in stats[group].items()]
            for name, item in items:
                lines.append(f"| {method} | {name} | {item['n_pass']}/{stats['n_planned']} | {item['n_unresolved']} | {item['n_not_applicable']} | {item['lower']} | {item['upper']} |")
        lines += ['', '交叉表（数量之和=N）：', '', '| 行 | 列 | 数量 |', '|---|---|---:|']
        lines += [f"| {r['row']} | {r['column']} | {r['n']} |" for r in metrics[source]['cross_table']]
        lines.append('')
        c=metrics[source]['core_cross_table']
        lines += ['核心交叉表（展示桶互斥，全部比例分母仍为N）：', '',
          '| 有效性通过 | 相关性通过 | 双必要+整合 | 单模态足够 | 常识/题干捷径 | 拼接 | 未决 | 合计 |',
          '|---|---|---:|---:|---:|---:|---:|---:|',
          f"| 是 | 是 | {c['joint_integrated']} | {c['single_modality']} | {c['knowledge_or_leakage']} | {c['stitched']} | {c['unresolved']} | {c['row_total']} |",'']
    (out / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    return metrics


def outcome_patterns(rows):
    """Diagnostic decomposition, not a metric: per method, how each planned attempt
    behaved under TI/T/I/0. C=scored correct, W=scored wrong, R=refused,
    U=answered but unscored/unresolved, -=no response. Shows what sits behind
    unresolved BDR (refusals are never converted to wrong here either)."""
    def code(record, condition):
        r = record['responses'][condition]
        if not r:
            return '-'
        if r['status'] == 'refused' or r.get('abstain'):
            return 'R'
        if r.get('score_status') == 'scored' and r.get('correct') in (0, 1):
            return 'C' if r['correct'] else 'W'
        return 'U'
    result = {}
    for record in rows:
        method = result.setdefault(record['method'], {'n_planned': 0, 'patterns': Counter(), 'attempts': {}})
        pattern = '/'.join(code(record, c) for c in CONDITIONS)
        method['n_planned'] += 1; method['patterns'][pattern] += 1
        method['attempts'][record['attempt_id']] = {'material_id': record['material_id'], 'pattern': pattern,
            'generation': record['generation']['status'], 'reference': record['reference']['status']}
    for method in result.values():
        method['patterns'] = dict(sorted(method['patterns'].items(), key=lambda item: (-item[1], item[0])))
    return {'order': list(CONDITIONS), 'legend': 'C correct, W wrong, R refused, U unscored, - no response',
            'by_method': result}

