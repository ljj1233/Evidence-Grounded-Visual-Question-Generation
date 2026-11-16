import json

from egvqg.studies.self_verification import (_plan, _sampling_curve, _selected, condition_material, parse_gold,
                                    score_gold)


def _derived(**overrides):
    material = {
        'category': 'derived',
        'material_id': 'syn-derived-000',
        'values': [3, 8, 6, 2],
        'threshold': 5,
        'text': 'Only items with value at least 5 count.',
        'image_path': 'data/processed/synthetic-composition-v1/images/syn-derived-000.png',
        'evaluation': {
            'full': {'answer_kind': 'numeric', 'expected': 14,
                     'question': 'What is the total value of the items that count?'},
            'image_hop': {'answer_kind': 'item_values', 'expected': [3, 8, 6, 2],
                          'question': 'What are the values of items A, B, C, and D?'},
            'text_hop': {'answer_kind': 'numeric', 'expected': 5,
                         'question': 'What is the minimum value an item must have to count?'},
        },
    }
    material.update(overrides)
    return material


def _bridging():
    return {
        'category': 'bridging',
        'material_id': 'syn-bridging-000',
        'values': [3, 6, 4],
        'text': 'The value in the right column is called alpha.',
        'image_path': 'data/processed/synthetic-composition-v1/images/syn-bridging-000.png',
        'evaluation': {
            'full': {'answer_kind': 'numeric', 'expected': 4, 'question': 'What is the value of alpha?'},
            'image_hop': {'answer_kind': 'numeric', 'expected': 4, 'question': 'What is the value in the right column?'},
            'text_hop': {'answer_kind': 'column', 'expected': 'right',
                         'question': 'Which column holds the value called alpha?'},
        },
    }


def test_parse_gold_matches_designed_questions_first():
    material = _derived()
    gold = parse_gold('What is the total value of the items that count?', material)
    assert gold == {'kind': 'designed', 'combine': 'sum', 'answer_kind': 'numeric', 'expected': 14,
                    'source': 'full'}
    gold = parse_gold('  what are the values of items a, b, c, and d? ', material)
    assert gold['source'] == 'image_hop' and gold['expected'] == [3, 8, 6, 2]
    alpha = parse_gold('What is the value of alpha?', _bridging())
    assert alpha['source'] == 'full' and alpha['expected'] == 4


def test_parse_gold_pattern_rules_use_the_question_threshold():
    material = _derived()
    which = parse_gold('Which items have a value of at least 8?', material)
    assert which['kind'] == 'item_set' and which['expected'] == ['B']
    assert parse_gold('Which items count?', material)['expected'] == ['B', 'C']
    assert parse_gold('How many items are there at least 5?', material)['expected'] == 2
    assert parse_gold('What is the value of item C?', material)['expected'] == 6
    assert parse_gold('What is the threshold?', material)['expected'] == 5
    assert parse_gold('Why is the sky blue?', material) is None


def test_score_gold_handles_numeric_and_item_sets():
    numeric = {'kind': 'designed', 'answer_kind': 'numeric', 'expected': 14}
    assert score_gold(json.dumps({'answer': '14', 'abstain': False}), numeric)[0] is True
    assert score_gold(json.dumps({'answer': '12', 'abstain': False}), numeric)[0] is False
    assert score_gold(json.dumps({'answer': None, 'abstain': True}), numeric)[0] is False
    items = {'kind': 'item_set', 'expected': ['B', 'C']}
    assert score_gold(json.dumps({'answer': 'C, B', 'abstain': False}), items)[0] is True
    assert score_gold(json.dumps({'answer': 'B', 'abstain': False}), items)[0] is False
    assert score_gold(json.dumps({'answer': None, 'abstain': True}), items)[0] is False


def test_condition_material_hides_the_removed_side(tmp_path):
    material = _derived()
    text, image = condition_material(material, 'TI', tmp_path)
    assert text == material['text'] and image == str(tmp_path / material['image_path'])
    text, image = condition_material(material, 'T', tmp_path)
    assert text == material['text'] and image is None
    text, image = condition_material(material, 'I', tmp_path)
    assert text == '' and image == str(tmp_path / material['image_path'])


def test_selection_rule_rejects_single_side_answers():
    def rows(ti, t, i):
        return [{'condition': 'TI', 'correct': ti}, {'condition': 'T', 'correct': t}, {'condition': 'I', 'correct': i}]

    assert _selected(rows(True, False, False)) is True
    assert _selected(rows(True, True, False)) is False
    assert _selected(rows(True, False, True)) is False
    assert _selected(rows(False, False, False)) is False
    assert _selected([{'condition': 'TI', 'correct': True}]) is None


def test_early_stop_skipped_rows_are_decided_rejections():
    def group(ti, t, i):
        return [{'condition': 'TI', 'correct': ti}, {'condition': 'T', 'correct': t},
                {'condition': 'I', 'correct': i}]

    skipped = [{'condition': 'TI', 'correct': False},
               {'condition': 'T', 'correct': None, 'skipped': 'ti_wrong'},
               {'condition': 'I', 'correct': None, 'skipped': 'ti_wrong'}]
    # a wrong full-material answer is a decided rejection, not an undetermined case
    assert _selected(skipped) is False
    assert _selected(group(True, False, False)) is True
    # a side that was never answered and never skipped keeps the group undetermined
    assert _selected(group(True, None, False)) is None


def test_plan_drops_uncovered_questions_and_reports_coverage():
    pool = [
        {'material_id': 'syn-derived-000', 'question': 'What is the total value of the items that count?',
         'category': 'derived', 'generated_label': 'dual_derived'},
        {'material_id': 'syn-derived-000', 'question': 'Why is the sky blue?',
         'category': 'derived', 'generated_label': 'other'},
        {'material_id': 'missing', 'question': 'What is the total value of the items that count?',
         'category': 'derived', 'generated_label': 'dual_derived'},
    ]
    plan, dropped = _plan(pool, [_derived()], 'run', ('direct',), ('TI', 'T', 'I'))
    assert len(plan) == 3 and {slot['condition'] for slot in plan} == {'TI', 'T', 'I'}
    assert len({slot['attempt_id'] for slot in plan}) == 3
    assert [row['reason'] for row in dropped] == ['no_gold', 'missing_material']


def test_selection_metrics_and_external_mapping(tmp_path):
    from egvqg.studies.self_verification import _external_selections, _selection_metrics

    universe = [
        {'material_id': 'm1', 'question': 'q1', 'label': 'dual_bridging', 'dual': True},
        {'material_id': 'm1', 'question': 'q2', 'label': 'single_image', 'dual': False},
        {'material_id': 'm2', 'question': 'q3', 'label': 'dual_bridging', 'dual': True},
    ]
    local = [{**item, 'selected': item['dual']} for item in universe]
    metrics = _selection_metrics(local)
    assert (metrics['recall']['k'], metrics['recall']['n']) == (2, 2)
    assert (metrics['precision']['k'], metrics['precision']['n']) == (2, 2)
    assert metrics['by_label']['single_image'] == {'k': 0, 'n': 1}
    path = tmp_path / 'selection.jsonl'
    path.write_text('\n'.join(json.dumps(row) for row in [
        {'material_id': 'm1', 'question': 'q1', 'sol_final_label': 'dual_candidate'},
        {'material_id': 'm1', 'question': 'q2', 'sol_final_label': 'dual_candidate'},
    ]) + '\n')
    external, unmatched = _external_selections(path, universe)
    assert unmatched == 1
    assert [row['selected'] for row in external] == [True, True]
    assert _selection_metrics(external)['precision']['k'] == 1


def test_per_item_and_compare_tiers():
    from egvqg.studies.self_verification import parse_gold, score_gold_compare, score_gold_per_item

    material = _derived(values=[11, 2, 17, 7], threshold=6,
                        evaluation={'full': {'answer_kind': 'numeric', 'expected': 35,
                                             'question': 'What is the total value of the items that count?'}})
    sum_gold = parse_gold('What is the total value of the items that count?', material)
    assert sum_gold['combine'] == 'sum' and sum_gold['expected'] == 35
    items = '[{"item":"A","value":11,"counts":true},{"item":"B","value":2,"counts":false},' \
            '{"item":"C","value":17,"counts":true},{"item":"D","value":7,"counts":true}]'
    ok, parsed, reason, detail = score_gold_per_item('{"items":' + items + ',"answer":"35"}', sum_gold)
    assert ok and parsed == 35 and detail['values'] == [11, 17, 7] and detail['consistent'] is True
    # the count field is what the model says, so an item wrongly marked as counting is a real error
    bad = items.replace('"B","value":2,"counts":false', '"B","value":2,"counts":true')
    assert score_gold_per_item('{"items":' + bad + '}', sum_gold)[0] is False

    compare = '{"rule":"Only items with value at least 6 count.","threshold":6,"items":[' \
              '{"item":"A","value":11},{"item":"B","value":2},{"item":"C","value":17},{"item":"D","value":7}]}'
    ok, parsed, _reason, detail = score_gold_compare(compare, sum_gold)
    assert ok and parsed == 35 and detail['threshold'] == 6 and detail['operator'] == '>='
    strict = compare.replace('at least 6', 'more than 6')
    ok_strict, parsed_strict, _reason, strict_detail = score_gold_compare(strict, sum_gold)
    assert strict_detail['operator'] == '>' and ok_strict and parsed_strict == 35
    wrong_threshold = compare.replace('at least 6', 'at least 10')
    assert score_gold_compare(wrong_threshold, sum_gold)[0] is False
    set_gold = parse_gold('Which items count?', material)
    assert score_gold_compare(compare, set_gold)[1] == ['A', 'C', 'D']


def test_full_material_error_taxonomy():
    from egvqg.studies.self_verification import _ti_error_taxonomy

    material = {'material_id': 'm', 'values': [11, 5, 17, 7], 'threshold': 6}
    rows = [
        {'material_id': 'm', 'correct': False, 'parsed_answer': 35 + 5,
         'gold': {'expected': 35}},                       # included the threshold-minus-one value
        {'material_id': 'm', 'correct': False, 'parsed_answer': 28,
         'gold': {'expected': 35}},                       # dropped a qualifying value
        {'material_id': 'm', 'correct': True, 'parsed_answer': 35, 'gold': {'expected': 35}},
    ]
    taxonomy = _ti_error_taxonomy(rows, {'m': material})
    assert taxonomy == {'dropped_qualifying': 1, 'included_threshold_minus_one': 1}


def test_prepare_real_pool_samples_evenly(tmp_path):
    from egvqg.studies.self_verification import prepare_real_pool

    candidates = tmp_path / 'candidates.jsonl'
    rows = []
    for material_id in ('m1', 'm2'):
        for index in range(4):
            rows.append({'material_id': material_id, 'question': f'{material_id}-q{index}',
                         'label_consensus': {'verdict': 'dual_necessary' if index == 1 else 'text_suff'}})
    candidates.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    materials = tmp_path / 'materials.jsonl'
    materials.write_text('\n'.join(json.dumps({'material_id': material_id, 'text': 't', 'image_path': 'i.png'})
                                   for material_id in ('m1', 'm2')) + '\n')
    pool, loaded = prepare_real_pool(candidates_path='candidates.jsonl', materials_path='materials.jsonl',
                                     sample_per_material=2, project_root=str(tmp_path))
    assert len(pool) == 4 and set(loaded) == {'m1', 'm2'}
    assert [row['position'] for row in pool if row['material_id'] == 'm1'] == [0, 3]
    assert sum(row['dual'] for row in pool) == 0


def test_sampling_curve_grows_with_more_samples():
    def item(material, index, dual, selected):
        return {'material_id': material, 'sample_index': index, 'dual': dual, 'selected': selected}

    selections = [item('m1', 0, False, False), item('m1', 1, True, True),
                  item('m2', 0, False, False), item('m2', 1, False, False)]
    curve = _sampling_curve(selections)
    assert [point['k'] for point in curve] == [1, 2]
    assert curve[0]['selected'] == 0 and curve[0]['materials_with_dual'] == 0
    assert curve[1]['selected'] == 1 and curve[1]['selected_dual'] == 1
    assert curve[1]['precision'] == 1.0
    assert curve[1]['materials_with_selected_dual'] == 1
    # a pool without sample indices has no sampling curve
    assert _sampling_curve([{'material_id': 'm1', 'sample_index': None, 'dual': True, 'selected': True}]) is None


def test_answerability_audit_scores_text_screen(tmp_path):
    from egvqg.studies import self_verification as sv
    directory = tmp_path / 'outputs' / 'audit'
    directory.mkdir(parents=True)
    pool = [{'material_id': 'm1', 'question': 'dual q', 'label': 'dual_necessary', 'dual': True, 'position': 0},
            {'material_id': 'm2', 'question': 'text q', 'label': 'text_suff', 'dual': False, 'position': 0}]
    answers = []
    for row, text_answer in ((pool[0], None), (pool[1], 'answered')):
        for condition in ('TI', 'T'):
            value = text_answer if condition == 'T' else 'answered'
            answers.append({'material_id': row['material_id'], 'question': row['question'],
                            'condition': condition,
                            'raw_output': json.dumps({'answer': value, 'abstain': value is None})})
    (directory / 'pool.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in pool))
    (directory / 'answers.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in answers))
    summary = sv.report_answerability_audit(run_id='audit', bootstrap=200, project_root=str(tmp_path))
    rule = summary['rules']['reject_if_text_answers']
    assert (rule['selected'], rule['recall']['k'], rule['precision']['k']) == (1, 1, 1)
    assert rule['text_answered_rate']['n'] == 2 and rule['text_answered_rate']['k'] == 1
    assert (directory / 'audit_report.md').exists()
    # every rule is present even when only two conditions were answered
    assert set(summary['rules']) >= {'reject_if_image_answers', 'reject_if_either_side_answers'}
    assert summary['rules']['reject_if_either_side_answers']['selected'] == 1


def test_answerability_audit_prior_condition_rejects_blind_answers(tmp_path):
    from egvqg.studies import self_verification as sv
    directory = tmp_path / 'outputs' / 'audit_prior'
    directory.mkdir(parents=True)
    pool = [{'material_id': 'm1', 'question': 'real dual', 'label': 'dual_necessary', 'dual': True, 'position': 0},
            {'material_id': 'm1', 'question': 'prior q', 'label': 'dual_necessary', 'dual': True, 'position': 1}]
    blind = {('m1', 'real dual'): False, ('m1', 'prior q'): True}
    answers = []
    for row in pool:
        key = (row['material_id'], row['question'])
        for condition in ('TI', 'T', 'I', '0'):
            # no side answers, but the blind condition answers the prior-knowledge question
            flag = blind[key] if condition == '0' else condition == 'TI' and blind[key]
            answers.append({'material_id': row['material_id'], 'question': row['question'],
                            'condition': condition,
                            'raw_output': json.dumps({'answer': '11' if flag else None,
                                                      'abstain': not flag})})
    (directory / 'pool.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in pool))
    (directory / 'answers.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in answers))
    summary = sv.report_answerability_audit(run_id='audit_prior', bootstrap=200, project_root=str(tmp_path))
    prior_rule = summary['rules']['reject_if_prior_answers']
    both = summary['rules']['reject_if_either_side_or_prior_answers']
    # both questions survive the single-side test, but the blind condition exposes the prior-knowledge one
    assert summary['rules']['reject_if_either_side_answers']['selected'] == 2
    assert (prior_rule['selected'], prior_rule['recall']['k']) == (1, 1)
    assert (both['selected'], both['precision']['k']) == (1, 1)
    assert summary['rules']['reject_if_prior_answers']['prior_answered_rate']['k'] == 1
    assert summary['evidence_ladder']['0']['on_duals']['n'] == 2
    assert summary['prior_known'] is True


def test_answerability_audit_two_sided_rule_and_api_comparison(tmp_path):
    from egvqg.studies import self_verification as sv
    directory = tmp_path / 'outputs' / 'audit2'
    directory.mkdir(parents=True)
    pool = [{'material_id': 'm1', 'question': 'dual q', 'label': 'dual_necessary', 'dual': True, 'position': 0},
            {'material_id': 'm1', 'question': 'image q', 'label': 'text_suff', 'dual': False, 'position': 1},
            {'material_id': 'm2', 'question': 'text q', 'label': 'image_suff', 'dual': False, 'position': 0}]
    answered = {('m1', 'dual q'): {'TI': True, 'T': False, 'I': False},
                ('m1', 'image q'): {'TI': True, 'T': False, 'I': True},
                ('m2', 'text q'): {'TI': True, 'T': True, 'I': False}}
    answers = []
    for row in pool:
        flags = answered[(row['material_id'], row['question'])]
        for condition, flag in flags.items():
            answers.append({'material_id': row['material_id'], 'question': row['question'], 'condition': condition,
                            'raw_output': json.dumps({'answer': 'x' if flag else None, 'abstain': not flag})})
    (directory / 'pool.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in pool))
    (directory / 'answers.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in answers))
    selection = tmp_path / 'selection.jsonl'
    selection.write_text(''.join(json.dumps({'material_id': r['material_id'], 'question': r['question'],
                                             'sol_final_label': label}) + '\n'
                                 for r, label in zip(pool, ['dual_candidate', 'text_suff', 'text_suff'])))
    summary = sv.report_answerability_audit(run_id='audit2', bootstrap=200,
                                            pvs_selection=str(selection), project_root=str(tmp_path))
    text_rule = summary['rules']['reject_if_text_answers']
    image_rule = summary['rules']['reject_if_image_answers']
    both = summary['rules']['reject_if_either_side_answers']
    # text-only: 'dual q' (T fails) and 'image q' (T fails) are kept, 'text q' is rejected
    assert text_rule['selected'] == 2
    # image-only: 'dual q' and 'text q' kept, 'image q' rejected
    assert image_rule['selected'] == 2
    # two-sided keeps only the question neither side answers
    assert (both['selected'], both['recall']['k'], both['precision']['k']) == (1, 1, 1)
    assert both['image_answered_rate']['k'] == 1
    comparison = summary['external']['reject_if_either_side_answers_vs_api']
    assert set(comparison) == {'precision_diff', 'recall_diff'}


def test_answer_evidence_levels_and_sample_rules(tmp_path):
    from egvqg.studies import self_verification as sv
    # stability needs agreement across samples; a single lucky answer is not stable
    stable = sv._answer_evidence(['{"answer": "Melzer"}', '{"answer": "Melzer"}', '{"answer": "Melzer"}'])
    noisy = sv._answer_evidence(['{"answer": "Melzer"}', '{"answer": "co-writer"}', '{"answer": "Melzer"}'])
    quiet = sv._answer_evidence(['{"answer": null, "abstain": true}'] * 2 + ['{"answer": "Melzer"}'])
    assert stable['stable'] is True and stable['share'] == 1.0
    assert noisy['stable'] is False and noisy['clusters'] == 2
    assert quiet['answered'] is True and quiet['answered_n'] == 1  # one answer out of three samples
    assert quiet['stable'] is False and quiet['share'] == round(1 / 3, 4)
    # the agreement test tolerates surface variation but not a different number
    assert sv._answers_match('melzer', 'she was formerly melzer')
    assert not sv._answers_match('35', '53')

    directory = tmp_path / 'outputs' / 'audit3'
    directory.mkdir(parents=True)
    pool = [{'material_id': 'm1', 'question': 'dual q', 'label': 'dual_necessary', 'dual': True, 'position': 0},
            {'material_id': 'm2', 'question': 'noisy q', 'label': 'text_suff', 'dual': False, 'position': 0}]
    answers = []
    for row, values in ((pool[0], {'TI': 'answer', 'T': None, 'I': None}),
                        (pool[1], {'TI': 'Philippa', 'T': ['Melzer', 'co-writer', 'Boyens'], 'I': 'Philippa'})):
        for condition, value in values.items():
            samples = value if isinstance(value, list) else [value] * 3
            for index, sample in enumerate(samples):
                answers.append({'material_id': row['material_id'], 'question': row['question'],
                                'condition': condition, 'sample_index': index,
                                'raw_output': json.dumps({'answer': sample, 'abstain': sample is None})})
    (directory / 'pool.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in pool))
    (directory / 'answers.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in answers))
    summary = sv.report_answerability_audit(run_id='audit3', bootstrap=200, project_root=str(tmp_path))
    assert summary['samples'] == 3
    assert summary['evidence_ladder']['T']['stable']['k'] == 0
    rules = summary['rules']
    # 'noisy q' happens to answer once; existence rejects it, stability does not
    assert rules['reject_if_text_answers']['selected'] == 1
    assert rules['reject_if_text_stable']['selected'] == 2
    assert rules['reject_if_text_agrees_full']['selected'] == 2
    assert rules['reject_if_either_side_answers']['selected'] == 1
