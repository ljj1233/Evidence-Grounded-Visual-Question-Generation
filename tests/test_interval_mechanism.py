from egvqg.interval_mechanism import choose_shift_interval, conditions_for, count_in_interval, summarize_lens


def test_count_in_interval_is_inclusive():
    assert count_in_interval([12, 7, 22, 30], 11, 20) == 1
    assert count_in_interval([11, 20], 11, 20) == 2
    assert count_in_interval([5, 40], 11, 20) == 0


def test_choose_shift_interval_changes_count_and_avoids_forbidden():
    values = [12, 7, 22, 30]
    shift = choose_shift_interval(11, 20, values, forbidden=(2,))
    assert shift is not None
    assert shift['expected'] not in (1, 2)
    assert shift['expected'] == count_in_interval(values, shift['lower'], shift['upper'])
    assert shift['expected'] > 0


def test_choose_shift_interval_prefers_small_movement():
    shift = choose_shift_interval(9, 34, [10, 5, 31, 35], forbidden=(3,))
    assert shift['expected'] == 1
    assert (shift['lower'], shift['upper']) == (11, 34)


def test_conditions_keep_the_four_isolated_materials():
    item = {'text': 'Only items with value from 11 to 20 count.', 'expected': 1, 'model_answer': 2,
            'lower': 11, 'upper': 20, 'values': [12, 7, 22, 30],
            'shift': {'lower': 7, 'upper': 22, 'expected': 3}}
    conditions = {c['condition']: c for c in conditions_for(item)}
    assert conditions['original']['text'] == item['text']
    assert conditions['text_ablate']['text'] == 'Only items with value in the stated interval count.'
    assert conditions['shifted']['expected'] == 3
    assert conditions['disjoint']['expected'] == 0
    assert conditions['image_only']['text'] == '' and conditions['image_only']['use_image']
    assert conditions['text_only']['use_image'] is False


def test_summarize_lens_reports_crossover_and_final_preference():
    rows = [
        {'condition': 'original', 'role': 'failure', 'crossover_layer': None, 'final_argmax_matches_model': True,
         'layers': [{'correct_prob': 0.1, 'wrong_prob': 0.7}, {'correct_prob': 0.4, 'wrong_prob': 0.5}]},
        {'condition': 'original', 'role': 'failure', 'crossover_layer': 1, 'final_argmax_matches_model': True,
         'layers': [{'correct_prob': 0.2, 'wrong_prob': 0.3}, {'correct_prob': 0.6, 'wrong_prob': 0.3}]},
    ]
    summary = summarize_lens(rows)['original:failure']
    assert summary['n'] == 2
    assert summary['crossover_count'] == 1
    assert summary['median_crossover_layer'] == 1
    assert summary['layers'][1]['fraction_correct_above_wrong'] == 0.5
    assert summary['layers'][0]['correct_prob'] == 0.15000000000000002


def test_expansion_delta_and_shifted_relation():
    from egvqg.interval_mechanism import expansion_delta, shifted_relation

    assert expansion_delta([12, 7, 22, 30], 11, 20, 1) == 0
    assert expansion_delta([12, 7, 22, 30], 11, 20, 2) == 2
    assert expansion_delta([12, 7, 22, 30], 11, 20, 3) == 4
    assert expansion_delta([12, 7, 22, 30], 11, 20, 5) is None
    assert shifted_relation([12, 7, 22, 30], 7, 22, 3) == 'tracks_both_bounds'
    assert shifted_relation([10, 5, 31, 35], 11, 34, 2) == 'lower_bound_only'
    assert shifted_relation([10, 5, 31, 35], 11, 34, 3) == 'upper_bound_only'
