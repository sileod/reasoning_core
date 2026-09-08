from dataclasses import replace
from itertools import islice

import pytest

from reasoning_core.evaluation.groups import TaskGroup
from reasoning_core.evaluation.composition import GroupMeasurement, compare_composition
from reasoning_core.evaluation import intrinsic
from reasoning_core.evaluation.training.arm import ArmSpec
from reasoning_core.evaluation.training.groups import group_arm
from reasoning_core.evaluation.influence import ArmPlan, run_influence


def test_group_identity_normalizes_order_and_scale():
    assert TaskGroup(('b', 'a'), (3, 1)) == TaskGroup(('a', 'b'), (2, 6))
    assert TaskGroup(('a',)).weights == (1.0,)
    assert TaskGroup(('a', 'b')).identifier != TaskGroup(('a', 'b'), (1, 2)).identifier
    for tasks, weights in [((), ()), (('a', 'a'), ()), (('a',), (0,)), (('a',), (float('nan'),))]:
        with pytest.raises(ValueError):
            TaskGroup(tasks, weights)


def test_composition_predicts_and_checks_comparability():
    group = TaskGroup(('a', 'b'), (1, 3))
    a = GroupMeasurement(TaskGroup(('a',)), 'protocol', 0, {'nll': 2.0})
    b = GroupMeasurement(TaskGroup(('b',)), 'protocol', 0, {'nll': 4.0})
    observed = GroupMeasurement(group, 'protocol', 0, {'nll': 3.0})
    result = compare_composition(observed, [a, b])
    assert result['predicted'] == {'nll': 3.5}
    assert result['residual'] == {'nll': -0.5}
    for components in ([a], [a, a], [a, replace(b, protocol_id='other')],
                       [a, replace(b, seed=1)]):
        with pytest.raises(ValueError):
            compare_composition(observed, components)


def test_group_intrinsic_keeps_members_and_evaluation_weights(monkeypatch):
    calls = []
    def reward(model, tokenizer, rows, spec, max_length):
        calls.append(rows)
        return {'reward': rows[0]['score'], 'reward_examples': min(spec.n_eval, len(rows))}
    monkeypatch.setattr(intrinsic, 'free_gen_reward', reward)
    rows = {'a': [{'score': 1}] * 20, 'b': [{'score': 0}] * 20}
    result = intrinsic.group_reward(None, None, rows, TaskGroup(('a','b')),
                                    intrinsic.FreeGenRewardSpec(n_eval=3), 64)
    assert len(calls) == 2
    assert result == {'reward/a': 1, 'reward/b': 0, 'reward_examples/a': 3,
                      'reward_examples/b': 3, 'reward': 0.5}
    monkeypatch.setattr(intrinsic, 'free_gen_reward', lambda *a: {'reward': None})
    with pytest.raises(RuntimeError, match='No scorable'):
        intrinsic.group_reward(None, None, rows, TaskGroup(('a','b')),
                               intrinsic.FreeGenRewardSpec(), 64)


class Tokenizer:
    eos_token = ' eos'
    def __call__(self, text):
        return {'input_ids': text.split()}


def spec():
    return ArmSpec('group-test', 'a+b', initialization_id='initial', main_data_id='pending',
                   max_length=32,
                   formatter='influence_legacy_v1')


def test_group_sampling_is_replayable_and_corrects_token_lengths():
    group = TaskGroup(('a', 'b'))
    main = [{'prompt': 'main ', 'completion': 'answer eos'}]
    aux = {'a': [{'prompt': 'a', 'answer': 'answer'}],
           'b': [{'prompt': 'b long long long', 'answer': 'answer'}]}
    plan = group_arm(spec(), group, main, aux, Tokenizer(), aux_token_fraction=0.5)
    samples = list(islice(plan.dataset(), 10000))
    assert samples[:10] == list(islice(plan.dataset(), 10))
    totals = dict.fromkeys(('main', 'a', 'b'), 0)
    for row in samples:
        totals[row['prompt'].split()[0]] += len((row['prompt'] + row['completion']).split())
    total = sum(totals.values())
    assert totals['main'] / total == pytest.approx(0.5, abs=0.03)
    assert totals['a'] / total == pytest.approx(0.25, abs=0.03)
    assert totals['b'] / total == pytest.approx(0.25, abs=0.03)
    assert plan.spec.aux_tasks == group.tasks
    assert plan.spec.aux_weights == group.weights
    changed = group_arm(spec(), TaskGroup(('a', 'b'), (1,3)), main, aux, Tokenizer(), aux_token_fraction=0.5)
    assert changed.spec.spec_id != plan.spec.spec_id
    with pytest.raises(ValueError, match='fit max_length'):
        group_arm(replace(spec(), max_length=2), group, main, aux, Tokenizer(), aux_token_fraction=0.5)


def test_group_plan_retains_initial_and_final_intrinsic_rewards(monkeypatch):
    from reasoning_core.evaluation import influence
    from reasoning_core.evaluation.training import groups
    monkeypatch.setattr(groups, 'group_reward', lambda model, *args: {
        'reward': model.weight, 'reward/a': model.weight, 'reward_examples/a': 3})
    main = [{'prompt': 'main ', 'completion': 'answer eos'}]
    rows = {'a': [{'prompt': 'a', 'answer': 'answer'}]}
    plan = group_arm(spec(), TaskGroup(('a',)), main, rows, Tokenizer(), aux_token_fraction=0.5,
                     reward_rows=rows, reward_spec=intrinsic.FreeGenRewardSpec(n_eval=3))
    class Model:
        def load_state_dict(self, state): self.weight = state['weight']
    def run(model, tokenizer, dataset, spec, evaluate=None, **kwargs):
        model.weight = 1
        return None, {'nll': 2.0, **(evaluate(model) if evaluate else {})}
    monkeypatch.setattr(influence, 'run_arm', run)
    baseline = ArmPlan(replace(plan.spec, arm_id='base'), lambda: [])
    result = run_influence(Model(), None, {'weight': 0}, baseline, (plan,), metric_names=('nll',))
    assert result.treatments['a+b']['initial/reward/a'] == 0
    assert result.treatments['a+b']['reward/a'] == 1
    assert result.treatments['a+b']['reward_examples/a'] == 3


def test_training_groups_can_share_a_larger_intrinsic_evaluation_group():
    main = [{'prompt': 'main ', 'completion': 'answer eos'}]
    rows = {t: [{'prompt': t, 'answer': 'answer'}] for t in ('a', 'b')}
    evaluation_group = TaskGroup(('a', 'b'))
    plans = [group_arm(spec(), TaskGroup((t,)), main, {t: rows[t]}, Tokenizer(),
                       aux_token_fraction=0.5, reward_rows=rows,
                       reward_spec=intrinsic.FreeGenRewardSpec(),
                       evaluation_group=evaluation_group) for t in ('a', 'b')]
    assert plans[0].spec.eval_ids == plans[1].spec.eval_ids
    assert plans[0].spec.aux_data_id != plans[1].spec.aux_data_id
