import importlib
import subprocess
import sys

import pytest


def test_core_does_not_import_or_inherit_reasoning_gym():
    code = '''
import sys
sys.modules['reasoning_gym'] = None
from reasoning_core.template import Task
assert Task.__bases__ == (object,)
assert 'reasoning_gym.dataset' not in sys.modules
'''
    subprocess.run([sys.executable, '-c', code], check=True, timeout=30)


@pytest.mark.parametrize('old,new', [
    ('training.arm', 'evaluation.training.arm'),
    ('training.evals', 'evaluation.metrics'),
    ('training.intrinsic_rewards', 'evaluation.intrinsic'),
    ('training.battery', 'evaluation.battery'),
    ('training.influence', 'evaluation.influence'),
    ('generation_worker', 'generation.worker'),
    ('collect', 'generation.collect'),
    ('zero_shot_eval', 'evaluation.zero_shot'),
])
def test_legacy_imports_share_module_identity(old, new):
    assert importlib.import_module('reasoning_core.' + old) is importlib.import_module('reasoning_core.' + new)


def test_reasoning_gym_registration_uses_adapter(monkeypatch):
    gym = pytest.importorskip('reasoning_gym')
    from reasoning_core import register_to_reasoning_gym
    from reasoning_core.template import Task
    monkeypatch.setattr(gym.factory, 'DATASETS', dict(gym.factory.DATASETS))
    gym.factory.DATASETS.pop('arithmetics', None)
    register_to_reasoning_gym(['arithmetics'])
    dataset = gym.create_dataset('arithmetics', size=2, seed=1)
    assert not isinstance(dataset, Task)
    assert len(dataset) == 2
    rows = list(dataset)
    assert len(rows) == 2
    assert all(dataset.score_answer(row['answer'], row) == 1 for row in rows)
    register_to_reasoning_gym(['arithmetics'])  # idempotent
    with pytest.raises(IndexError):
        dataset[2]
