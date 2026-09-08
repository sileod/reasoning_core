import random
import ast

from reasoning_core.tasks.generated.wave10.counterfactual_replay.counterfactual_replay import (
    CounterfactualReplay,
    CounterfactualReplayConfig,
    _replay,
    _expr,
)


def test_generate_example_level_default():
    random.seed(1)
    task = CounterfactualReplay()
    ex = task.generate_example()
    assert ex.answer is not None
    assert ex.answer.lstrip("-").isdigit() or ex.answer == "0" or ex.answer.lstrip("-").isdigit()


def test_round_trip_scores_one():
    random.seed(2)
    task = CounterfactualReplay()
    for _ in range(20):
        ex = task.generate_entry()
        prompt = task.render_prompt(ex.metadata)
        assert task.score_answer(ex.answer, ex) == 1.0
        assert prompt.startswith("Consider these quantities")
        # gold answer must differ from the original target value: counterfactual changed the outcome
        assert int(ex.answer) != int(ex.metadata["c_orig"]) or ex.metadata["changed"] != ex.metadata["target"]


def test_garbage_scores_zero():
    random.seed(3)
    task = CounterfactualReplay()
    ex = task.generate_entry()
    assert task.score_answer("not a number", ex) == 0.0
    assert task.score_answer("", ex) == 0.0
    assert task.score_answer("abc", ex) == 0.0


def test_wrong_answer_scores_zero():
    random.seed(4)
    task = CounterfactualReplay()
    for _ in range(20):
        ex = task.generate_entry()
        wrong = int(ex.answer) + 1
        assert task.score_answer(str(wrong), ex) == 0.0


def test_difficulty_changes_config():
    cfg = CounterfactualReplayConfig()
    l0 = (cfg.n_layers, cfg.n_per_layer, cfg.w_range, cfg.off_range)
    task = CounterfactualReplay()
    cfg.set_level(5)
    l5 = (cfg.n_layers, cfg.n_per_layer, cfg.w_range, cfg.off_range)
    assert l5 != l0
    assert cfg.n_layers >= l0[0]


def test_all_levels_generate():
    task = CounterfactualReplay()
    for level in range(7):
        cfg = CounterfactualReplayConfig()
        cfg.set_level(level)
        t = CounterfactualReplay(config=cfg)
        for _ in range(5):
            ex = t.generate_entry()
            assert t.score_answer(ex.answer, ex) == 1.0


def test_replay_retains_independent():
    # Check that replay keeps non-descendants (independent events) at original values
    # while recomputing the changed source's descendants.
    layers = [[5, 7], [99]]
    parents = [[([0, 1], [2, 3], 0)]]  # layer1 node0 = 2*c + 3*indep + 0
    desc = [{0}, {0}]                  # node0 of each layer is a descendant of c
    rep = _replay(layers, parents, desc, 0, 9)
    assert rep[0] == [9, 7]            # c changed to 9, independent node 7 retained
    assert rep[1] == [2 * 9 + 3 * 7]   # target recomputed = 39


def test_metadata_json_serializable():
    import json
    random.seed(6)
    task = CounterfactualReplay()
    ex = task.generate_entry()
    json.dumps(ex.metadata)
