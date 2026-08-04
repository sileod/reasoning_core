import json
from types import SimpleNamespace

import pytest

from reasoning_core.training.battery import (
    EvalBattery, EvalLeg, evaluate_battery, load_battery_manifest, paper_battery,
)
from reasoning_core.training.checkpointing import (
    COMPLETE_MARKER,
    ResumableCheckpointCallback,
    latest_complete_checkpoint,
    prepare_checkpoint_dir,
)
from reasoning_core.training.paths import HOME, home_path
from reasoning_core.training.arm import ArmSpec, _evaluate_model, optimizer_eval_mode
from reasoning_core.training.data import format_row
from reasoning_core.training.data import (
    StreamSpec, content_id, fraction_for_token_share, load_stream, mix_streams,
    ratio_to_fraction, replay_after, source_id, steps_for_token_budget,
)
from reasoning_core.training.evals import (
    EvalExample, eval_id, evaluate_lm_nll, evaluate_mcq, evaluate_qa_nll, load_eval_suite,
    load_qa_jsonl,
)
from reasoning_core.training.influence import ArmPlan, run_influence
from reasoning_core.training.intrinsic_rewards import (
    FreeGenRewardSpec, free_gen_reward, reward_id,
)


def test_write_paths_must_stay_under_home(tmp_path):
    assert home_path(HOME / "runs") == HOME / "runs"
    outside = tmp_path if HOME not in tmp_path.parents else HOME.parent / "outside"
    with pytest.raises(ValueError):
        home_path(outside)


def test_only_complete_checkpoint_is_resumable(tmp_path):
    (tmp_path / "checkpoint-20").mkdir()
    complete = tmp_path / "checkpoint-10"
    complete.mkdir()
    (complete / COMPLETE_MARKER).touch()
    assert latest_complete_checkpoint(tmp_path) == str(complete)


def test_existing_trainer_checkpoint_is_adopted(tmp_path):
    checkpoint = tmp_path / "checkpoint-3"
    checkpoint.mkdir()
    for name in ("model.safetensors", "trainer_state.json", "optimizer.pt", "scheduler.pt"):
        (checkpoint / name).touch()
    prepare_checkpoint_dir(tmp_path)
    assert latest_complete_checkpoint(tmp_path) == str(checkpoint)


def test_wall_clock_checkpoint_marks_completed_save(tmp_path, monkeypatch):
    clock = iter((0.0, 61.0))
    monkeypatch.setattr("reasoning_core.training.checkpointing.time.monotonic", lambda: next(clock))
    callback = ResumableCheckpointCallback(every_minutes=1, stop_signals=())
    args = SimpleNamespace(output_dir=str(tmp_path))
    state = SimpleNamespace(global_step=7)
    control = SimpleNamespace(should_save=False, should_training_stop=False)

    callback.on_train_begin(args, state, control)
    callback.on_step_end(args, state, control)
    assert control.should_save
    (tmp_path / "checkpoint-7").mkdir()
    callback.on_save(args, state, control)
    assert (tmp_path / "checkpoint-7" / COMPLETE_MARKER).exists()


def test_signal_save_stops_without_implying_completion(tmp_path):
    callback = ResumableCheckpointCallback(every_minutes=0, stop_signals=())
    args = SimpleNamespace(output_dir=str(tmp_path))
    state = SimpleNamespace(global_step=9)
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    callback.on_train_begin(args, state, control)
    callback._request_stop(None, None)
    callback.on_step_end(args, state, control)
    (tmp_path / "checkpoint-9").mkdir()
    callback.on_save(args, state, control)
    assert callback.interrupted
    assert control.should_training_stop


def test_short_arm_can_skip_forced_final_checkpoint(tmp_path):
    callback = ResumableCheckpointCallback(every_minutes=60, save_final=False, stop_signals=())
    args = SimpleNamespace(output_dir=str(tmp_path))
    state = SimpleNamespace(global_step=1, max_steps=1)
    control = SimpleNamespace(should_save=True, should_training_stop=True)
    callback.on_train_begin(args, state, control)
    callback.on_step_end(args, state, control)
    assert not control.should_save


def test_versioned_formatters_preserve_both_contracts():
    row = {"prompt": "1 + 1?", "answer": "2"}
    assert format_row(row, "</s>", "sft_qa_v1") == {
        "prompt": "Q: 1 + 1?\nA:",
        "completion": " 2</s>",
    }
    assert format_row(row, "</s>", "influence_legacy_v1") == {
        "prompt": "1 + 1?\n",
        "completion": "2</s>",
    }
    assert format_row(row, "</s>", "influence_auto_v1") == {
        "prompt": "1 + 1?\n",
        "completion": "2</s>",
    }
    assert format_row({"text": "x" * 1300}, "</s>", "influence_auto_v1") == {
        "prompt": "",
        "completion": "x" * 1200 + "</s>",
    }
    assert format_row(row, "</s>", "sft_qa_v1", "<SPECIAL>\n") == {
        "prompt": "<SPECIAL>\nQ: 1 + 1?\nA:",
        "completion": " 2</s>",
    }


def test_optimizer_eval_mode_restores_prior_mode():
    calls = []
    inner = SimpleNamespace(param_groups=[{"train_mode": True}])
    optimizer = SimpleNamespace(
        optimizer=inner,
        eval=lambda: calls.append("eval"),
        train=lambda: calls.append("train"),
    )
    with optimizer_eval_mode(optimizer):
        calls.append("body")
    assert calls == ["eval", "body", "train"]


def test_external_evaluation_uses_schedule_free_weights():
    calls = []
    inner = SimpleNamespace(param_groups=[{"train_mode": True}])
    optimizer = SimpleNamespace(
        optimizer=inner,
        eval=lambda: calls.append("eval"),
        train=lambda: calls.append("train"),
    )
    result = _evaluate_model(None, lambda model: calls.append("score") or {"nll": 1.0},
                             optimizer, schedule_free=True)
    assert result == {"nll": 1.0}
    assert calls == ["eval", "score", "train"]


def test_local_and_mixed_streams_replay_exactly(tmp_path):
    main_path, aux_path = tmp_path / "main.jsonl", tmp_path / "aux.jsonl"
    main_path.write_text("".join(
        f'{{"prompt":"m{i}","answer":"{i}"}}\n' for i in range(40)
    ))
    aux_path.write_text("".join(
        f'{{"prompt":"a{i}","answer":"{i}"}}\n' for i in range(40)
    ))
    tokenizer = SimpleNamespace(eos_token="</s>")

    def factory():
        main = load_stream(StreamSpec(str(main_path), "sft_qa_v1"), tokenizer)
        aux = load_stream(StreamSpec(str(aux_path), "sft_qa_v1", cycle=True), tokenizer)
        return mix_streams(main, aux, aux_fraction=0.25, seed=42, shuffle_buffer=10)

    stream = factory()
    iterator = iter(stream)
    consumed = [next(iterator) for _ in range(17)]
    expected = [next(iterator) for _ in range(10)]
    resumed = list(replay_after(factory, len(consumed)).take(10))
    assert resumed == expected
    assert {row["_source"] for row in consumed} == {str(main_path), str(aux_path)}


def test_token_budget_matches_run_sft_iterable_formula():
    assert steps_for_token_budget(1_000, 0.2, 100, 2) == 6
    assert ratio_to_fraction(0.2) == pytest.approx(1 / 6)
    assert fraction_for_token_share(0.2, 100, 200) == pytest.approx(1 / 9)
    assert fraction_for_token_share(0.2, 100, 100) == pytest.approx(0.2)


def test_influence_mix_is_an_absolute_fraction(monkeypatch):
    captured = {}

    def fake_interleave(parts, probabilities, **kwargs):
        captured["probabilities"] = probabilities
        return parts[0]

    monkeypatch.setattr("reasoning_core.training.data.interleave_datasets", fake_interleave)
    stream = SimpleNamespace(shuffle=lambda **kwargs: None)
    mix_streams(stream, stream, aux_fraction=0.2, shuffle_buffer=0)
    assert captured["probabilities"] == [0.8, 0.2]


def test_exact_token_filter_rejects_overlong_aux(tmp_path):
    path = tmp_path / "aux.jsonl"
    path.write_text(
        '{"prompt":"short","answer":"ok"}\n'
        '{"prompt":"one two three four","answer":"too long"}\n'
    )

    class Tokenizer:
        eos_token = "<eos>"

        def __call__(self, text, add_special_tokens):
            ids = text.replace("<eos>", " <eos>").split()
            return {"input_ids": ([0] if add_special_tokens else []) + list(range(len(ids)))}

    rows = list(load_stream(
        StreamSpec(str(path), "influence_legacy_v1"),
        Tokenizer(), max_length=100, max_tokens=4,
    ))
    assert [row["_source_index"] for row in rows] == [0]


def test_local_content_id_changes_with_file_and_directory_contents(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"value":1}\n')
    first = content_id(path)
    path.write_text('{"value":2}\n')
    assert content_id(path) != first
    directory = tmp_path / "parts"
    directory.mkdir()
    (directory / "part.jsonl").write_text("one\n")
    first = content_id(directory)
    (directory / "part.jsonl").write_text("two\n")
    assert content_id(directory) != first


def test_source_id_binds_remote_sources_to_exact_commits(tmp_path):
    local = tmp_path / "data.jsonl"
    local.write_text("row\n")
    assert source_id(local) == content_id(local)
    with pytest.raises(ValueError, match="does not match"):
        source_id(local, supplied="sha256:stale")
    revision = "a" * 40
    assert source_id("org/data", revision=revision) == f"hf:org/data@{revision}"
    with pytest.raises(ValueError, match="40-character"):
        source_id("org/data", revision="main")


def test_remote_stream_passes_its_pinned_revision(monkeypatch):
    import reasoning_core.training.data as data_module

    captured = {}
    monkeypatch.setattr(
        data_module, "load_dataset",
        lambda *args, **kwargs: captured.update(args=args, kwargs=kwargs) or "stream",
    )
    with pytest.raises(ValueError, match="40-character"):
        data_module._raw_stream(StreamSpec("org/data", "text_v1"))
    revision = "b" * 40
    assert data_module._raw_stream(
        StreamSpec("org/data", "text_v1", revision=revision)
    ) == "stream"
    assert captured["kwargs"]["revision"] == revision


def test_local_stream_can_filter_task_column(tmp_path):
    path = tmp_path / "aux.jsonl"
    path.write_text(
        '{"task":"logic","prompt":"p","answer":"a"}\n'
        '{"task":"math","prompt":"q","answer":"b"}\n'
    )
    tokenizer = SimpleNamespace(eos_token="</s>")
    rows = list(load_stream(
        StreamSpec(str(path), "influence_legacy_v1", task="logic"), tokenizer,
    ))
    assert [row["prompt"] for row in rows] == ["p\n"]


def test_local_stream_can_filter_mode_and_level(tmp_path):
    path = tmp_path / "aux.jsonl"
    path.write_text(
        '{"task":"logic","mode":"instruct","level":1,"prompt":"keep","answer":"a"}\n'
        '{"task":"logic","mode":"verify","level":1,"prompt":"mode","answer":"b"}\n'
        '{"task":"logic","mode":"instruct","level":3,"prompt":"level","answer":"c"}\n'
    )
    tokenizer = SimpleNamespace(eos_token="</s>")
    rows = list(load_stream(
        StreamSpec(str(path), "influence_legacy_v1", task="logic",
                   mode="instruct", max_level=2),
        tokenizer,
    ))
    assert [row["prompt"] for row in rows] == ["keep\n"]


def test_mixed_local_stream_dispatches_by_non_null_content(tmp_path):
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        '{"prompt":"question","answer":"answer"}\n'
        '{"text":"document"}\n'
    )
    tokenizer = SimpleNamespace(eos_token="</s>")
    rows = list(load_stream(
        StreamSpec(str(path), "influence_auto_v1"), tokenizer,
    ))
    assert [(row["prompt"], row["completion"]) for row in rows] == [
        ("question\n", "answer</s>"), ("", "document</s>"),
    ]


def test_frozen_qa_eval_contract_and_content_id(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        '{"prompt":"Question?","answer":"Answer"}\n'
        '{"prompt":"","answer":"ignored"}\n'
    )
    assert load_qa_jsonl(path, "</s>") == [
        ("Question?\n", "Answer</s>"), ("", "ignored</s>"),
    ]
    suite = load_eval_suite(path, "</s>", name="empty_prompt")
    assert suite.examples[1].prompt == ""
    assert suite.identifier.startswith("empty_prompt/eval@v1:")
    assert eval_id("logic", path).startswith("logic/answer_nll@v1:")
    assert eval_id("logic", path, 1) != eval_id("logic", path, 2)


def test_paper_battery_is_ordered_data_not_engine_logic(tmp_path):
    battery = paper_battery(tmp_path)
    assert [leg.name for leg in battery.legs] == [
        "dolci", "fw", "flan", "mbpp", "mmlu_math_cloze", "mmlu_logic_cloze",
        "drop", "gsm8k", "logiqa", "arc_easy", "arc_challenge", "blimp",
        "folio", "mmlu_other_cloze", "mmlu_math_macro", "bbh_dev",
        "bbh_dev_cloze", "bbh_test", "bbh_test_cloze", "bbh_open",
        "bbh_test_open",
    ]
    assert [leg.kind for leg in battery.legs].count("mcq") == 11
    legs = {leg.name: leg for leg in battery.legs}
    assert legs["bbh_dev"].path == legs["bbh_dev_cloze"].path
    assert legs["bbh_test"].path == legs["bbh_test_cloze"].path
    assert legs["bbh_dev_cloze"].accuracy_key == "bbh_dev_mc_cloze_acc"
    assert legs["bbh_test_cloze"].accuracy_key == "bbh_test_mc_cloze_acc"

    manifest = tmp_path / "custom.json"
    manifest.write_text(json.dumps({
        "name": "custom", "max_length": 99,
        "legs": [{
            "name": "new_leg", "path": "new.jsonl", "kind": "qa_nll",
            "output_key": "new_nll", "limit": 7,
        }],
    }))
    custom = load_battery_manifest(manifest)
    assert custom.max_length == 99
    assert custom.legs == (
        EvalLeg("new_leg", str(tmp_path / "new.jsonl"), "qa_nll", "new_nll", 7),
    )


def test_battery_rejects_metric_key_collisions():
    with pytest.raises(ValueError, match="metric keys"):
        EvalBattery("bad", (
            EvalLeg("first", "a.jsonl", "qa_nll", "same"),
            EvalLeg("second", "b.jsonl", "qa_nll", "same"),
        ))


def test_battery_mcq_pairs_nll_and_accuracy_in_one_result(tmp_path):
    path = tmp_path / "mcq.jsonl"
    path.write_text(
        '{"prompt":"question","answer":"short","choices":["short","two tokens"],'
        '"answer_idx":0}\n'
    )

    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            return SimpleNamespace(input_ids=list(range(len(text.split()))))

    class Model:
        training = True

        def parameters(self):
            yield __import__("torch").zeros(1)

        def eval(self):
            self.training = False

        def train(self, mode=True):
            self.training = mode

        def __call__(self, input_ids, labels):
            return SimpleNamespace(loss=SimpleNamespace(item=lambda: float(input_ids.shape[1])))

    result = evaluate_battery(
        Model(), Tokenizer(),
        EvalBattery("pair", (EvalLeg(
            "logic", str(path), "mcq", "logic_nll",
            accuracy_key="logic_acc", margin_key="logic_margin",
        ),), 8),
        "</s>",
    )
    assert result.metrics == {
        "logic_nll": 2.0,
        "logic_acc": 1.0,
        "logic_margin": 1.0,
    }
    assert result.legs["logic"]["scored_examples"] == 1


def test_qa_nll_matches_production_weighting_and_restores_mode():
    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            return SimpleNamespace(input_ids=list(range(len(text.split()))))

    class Model:
        training = True

        def parameters(self):
            yield __import__("torch").zeros(1)

        def eval(self):
            self.training = False

        def train(self, mode=True):
            self.training = mode

        def __call__(self, input_ids, labels):
            return SimpleNamespace(loss=SimpleNamespace(item=lambda: float(input_ids.shape[1])))

    model = Model()
    result = evaluate_qa_nll(
        model, Tokenizer(), [("one ", "two three"), ("one two ", "three")], max_length=8,
    )
    assert result["nll"] == pytest.approx((3 * 2 + 3 * 1) / 3)
    assert result["tokens"] == 3
    assert model.training


def test_qa_nll_rejects_an_all_skipped_suite():
    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            return SimpleNamespace(input_ids=list(range(len(text.split()))))

    class Model:
        training = True

        def parameters(self):
            yield __import__("torch").zeros(1)

        def eval(self):
            self.training = False

        def train(self, mode=True):
            self.training = mode

    with pytest.raises(RuntimeError, match="no answer tokens"):
        evaluate_qa_nll(
            Model(), Tokenizer(), [("too many prompt tokens", "and answer tokens")],
            max_length=1,
        )
    with pytest.raises(RuntimeError, match="no tokens"):
        evaluate_lm_nll(Model(), Tokenizer(), ["one"], max_length=1)


def test_mcq_choice_scoring_and_margin_contract():
    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            return SimpleNamespace(input_ids=list(range(len(text.split()))))

    class Model:
        training = True

        def parameters(self):
            yield __import__("torch").zeros(1)

        def eval(self):
            self.training = False

        def train(self, mode=True):
            self.training = mode

        def __call__(self, input_ids, labels):
            return SimpleNamespace(loss=SimpleNamespace(item=lambda: float(input_ids.shape[1])))

    model = Model()
    result = evaluate_mcq(
        model, Tokenizer(),
        [EvalExample("prompt", "short", ("short", "two tokens"), 0)],
        max_length=8,
    )
    assert result["accuracy"] == 1
    assert result["gold_nll"] == 2
    assert result["margin"] == 1
    assert model.training


def test_mcq_overlong_gold_matches_legacy_accuracy_denominator():
    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            return SimpleNamespace(input_ids=list(range(len(text.split()))))

    class Model:
        training = True

        def parameters(self):
            yield __import__("torch").zeros(1)

        def eval(self):
            self.training = False

        def train(self, mode=True):
            self.training = mode

        def __call__(self, input_ids, labels):
            return SimpleNamespace(loss=SimpleNamespace(item=lambda: float(input_ids.shape[1])))

    result = evaluate_mcq(
        Model(), Tokenizer(),
        [EvalExample("prompt", "gold is long", ("gold is long", "short"), 0)],
        max_length=3,
    )
    assert result["accuracy"] == 0
    assert result["scored_examples"] == 1
    assert result["gold_nll"] is None
    assert result["margin"] is None


def test_arm_identity_changes_with_behavioral_inputs():
    base = ArmSpec("experiment", "baseline", initialization_id="init-a", main_data_id="main-a")
    same = ArmSpec("experiment", "baseline", initialization_id="init-a", main_data_id="main-a")
    changed_init = ArmSpec("experiment", "baseline", initialization_id="init-b",
                           main_data_id="main-a")
    changed_eval = ArmSpec("experiment", "baseline", initialization_id="init-a",
                           main_data_id="main-a",
                           eval_ids=("logic/v2",))
    assert base.spec_id == same.spec_id
    assert base.run_dir == same.run_dir
    assert len({base.spec_id, changed_init.spec_id, changed_eval.spec_id}) == 3


def test_arm_requires_immutable_input_ids():
    with pytest.raises(ValueError, match="initialization_id, main_data_id"):
        ArmSpec("experiment", "baseline")
    with pytest.raises(ValueError, match="aux_data_id"):
        ArmSpec("experiment", "treatment", initialization_id="init", main_data_id="main",
                aux_source="aux.jsonl")
    with pytest.raises(ValueError, match="aux_data_id"):
        ArmSpec("experiment", "treatment", initialization_id="init", main_data_id="main",
                aux_fraction=0.2)


def test_paired_influence_uses_one_arm_runner_and_resets_weights(monkeypatch):
    seen = []

    class Model:
        def load_state_dict(self, state):
            seen.append(("reset", state["weight"]))

    def fake_run_arm(model, tokenizer, dataset, spec, evaluate=None):
        seen.append((spec.arm_id, dataset))
        return None, {
            "nll": 2.0 if spec.arm_id == "baseline" else 1.5,
            "global_step": 10,
            "eval_runtime": 1.0 if spec.arm_id == "baseline" else 2.0,
        }

    monkeypatch.setattr("reasoning_core.training.influence.run_arm", fake_run_arm)
    result = run_influence(
        Model(), None, {"weight": 7},
        ArmPlan(ArmSpec("exp", "baseline", initialization_id="init", main_data_id="main"),
                lambda: "main"),
        (ArmPlan(ArmSpec("exp", "treatment", initialization_id="init", main_data_id="main"),
                 lambda: "mixed"),),
        metric_names=("nll",),
    )
    assert seen == [
        ("reset", 7), ("baseline", "main"),
        ("reset", 7), ("treatment", "mixed"),
    ]
    assert result.deltas == {"treatment": {"nll": -0.5}}
    assert result.initial == {}


def test_influence_evaluates_shared_initial_and_each_arm_endpoint(monkeypatch):
    class Model:
        weight = 0

        def load_state_dict(self, state):
            self.weight = state["weight"]

    def fake_run_arm(model, tokenizer, dataset, spec, evaluate=None):
        model.weight += dataset
        return None, {"reward": evaluate(model)["reward"]}

    monkeypatch.setattr("reasoning_core.training.influence.run_arm", fake_run_arm)
    spec = lambda arm: ArmSpec(
        "exp", arm, initialization_id="init", main_data_id="main",
        eval_ids=("reward/v1",),
    )
    result = run_influence(
        Model(), None, {"weight": 0}, ArmPlan(spec("base"), lambda: 1),
        (ArmPlan(spec("treatment"), lambda: 2),), metric_names=("reward",),
        evaluate_endpoints=lambda model: {"reward": model.weight},
    )
    assert result.initial == {"reward": 0}
    assert result.baseline["reward"] == 1
    assert result.treatments["treatment"]["reward"] == 2


def test_influence_can_evaluate_reward_only_on_treatment(monkeypatch):
    calls = []

    class Model:
        weight = 0

        def load_state_dict(self, state):
            self.weight = state["weight"]

    def fake_run_arm(model, tokenizer, dataset, spec, evaluate=None):
        model.weight += dataset
        metrics = {"nll": float(model.weight)}
        if evaluate:
            metrics.update(evaluate(model))
        calls.append((spec.arm_id, tuple(metrics)))
        return None, metrics

    monkeypatch.setattr("reasoning_core.training.influence.run_arm", fake_run_arm)
    base = ArmSpec("exp", "base", initialization_id="init", main_data_id="main",
                   eval_ids=("battery",))
    treatment = ArmSpec("exp", "task", initialization_id="init", main_data_id="main",
                        aux_fraction=0.2, aux_data_id="aux", eval_ids=("battery", "reward"))
    reward = lambda model: {"reward": float(model.weight)}
    result = run_influence(
        Model(), None, {"weight": 0}, ArmPlan(base, lambda: 1),
        (ArmPlan(treatment, lambda: 2, evaluate_endpoint=reward),),
        metric_names=("nll",), evaluate_initial=reward,
    )
    assert calls == [("base", ("nll",)), ("task", ("nll", "reward"))]
    assert result.initial == {"reward": 0.0}


def test_free_gen_reward_is_configured_and_content_addressed(monkeypatch):
    import torch

    class Tokenizer:
        eos_token = "<eos>"
        eos_token_id = 0

        def __call__(self, text, add_special_tokens=False):
            words = text.replace("<eos>", " <eos>").split()
            return SimpleNamespace(input_ids=list(range(1, len(words) + 1)))

        def decode(self, tokens, skip_special_tokens=True):
            return "equivalent"

    class Model:
        training = True

        def parameters(self):
            yield torch.zeros(1)

        def eval(self):
            self.training = False

        def train(self, mode=True):
            self.training = mode

        def generate(self, inputs, **kwargs):
            return torch.cat((inputs, torch.tensor([[1]], device=inputs.device)), dim=1)

    rows = [
        {"prompt": f"p{i}", "answer": "correct", "mode": "instruct", "task": "x"}
        for i in range(5)
    ] + [{"prompt": "ignored", "answer": "wrong", "mode": "verify", "task": "x"}]
    monkeypatch.setattr(
        "reasoning_core.training.intrinsic_rewards.score_answer",
        lambda prediction, entry: 0.5 if entry.prompt.startswith("p") else 0.0,
    )
    spec = FreeGenRewardSpec(mode="instruct", n_eval=3, max_tokens=9)
    result = free_gen_reward(Model(), Tokenizer(), rows, spec, max_length=20)
    assert result == {"reward": 0.5, "reward_examples": 3}
    assert reward_id(rows, spec, 20) != reward_id(rows[:-1], spec, 20)
    row_object = SimpleNamespace(to_dict=lambda: rows[0])
    assert reward_id([row_object], spec, 20) == reward_id([rows[0]], spec, 20)


def test_paired_influence_rejects_duplicate_arm_ids():
    spec = ArmSpec("exp", "same", initialization_id="init", main_data_id="main")
    with pytest.raises(ValueError, match="unique"):
        run_influence(
            SimpleNamespace(), None, {}, ArmPlan(spec, lambda: "main"),
            (ArmPlan(spec, lambda: "mixed"),), metric_names=("nll",),
        )


def test_paired_influence_clones_a_shallow_torch_state_dict(monkeypatch):
    import torch
    import reasoning_core.training.influence as influence_module

    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    starts = []

    def fake_run_arm(model, tokenizer, dataset, spec, evaluate=None):
        starts.append(model.weight.item())
        with torch.no_grad():
            model.weight.add_(1)
        return None, {"nll": model.weight.item()}

    monkeypatch.setattr(influence_module, "run_arm", fake_run_arm)
    base = ArmSpec("exp", "base", initialization_id="init", main_data_id="main")
    treatment = ArmSpec("exp", "treatment", initialization_id="init", main_data_id="main")
    run_influence(
        model, None, model.state_dict(), ArmPlan(base, lambda: None),
        (ArmPlan(treatment, lambda: None),), metric_names=("nll",),
    )
    assert starts == [0, 0]
