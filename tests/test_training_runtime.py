from types import SimpleNamespace

import pytest

from reasoning_core.training.checkpointing import (
    COMPLETE_MARKER,
    ResumableCheckpointCallback,
    latest_complete_checkpoint,
    prepare_checkpoint_dir,
)
from reasoning_core.training.paths import HOME, home_path
from reasoning_core.training.arm import ArmSpec, optimizer_eval_mode
from reasoning_core.training.data import format_row
from reasoning_core.training.data import (
    StreamSpec, fraction_for_token_share, load_stream, mix_streams, ratio_to_fraction,
    replay_after, steps_for_token_budget,
)
from reasoning_core.training.evals import (
    EvalExample, eval_id, evaluate_mcq, evaluate_qa_nll, load_eval_suite, load_qa_jsonl,
)
from reasoning_core.training.influence import ArmPlan, run_influence


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


def test_arm_identity_changes_with_behavioral_inputs():
    base = ArmSpec("experiment", "baseline", initialization_id="init-a")
    same = ArmSpec("experiment", "baseline", initialization_id="init-a")
    changed_init = ArmSpec("experiment", "baseline", initialization_id="init-b")
    changed_eval = ArmSpec("experiment", "baseline", initialization_id="init-a",
                           eval_ids=("logic/v2",))
    assert base.spec_id == same.spec_id
    assert base.run_dir == same.run_dir
    assert len({base.spec_id, changed_init.spec_id, changed_eval.spec_id}) == 3


def test_paired_influence_uses_one_arm_runner_and_resets_weights(monkeypatch):
    seen = []

    class Model:
        def load_state_dict(self, state):
            seen.append(("reset", state["weight"]))

    def fake_run_arm(model, tokenizer, dataset, spec, evaluate=None):
        seen.append((spec.arm_id, dataset))
        return None, {"nll": 2.0 if spec.arm_id == "baseline" else 1.5}

    monkeypatch.setattr("reasoning_core.training.influence.run_arm", fake_run_arm)
    result = run_influence(
        Model(), None, {"weight": 7},
        ArmPlan(ArmSpec("exp", "baseline"), lambda: "main"),
        (ArmPlan(ArmSpec("exp", "treatment"), lambda: "mixed"),),
    )
    assert seen == [
        ("reset", 7), ("baseline", "main"),
        ("reset", 7), ("treatment", "mixed"),
    ]
    assert result.deltas == {"treatment": {"nll": -0.5}}
