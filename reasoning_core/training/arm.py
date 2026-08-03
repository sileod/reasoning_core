"""One resumable, content-addressed training arm."""

import hashlib
import json
import os
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace

from trl import SFTConfig, SFTTrainer

from reasoning_core import __version__
from reasoning_core.training.checkpointing import (
    ResumableCheckpointCallback,
    latest_complete_checkpoint,
    prepare_checkpoint_dir,
)
from reasoning_core.training.local_metrics import LocalMetricsCallback, LocalMetricsSink
from reasoning_core.training.optimizers import (
    create_optimizer_and_scheduler,
    trainer_cls_for_optimizer,
)
from reasoning_core.training.paths import RUNS_HOME, home_path


ENGINE_VERSION = 3


@dataclass(frozen=True)
class ArmSpec:
    """Everything that can change the result of one training arm.

    Data and initialization content hashes are supplied by the caller because a
    generic trainer cannot infer them reliably from an iterable or model object.
    """

    experiment_id: str
    arm_id: str
    model: str = ""
    model_revision: str | None = None
    optimizer: str = "prodigy"
    learning_rate: float = 1.0
    weight_decay: float = 0.01
    lr_scheduler_type: str = "constant"
    max_steps: int = 10
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_length: int = 128
    checkpoint_every_minutes: float = 60
    adamc_weight_decay: float = 20.0
    adamc_r: float = 0.0
    seed: int = 0
    save_final: bool = False
    gradient_checkpointing: bool = False
    formatter: str = "sft_qa_v1"
    aux_formatter: str | None = None
    prompt_prefix: str = ""
    aux_prompt_prefix: str = ""
    main_source: str = "synthetic"
    main_config: str | None = None
    main_revision: str | None = None
    aux_source: str | None = None
    aux_config: str | None = None
    aux_revision: str | None = None
    aux_task: str | None = None
    aux_fraction: float = 0.0
    target_aux_token_fraction: float | None = None
    shuffle_buffer: int = 0
    length_margin: int = 0
    packing: bool = True
    bf16: bool = False
    initialization_id: str = ""
    main_data_id: str = ""
    aux_data_id: str = ""
    eval_ids: tuple[str, ...] = ()
    callback_ids: tuple[str, ...] = ()

    def __post_init__(self):
        missing = [name for name in ("initialization_id", "main_data_id")
                   if not getattr(self, name)]
        uses_aux = (
            self.aux_source is not None or self.aux_fraction > 0
            or self.target_aux_token_fraction is not None
        )
        if uses_aux and not self.aux_data_id:
            missing.append("aux_data_id")
        if missing:
            raise ValueError(f"ArmSpec requires immutable {', '.join(missing)}")

    @property
    def spec_id(self):
        payload = {
            "engine_version": ENGINE_VERSION,
            "package_version": __version__,
            "dependencies": _dependency_versions(),
            "spec": asdict(self),
        }
        return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()[:16]

    @property
    def run_dir(self):
        name = f"{safe_name(self.arm_id)}-{self.spec_id}"
        return home_path(RUNS_HOME / "arms" / safe_name(self.experiment_id) / name)


def run_arm(model, tokenizer, dataset, spec, eval_dataset=None, callbacks=(), evaluate=None):
    """Train or resume one arm and return ``(trainer, metrics)``.

    A completed arm returns ``(None, metrics)``. The content-addressed directory
    makes changed specs new runs; the status check also refuses corrupted or
    manually moved state before considering a checkpoint resumable.
    """

    if len(callbacks) != len(spec.callback_ids):
        raise ValueError("Every external callback requires a matching ArmSpec.callback_ids entry")
    if evaluate is not None and not spec.eval_ids:
        raise ValueError("External evaluation requires a versioned ArmSpec.eval_ids entry")

    run_dir = spec.run_dir
    prepare_checkpoint_dir(run_dir)
    status_path = run_dir / "status.json"
    status = _read_json(status_path)
    serialized_spec = json.loads(_canonical_json(asdict(spec)))
    if status and status.get("spec") != serialized_spec:
        raise RuntimeError(f"Arm state has a different spec in {status_path}")
    if status and status.get("state") == "complete":
        return None, status["metrics"]

    provenance = _provenance(spec)
    _write_json(status_path, {
        "state": "running", "spec": serialized_spec, "provenance": provenance,
    })
    sink = LocalMetricsSink(
        run_dir / "metrics.jsonl",
        run_hash=f"{spec.experiment_id}/{spec.arm_id}/{spec.spec_id}",
        group_id=spec.experiment_id,
        script_args=SimpleNamespace(**serialized_spec, stage_name=spec.arm_id),
        eff_batch=spec.batch_size * spec.gradient_accumulation_steps,
    )
    schedule_free = spec.optimizer in {"prodigy", "adamc"}
    trainer_cls, trainer_kwargs = SFTTrainer, {}
    if schedule_free:
        optimizer_payload = {**serialized_spec, "decay": spec.weight_decay,
                             "train_source_loss": False}
        optimizer_args = SimpleNamespace(**optimizer_payload)
        optimizer, scheduler = create_optimizer_and_scheduler(model, optimizer_args)
        trainer_cls = schedule_free_trainer(trainer_cls_for_optimizer(optimizer_args))
        trainer_kwargs["optimizers"] = (optimizer, scheduler)
    checkpoint = ResumableCheckpointCallback(
        spec.checkpoint_every_minutes, save_final=spec.save_final,
    )
    trainer = trainer_cls(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        callbacks=[LocalMetricsCallback(sink), checkpoint, *callbacks],
        args=SFTConfig(
            output_dir=str(run_dir),
            max_steps=spec.max_steps,
            per_device_train_batch_size=spec.batch_size,
            gradient_accumulation_steps=spec.gradient_accumulation_steps,
            learning_rate=spec.learning_rate,
            weight_decay=spec.weight_decay,
            lr_scheduler_type=spec.lr_scheduler_type,
            optim=spec.optimizer if not schedule_free else "adamw_torch",
            max_grad_norm=0.0 if schedule_free else 1.0,
            max_length=spec.max_length,
            completion_only_loss=True,
            packing=spec.packing,
            bf16=spec.bf16,
            gradient_checkpointing=spec.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False}
            if spec.gradient_checkpointing else None,
            report_to="none",
            logging_steps=1,
            save_strategy="steps",
            save_steps=10**12,
            save_total_limit=1,
            seed=spec.seed,
            disable_tqdm=True,
        ),
        **trainer_kwargs,
    )
    result = trainer.train(resume_from_checkpoint=latest_complete_checkpoint(run_dir))
    if checkpoint.interrupted:
        _write_json(status_path, {
            "state": "interrupted", "spec": serialized_spec, "provenance": provenance,
        })
        return trainer, None
    metrics = {"train_loss": result.training_loss, "global_step": trainer.state.global_step}
    if eval_dataset is not None:
        metrics.update(trainer.evaluate())
    if evaluate is not None:
        metrics.update(_evaluate_model(model, evaluate, trainer.optimizer, schedule_free))
    _write_json(status_path, {
        "state": "complete", "spec": serialized_spec,
        "provenance": provenance, "metrics": metrics,
    })
    return trainer, metrics


# Compatibility with the prototype name. New code should use run_arm.
train_arm = run_arm


def record_event(spec, kind, metrics):
    path = spec.run_dir.parent / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "experiment_id": spec.experiment_id, "arm_id": spec.arm_id,
        "spec_id": spec.spec_id, "engine_version": ENGINE_VERSION, "kind": kind,
        "optimizer": spec.optimizer, "formatter": spec.formatter,
        "aux_formatter": spec.aux_formatter, "aux_fraction": spec.aux_fraction,
        **metrics,
    }
    encoded = json.dumps(row, sort_keys=True)
    if path.exists() and encoded in path.read_text().splitlines():
        return
    with path.open("a") as file:
        file.write(encoded + "\n")


def schedule_free_trainer(base):
    class ArmTrainer(ScheduleFreeTrainerMixin, base):
        pass
    return ArmTrainer


class ScheduleFreeTrainerMixin:
    def evaluate(self, *args, **kwargs):
        with optimizer_eval_mode(self.optimizer):
            return super().evaluate(*args, **kwargs)

    def _save_checkpoint(self, *args, **kwargs):
        with optimizer_eval_mode(self.optimizer):
            return super()._save_checkpoint(*args, **kwargs)


@contextmanager
def optimizer_eval_mode(optimizer):
    inner = optimizer
    while hasattr(inner, "optimizer"):
        inner = inner.optimizer
    was_training = any(group.get("train_mode", True) for group in inner.param_groups)
    optimizer.eval()
    try:
        yield
    finally:
        (optimizer.train if was_training else optimizer.eval)()


def _evaluate_model(model, evaluate, optimizer, schedule_free):
    context = optimizer_eval_mode(optimizer) if schedule_free else nullcontext()
    with context:
        return evaluate(model)


def safe_name(value):
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _provenance(spec):
    return {
        "engine": "reasoning_core.training.arm",
        "engine_version": ENGINE_VERSION,
        "package_version": __version__,
        "spec_id": spec.spec_id,
        "initialization_id": spec.initialization_id,
        "main_data_id": spec.main_data_id,
        "aux_data_id": spec.aux_data_id,
        "eval_ids": list(spec.eval_ids),
        "callback_ids": list(spec.callback_ids),
        "dependencies": _dependency_versions(),
    }


def _dependency_versions():
    dependencies = {}
    for package in ("torch", "transformers", "trl", "datasets"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            pass
    return dependencies


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
