import signal
import time
from pathlib import Path

from transformers import TrainerCallback


COMPLETE_MARKER = ".complete"
FORMAT_MARKER = ".resumable-checkpoints-v1"


def prepare_checkpoint_dir(output_dir):
    """Adopt complete pre-marker Trainer checkpoints once, then require markers."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    format_marker = output_dir / FORMAT_MARKER
    if not format_marker.exists():
        for checkpoint in output_dir.glob("checkpoint-*"):
            has_model = any((checkpoint / name).exists() for name in (
                "model.safetensors", "pytorch_model.bin", "adapter_model.safetensors",
            ))
            required = ("trainer_state.json", "optimizer.pt", "scheduler.pt")
            if has_model and all((checkpoint / name).exists() for name in required):
                (checkpoint / COMPLETE_MARKER).touch()
        format_marker.touch()
    return output_dir


def latest_complete_checkpoint(output_dir):
    checkpoints = []
    for path in Path(output_dir).glob("checkpoint-*"):
        try:
            step = int(path.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        if (path / COMPLETE_MARKER).exists():
            checkpoints.append((step, path))
    return str(max(checkpoints)[1]) if checkpoints else None


class ResumableCheckpointCallback(TrainerCallback):
    """Request saves by elapsed time and mark only fully written checkpoints resumable."""

    def __init__(self, every_minutes=60, save_final=True,
                 stop_signals=(signal.SIGTERM, signal.SIGUSR1)):
        self.interval = max(0.0, float(every_minutes) * 60)
        self.save_final = save_final
        self.deadline = None
        self.signal_requested = False
        self.stop_after_save = False
        self.interrupted = False
        self._previous_handlers = {}
        self.stop_signals = stop_signals

    def on_train_begin(self, args, state, control, **kwargs):
        prepare_checkpoint_dir(args.output_dir)
        self.deadline = time.monotonic() + self.interval if self.interval else None
        for sig in self.stop_signals:
            self._previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._request_stop)

    def on_step_end(self, args, state, control, **kwargs):
        now = time.monotonic()
        due = self.signal_requested or (self.deadline is not None and now >= self.deadline)
        if due:
            control.should_save = True
            self.stop_after_save = self.signal_requested
            self.signal_requested = False
            if self.interval:
                self.deadline = now + self.interval
        elif state.global_step >= state.max_steps and not self.save_final:
            control.should_save = False
        return control

    def on_save(self, args, state, control, **kwargs):
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        (checkpoint / COMPLETE_MARKER).touch()
        if self.stop_after_save:
            self.interrupted = True
            control.should_training_stop = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        for sig, handler in self._previous_handlers.items():
            signal.signal(sig, handler)

    def _request_stop(self, signum, frame):
        self.signal_requested = True
