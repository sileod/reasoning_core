import torch
from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree
from transformers import get_constant_schedule
from trl import SFTTrainer
from reasoning_core.training.source_signals import SourceLossMixin


def add_optimizer_args(parser):
    parser.add_argument(
        "--optimizer",
        type=str,
        default="prodigy",
        choices=["prodigy", "adamc"],
        help="Optimizer to use for SFT.",
    )
    parser.add_argument(
        "--adamc_weight_decay",
        type=float,
        default=20.0,
        help="AdamC ScheduleFree+ decay; this is not on the same scale as AdamW/Prodigy decay.",
    )
    parser.add_argument(
        "--adamc_r",
        type=float,
        default=0.0,
        help="Schedule-Free averaging power for AdamC ScheduleFree+.",
    )
    parser.add_argument("--train_source_loss", type=_str_to_bool, nargs="?", const=True, default=False)

def create_optimizer_and_scheduler(model, args):
    if args.optimizer == "prodigy":
        optimizer = ProdigyPlusScheduleFree(
            model.parameters(),
            lr=1.0,
            weight_decay=args.decay,
            use_bias_correction=False,
            betas=(0.95, 0.99),
        )
    elif args.optimizer == "adamc":
        optimizer_cls = _loss_aware_adamc_schedulefree_plus_paper_cls()
        optimizer = optimizer_cls(
            model.parameters(),
            lr=1.0,
            weight_decay=args.adamc_weight_decay,
            betas=(0.9, 0.95),
            sf_beta1=0.9,
            r=args.adamc_r,
            polyak_beta=0,
            c_warmup=0,
            sf_beta1_anneal_steps=0,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    return optimizer, get_constant_schedule(optimizer)


def trainer_cls_for_optimizer(args):
    if args.optimizer == "adamc":
        return SourceLossAwareSFTTrainer if args.train_source_loss else LossAwareSFTTrainer
    if args.train_source_loss:
        return SourceSFTTrainer
    return SFTTrainer


def _str_to_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).lower() in {"1", "true", "yes", "y", "on"}


class SourceSFTTrainer(SourceLossMixin, SFTTrainer):
    pass


class LossAwareSFTTrainer(SFTTrainer):
    """SFTTrainer variant for optimizers whose step needs the current loss."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._losses_for_optimizer_step = []

    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)
        loss_for_step = loss.detach().float()
        grad_accum = getattr(self, "current_gradient_accumulation_steps", None)
        if grad_accum:
            loss_for_step = loss_for_step * grad_accum
        self._losses_for_optimizer_step.append(loss_for_step)

        if self.accelerator.sync_gradients:
            stacked = torch.stack(
                [x.to(loss_for_step.device) for x in self._losses_for_optimizer_step]
            )
            _set_loss_for_step(self.optimizer, stacked.mean())
            self._losses_for_optimizer_step = []

        return loss


class SourceLossAwareSFTTrainer(SourceLossMixin, LossAwareSFTTrainer):
    pass


def _loss_aware_adamc_schedulefree_plus_paper_cls():
    try:
        from schedulefree.adamc_schedulefree_plus_paper import AdamCScheduleFreePlusPaper
    except ImportError as exc:
        raise ImportError(
            "The AdamC ScheduleFree+ paper optimizer needs the upstream "
            "`schedulefree` package containing `adamc_schedulefree_plus_paper.py`. "
            "Install it from https://github.com/facebookresearch/schedule_free "
            "before using `--optimizer adamc`."
        ) from exc

    class LossAwareAdamCScheduleFreePlusPaper(AdamCScheduleFreePlusPaper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._loss_for_step = None

        def set_loss_for_step(self, loss):
            self._loss_for_step = float(loss.detach().float().cpu())

        @torch.no_grad()
        def step(self, closure=None):
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
                self.set_loss_for_step(loss)

            if self._loss_for_step is None:
                raise RuntimeError("Missing loss for AdamC ScheduleFree+ Polyak optimizer step.")

            function_value = self._loss_for_step
            self._loss_for_step = None
            return self.step_func(function_value=function_value)

    return LossAwareAdamCScheduleFreePlusPaper


def _unwrap_optimizer(optimizer):
    while hasattr(optimizer, "optimizer"):
        optimizer = optimizer.optimizer
    return optimizer


def _set_loss_for_step(optimizer, loss):
    optimizer = _unwrap_optimizer(optimizer)
    if hasattr(optimizer, "set_loss_for_step"):
        optimizer.set_loss_for_step(loss)
