"""Goal-directed, recursively productive Python synthesis for code reasoning.

V3 separates program generation from task labels. Risky operations are ordinary
constructors; runnability is observed by executing one program on sampled inputs.
Difficulty combines static slicing with a traced accepted call. Alpha-renaming is
an optional final pass.
"""

from ._imperative_mesopy_analysis import structural_fingerprint
from ._imperative_mesopy_expr import _ExpressionMixin
from ._imperative_mesopy_generation import _GenerationMixin
from ._imperative_mesopy_phenomena import _PhenomenaMixin
from ._imperative_mesopy_runtime import _RuntimeMixin
from ._imperative_mesopy_stmt import _StatementMixin
from ._imperative_mesopy_types import (
    BOOL, CONTROLLED_PHENOMENA, DICT_INT, INT, LIST_INT, OBSERVED_ERRORS, PHENOMENA,
    TUPLE_INT, CallOutcome, MesopyComplexity, MesopyConfig, MesopyGoal, MesopySample, MinimalPair,
)


class ImperativeMesopy(
    _GenerationMixin, _ExpressionMixin, _StatementMixin, _PhenomenaMixin, _RuntimeMixin
):
    pass


__all__ = [
    "BOOL", "CONTROLLED_PHENOMENA", "DICT_INT", "INT", "LIST_INT",
    "OBSERVED_ERRORS", "PHENOMENA", "TUPLE_INT", "CallOutcome",
    "ImperativeMesopy", "MesopyComplexity", "MesopyConfig", "MesopyGoal",
    "MesopySample", "MinimalPair", "structural_fingerprint",
]
