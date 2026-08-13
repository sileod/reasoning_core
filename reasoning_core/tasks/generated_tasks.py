from reasoning_core.template import Task
from reasoning_core.tasks.generated.controlled_code_execution import ControlledCodeExecutionMixin
from reasoning_core.tasks.generated.boolean_propagation_search import BooleanPropagationSearchMixin
from reasoning_core.tasks.generated.backtracking_search import BacktrackingSearchMixin
from reasoning_core.tasks.generated.dynamic_programming import DynamicProgrammingMixin
from reasoning_core.tasks.generated.fixpoint_iteration import FixpointIterationMixin
from reasoning_core.tasks.generated.variable_elimination import VariableEliminationMixin
from reasoning_core.tasks.generated.shift_reduce_parsing import ShiftReduceParsingMixin


class ControlledCodeExecution(ControlledCodeExecutionMixin, Task):
    pass


class BooleanPropagationSearch(BooleanPropagationSearchMixin, Task):
    pass


class BacktrackingSearch(BacktrackingSearchMixin, Task):
    pass


class DynamicProgramming(DynamicProgrammingMixin, Task):
    pass


class FixpointIteration(FixpointIterationMixin, Task):
    pass


class VariableElimination(VariableEliminationMixin, Task):
    pass


class ShiftReduceParsing(ShiftReduceParsingMixin, Task):
    pass
