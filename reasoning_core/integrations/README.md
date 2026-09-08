# Python integrations

`reasoning_gym.py` and `synlogic.py` adapt external generators to the core task
contract. `as_reasoning_gym(TaskClass)` adapts a core task in the other direction;
`reasoning_core.register_to_reasoning_gym(["arithmetics"])` registers selected
adapters. Core `Task` never inherits an optional dependency's base class.

Importing `reasoning_core.integrations` loads no external framework. Install the
relevant dependency before importing/using an adapter. Existing
`get_task("reasoning_gym")`, `get_task("synlogic")`, and collection import paths
remain supported.

OpenEnv and Prime Intellect are separately packaged applications under root
[`integrations/`](../../integrations/README.md), outside the core wheel.
