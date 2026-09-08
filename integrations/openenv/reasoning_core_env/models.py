"""Action and observation models for the Reasoning Core environment."""

from typing import Any

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class ReasoningCoreAction(Action):
    """Submit an answer to the current symbolic reasoning problem."""

    answer: str = Field(..., description="The final answer to the current problem")


class ReasoningCoreObservation(Observation):
    """A symbolic reasoning problem or its scored result."""

    prompt: str | None = Field(default=None, description="Problem to solve")
    score: float | None = Field(default=None, description="Answer score from 0 to 1")
    correct_answer: str | None = Field(
        default=None,
        description="Reference answer, revealed after submission",
    )
    task_name: str | None = Field(default=None, description="Reasoning task family")
    dataset_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Metadata associated with the dataset example",
    )
