"""Client for the Reasoning Core OpenEnv server."""

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ReasoningCoreAction, ReasoningCoreObservation


class ReasoningCoreEnv(
    EnvClient[ReasoningCoreAction, ReasoningCoreObservation, State]
):
    """HTTP/WebSocket client for Reasoning Core."""

    def _step_payload(self, action: ReasoningCoreAction) -> dict[str, Any]:
        return {"answer": action.answer}

    def _parse_result(
        self,
        payload: dict[str, Any],
    ) -> StepResult[ReasoningCoreObservation]:
        obs_data = payload.get("observation", {})
        observation = ReasoningCoreObservation(
            prompt=obs_data.get("prompt"),
            score=obs_data.get("score"),
            correct_answer=obs_data.get("correct_answer"),
            task_name=obs_data.get("task_name"),
            dataset_metadata=obs_data.get("dataset_metadata"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
