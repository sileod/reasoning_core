"""FastAPI application for the Reasoning Core environment."""

import os

from openenv.core.env_server.http_server import create_app

try:
    from ..models import ReasoningCoreAction, ReasoningCoreObservation
    from .reasoning_core_environment import ReasoningCoreEnvironment
except ImportError:
    from models import ReasoningCoreAction, ReasoningCoreObservation
    from server.reasoning_core_environment import ReasoningCoreEnvironment


def create_reasoning_core_environment() -> ReasoningCoreEnvironment:
    return ReasoningCoreEnvironment()


app = create_app(
    create_reasoning_core_environment,
    ReasoningCoreAction,
    ReasoningCoreObservation,
    env_name="reasoning_core",
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "16")),
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
