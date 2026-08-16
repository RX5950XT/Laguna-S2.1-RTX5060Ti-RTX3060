from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import DeploymentPaths, Profile, build_server_arguments


@dataclass(frozen=True)
class LaunchSpec:
    executable: Path
    arguments: tuple[str, ...]
    environment: dict[str, str]
    stdout_log: Path
    stderr_log: Path
    pid_file: Path


def create_launch_spec(
    *,
    executable: Path,
    model: Path,
    paths: DeploymentPaths,
    profile: Profile,
) -> LaunchSpec:
    resolved_executable = executable.resolve()
    if not resolved_executable.is_relative_to(paths.root):
        raise ValueError("llama-server 必須位於專案目錄內")
    return LaunchSpec(
        executable=resolved_executable,
        arguments=tuple(build_server_arguments(model, profile, paths)),
        environment=paths.cache_environment(),
        stdout_log=paths.logs / "server.stdout.log",
        stderr_log=paths.logs / "server.stderr.log",
        pid_file=paths.logs / "server.pid",
    )


def process_started_successfully(*, has_exited: bool, health_status: int | None) -> bool:
    return not has_exited and health_status == 200
