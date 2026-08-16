from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DeploymentPaths:
    root: Path
    models: Path
    huggingface_cache: Path
    uv_cache: Path
    temporary: Path
    build: Path
    logs: Path
    benchmark_results: Path

    @classmethod
    def from_root(cls, root: Path) -> "DeploymentPaths":
        if not root.is_absolute():
            raise ValueError("專案根目錄必須是絕對路徑")
        resolved = root.resolve()
        return cls(
            root=resolved,
            models=resolved / "models",
            huggingface_cache=resolved / ".cache" / "huggingface",
            uv_cache=resolved / ".cache" / "uv",
            temporary=resolved / "tmp",
            build=resolved / "build",
            logs=resolved / "logs",
            benchmark_results=resolved / "benchmarks" / "results",
        )

    def large_artifact_paths(self) -> tuple[Path, ...]:
        return (
            self.models,
            self.huggingface_cache,
            self.uv_cache,
            self.temporary,
            self.build,
            self.logs,
            self.benchmark_results,
        )

    def cache_environment(self) -> dict[str, str]:
        values = {
            "HF_HOME": self.huggingface_cache,
            "HF_HUB_CACHE": self.huggingface_cache / "hub",
            "HUGGINGFACE_HUB_CACHE": self.huggingface_cache / "hub",
            "HF_XET_CACHE": self.huggingface_cache / "xet",
            "TRANSFORMERS_CACHE": self.huggingface_cache / "transformers",
            "UV_CACHE_DIR": self.uv_cache,
            "XDG_CACHE_HOME": self.root / ".cache",
            "TMP": self.temporary,
            "TEMP": self.temporary,
        }
        return {name: str(path.resolve()) for name, path in values.items()}


@dataclass(frozen=True)
class Profile:
    context_size: int = 262_144
    minimum_decode_tps: float = 20.0
    maximum_quality_loss: float = 0.049_999

    def __post_init__(self) -> None:
        if self.context_size < 262_144:
            raise ValueError("Laguna S 2.1 部署不得縮短 256K context")
        if self.minimum_decode_tps < 20.0:
            raise ValueError("吞吐門檻不得低於 20 tok/s")
        if not 0.0 <= self.maximum_quality_loss < 0.05:
            raise ValueError("品質損失門檻必須嚴格小於 5%")


def build_server_arguments(
    model: Path,
    profile: Profile,
    paths: DeploymentPaths,
) -> list[str]:
    resolved_model = model.resolve()
    if not resolved_model.is_relative_to(paths.root):
        raise ValueError("模型必須位於專案目錄內")
    return [
        "--model",
        str(resolved_model),
        "--ctx-size",
        str(profile.context_size),
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--parallel",
        "1",
        "--jinja",
        "--flash-attn",
        "on",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--threads",
        "8",
        "--threads-batch",
        "16",
    ]
