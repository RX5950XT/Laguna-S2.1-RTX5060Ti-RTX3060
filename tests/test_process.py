import tempfile
import unittest
from pathlib import Path

from laguna_deploy.config import DeploymentPaths, Profile
from laguna_deploy.process import create_launch_spec, process_started_successfully


class LaunchSpecTests(unittest.TestCase):
    def test_requires_runtime_and_model_under_project(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = DeploymentPaths.from_root(Path(directory))
            with self.assertRaises(ValueError):
                create_launch_spec(
                    executable=Path("C:/outside/llama-server.exe"),
                    model=paths.models / "model.gguf",
                    paths=paths,
                    profile=Profile(),
                )

    def test_launch_spec_contains_project_local_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = DeploymentPaths.from_root(Path(directory))
            spec = create_launch_spec(
                executable=paths.build / "llama-server.exe",
                model=paths.models / "model.gguf",
                paths=paths,
                profile=Profile(),
            )

            self.assertEqual(spec.arguments[spec.arguments.index("--ctx-size") + 1], "262144")
            self.assertEqual(spec.arguments[spec.arguments.index("--host") + 1], "127.0.0.1")
            self.assertTrue(Path(spec.environment["HF_HOME"]).is_relative_to(paths.root))
            self.assertTrue(spec.stdout_log.is_relative_to(paths.root))


class ProcessHealthTests(unittest.TestCase):
    def test_rejects_process_that_exited_during_model_load(self) -> None:
        self.assertFalse(process_started_successfully(has_exited=True, health_status=None))

    def test_requires_successful_http_health(self) -> None:
        self.assertFalse(process_started_successfully(has_exited=False, health_status=503))
        self.assertTrue(process_started_successfully(has_exited=False, health_status=200))


if __name__ == "__main__":
    unittest.main()
