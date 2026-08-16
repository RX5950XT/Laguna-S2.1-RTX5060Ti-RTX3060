import tempfile
import unittest
from pathlib import Path

from laguna_deploy.config import DeploymentPaths, Profile, build_server_arguments


class DeploymentPathsTests(unittest.TestCase):
    def test_all_large_artifact_paths_stay_under_project(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = DeploymentPaths.from_root(Path(directory))

            for path in paths.large_artifact_paths():
                self.assertTrue(path.is_relative_to(paths.root))

    def test_rejects_root_that_is_not_absolute(self) -> None:
        with self.assertRaises(ValueError):
            DeploymentPaths.from_root(Path("relative"))

    def test_cache_environment_is_fully_project_local(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = DeploymentPaths.from_root(Path(directory))

            for value in paths.cache_environment().values():
                self.assertTrue(Path(value).is_relative_to(paths.root))


class ProfileTests(unittest.TestCase):
    def test_context_cannot_be_reduced_below_256k(self) -> None:
        with self.assertRaises(ValueError):
            Profile(context_size=131072)

    def test_default_profile_preserves_acceptance_gates(self) -> None:
        profile = Profile()

        self.assertEqual(profile.context_size, 262144)
        self.assertEqual(profile.minimum_decode_tps, 20.0)
        self.assertLess(profile.maximum_quality_loss, 0.05)


class ServerArgumentsTests(unittest.TestCase):
    def test_server_is_local_only_and_preserves_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = DeploymentPaths.from_root(Path(directory))
            model = paths.models / "laguna.gguf"
            arguments = build_server_arguments(model, Profile(), paths)

        joined = " ".join(arguments)
        self.assertIn("--ctx-size 262144", joined)
        self.assertIn("--host 127.0.0.1", joined)
        self.assertIn("--parallel 1", joined)
        self.assertNotIn("0.0.0.0", joined)

    def test_rejects_model_outside_project(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = DeploymentPaths.from_root(Path(directory))
            with self.assertRaises(ValueError):
                build_server_arguments(Path("C:/outside/model.gguf"), Profile(), paths)


if __name__ == "__main__":
    unittest.main()
