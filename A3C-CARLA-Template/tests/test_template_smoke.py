"""Smoke checks that do not need a CARLA server."""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import importlib
import subprocess
import tempfile

from carla_multiserver_launcher import (  # noqa: E402
    build_cmd, docker_gpu_device, graphics_adapter_for, hang_warmup_allows_miss,
    launcher_config_error, parse_args, resolve_binary)
from rl_configuration import Actions  # noqa: E402
from town03 import SCENARIOS  # noqa: E402
import settings  # noqa: E402


class ActionSpaceTests(unittest.TestCase):
    def test_names_match_control_map(self):
        self.assertEqual(
            set(Actions.ACTION_CONTROL), set(Actions.ACTIONS_NAMES))
        self.assertEqual(
            set(Actions.ACTIONS_VALUES.values()), set(Actions.ACTIONS_NAMES))
        for name in Actions.ACTIONS_NAMES.values():
            self.assertTrue(hasattr(Actions, name))
            self.assertEqual(
                getattr(Actions, name), Actions.ACTIONS_VALUES[name])


class LauncherCmdTests(unittest.TestCase):
    def test_apptainer_argv(self):
        cmd = build_cmd(
            "apptainer", 2000, 0, "/path/to/carla.sif",
            "/home/carla/CarlaUE4.sh")
        self.assertEqual(cmd[:4], ["apptainer", "exec", "--nv", "/path/to/carla.sif"])
        self.assertIn("-carla-rpc-port=2000", cmd)
        self.assertIn("-graphicsadapter=1", cmd)
        self.assertIn("-RenderOffScreen", cmd)

    def test_docker_uses_host_network(self):
        cmd = build_cmd(
            "docker", 2100, 1, "carlasim/carla:0.9.15",
            "/home/carla/CarlaUE4.sh")
        self.assertEqual(cmd[0], "docker")
        self.assertIn("--ipc=host", cmd)
        self.assertIn("--network", cmd)
        self.assertEqual(cmd[cmd.index("--network") + 1], "host")
        self.assertIn("--gpus", cmd)
        self.assertEqual(cmd[cmd.index("--gpus") + 1], "device=1")
        self.assertIn("--entrypoint", cmd)
        self.assertEqual(
            cmd[cmd.index("--entrypoint") + 1], "/home/carla/CarlaUE4.sh")
        self.assertIn("carlasim/carla:0.9.15", cmd)
        self.assertNotIn("/home/carla/CarlaUE4.sh", cmd[cmd.index("carlasim/carla:0.9.15") + 1:])
        self.assertIn("-carla-rpc-port=2100", cmd)
        self.assertIn("-graphicsadapter=0", cmd)

    def test_native_argv(self):
        cmd = build_cmd("native", 2000, 0, "", "/opt/carla/CarlaUE4.sh")
        self.assertEqual(cmd[0], "/opt/carla/CarlaUE4.sh")
        self.assertIn("-carla-rpc-port=2000", cmd)
        self.assertEqual(graphics_adapter_for("native", 0), 0)
        self.assertEqual(graphics_adapter_for("apptainer", 0), 1)

    def test_docker_uses_host_gpu_from_cuda_visible_devices(self):
        old = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
        try:
            self.assertEqual(docker_gpu_device(0), "2")
            self.assertEqual(docker_gpu_device(1), "3")
            cmd = build_cmd(
                "docker", 2000, 0, "carlasim/carla:0.9.15",
                "/home/carla/CarlaUE4.sh")
            self.assertEqual(cmd[cmd.index("--gpus") + 1], "device=2")
        finally:
            if old is None:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old

    def test_image_required(self):
        with self.assertRaises(ValueError):
            build_cmd("apptainer", 2000, 0, "", "/home/carla/CarlaUE4.sh")
        with self.assertRaises(ValueError):
            build_cmd("docker", 2000, 0, "", "/home/carla/CarlaUE4.sh")

    def test_resolve_native_binary_under_carla_path(self):
        path = resolve_binary("CarlaUE4.sh", "/opt/CARLA_0.9.15", "native")
        self.assertEqual(path, "/opt/CARLA_0.9.15/CarlaUE4.sh")

    def test_hang_warmup_skips_strikes_until_first_listen(self):
        self.assertTrue(hang_warmup_allows_miss(False))
        self.assertFalse(hang_warmup_allows_miss(True))

    def test_port_step_must_be_at_least_two(self):
        args = parse_args(["--runtime", "native", "--port-step", "1"])
        error = launcher_config_error(args)
        self.assertIsNotNone(error)
        self.assertIn("port-step", error)
        args_ok = parse_args(["--runtime", "native", "--port-step", "2"])
        self.assertIsNone(launcher_config_error(args_ok))

    def test_change_me_image_rejected_for_containers(self):
        docker_args = parse_args([
            "--runtime", "docker", "--image", "CHANGE_ME"])
        error = launcher_config_error(docker_args)
        self.assertIsNotNone(error)
        self.assertIn("image", error)
        apptainer_args = parse_args([
            "--runtime", "apptainer", "--image", "foo.sif"])
        self.assertIsNone(launcher_config_error(apptainer_args))


class ScenarioCatalogTests(unittest.TestCase):
    def test_town03_scenario_14_routes(self):
        spec = SCENARIOS[14]
        self.assertEqual(spec["select"], "cycle")
        self.assertEqual(spec["routes"][0], (28, 155))
        self.assertEqual(len(spec["routes"]), 5)

    def test_missing_town04_module_raises(self):
        with self.assertRaises(ImportError):
            importlib.import_module("town04")

    def test_episode_cap_constants(self):
        self.assertEqual(settings.EPISODE_MAX_DECISIONS, 200)
        self.assertEqual(settings.ACTION_REPEAT, 2)
        self.assertEqual(
            settings.STEP_COUNTER,
            settings.EPISODE_MAX_DECISIONS * settings.ACTION_REPEAT)


class CheckpointLookupTests(unittest.TestCase):
    def test_find_latest_ignores_best_checkpoint(self):
        try:
            from run_a3c import find_latest_checkpoint
        except ImportError:
            self.skipTest("torch is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            best = os.path.join(tmp, "best_checkpoint.pth")
            last = os.path.join(tmp, "checkpoint.pth")
            with open(best, "w") as handle:
                handle.write("best")
            with open(last, "w") as handle:
                handle.write("last")
            with open(os.path.join(tmp, "checkpoint_step.txt"), "w") as handle:
                handle.write("42")
            path, step = find_latest_checkpoint(tmp)
            self.assertEqual(path, last)
            self.assertEqual(step, 42)
            self.assertNotIn("best_checkpoint", path)

    def test_find_latest_empty_without_last(self):
        try:
            from run_a3c import find_latest_checkpoint
        except ImportError:
            self.skipTest("torch is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "best_checkpoint.pth"), "w") as handle:
                handle.write("best")
            path, step = find_latest_checkpoint(tmp)
            self.assertIsNone(path)
            self.assertEqual(step, 0)


class ShellSyntaxTests(unittest.TestCase):
    def test_bash_n_example_scripts(self):
        scripts = [
            os.path.join(ROOT, "examples", "local", "run_servers.sh"),
            os.path.join(ROOT, "examples", "docker", "run_servers.sh"),
            os.path.join(ROOT, "examples", "run_train.sh"),
            os.path.join(ROOT, "examples", "hpc", "train.slurm"),
        ]
        for script in scripts:
            result = subprocess.run(
                ["bash", "-n", script],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(
                result.returncode, 0,
                "{}: {}".format(script, result.stderr.decode("utf-8", "replace")))

    def test_docker_wrapper_rejects_change_me(self):
        script = os.path.join(ROOT, "examples", "docker", "run_servers.sh")
        with open(script) as handle:
            text = handle.read()
        self.assertIn("CHANGE_ME", text)


class ModelImportTests(unittest.TestCase):
    def test_shared_actor_critic_shapes(self):
        try:
            import torch
            from models.shared_actor_critic import SharedActorCritic
        except ImportError:
            self.skipTest('torch is not installed')
        model = SharedActorCritic([250, 250, 3], 10, 1, torch.device("cpu"))
        images = torch.zeros(1, 3, 250, 250)
        speed = torch.zeros(1, 1)
        maneuver = torch.zeros(1, dtype=torch.long)
        policy, value = model(images, speed, maneuver)
        self.assertEqual(tuple(policy.shape), (1, 10))
        self.assertEqual(tuple(value.shape), (1, 1))


if __name__ == "__main__":
    unittest.main()
