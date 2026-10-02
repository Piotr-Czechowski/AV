"""Smoke checks that do not need a CARLA server."""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import subprocess
import tempfile
from types import SimpleNamespace

from carla_multiserver_launcher import (  # noqa: E402
    build_cmd, docker_container_name, docker_gpu_device,
    hang_warmup_allows_miss, launcher_config_error, parse_args)
from rl_configuration import ACTIONS, N_ACTIONS, observation_spec  # noqa: E402
from town03 import SCENARIOS  # noqa: E402


class ActionSpaceTests(unittest.TestCase):
    def test_action_table_rows(self):
        self.assertEqual(N_ACTIONS, len(ACTIONS))
        self.assertEqual(len({row[0] for row in ACTIONS}), N_ACTIONS)
        for name, throttle, brake, steer in ACTIONS:
            self.assertTrue(0 <= throttle <= 1 and 0 <= brake <= 1)
            self.assertTrue(-1 <= steer <= 1)


class LauncherCmdTests(unittest.TestCase):
    def test_apptainer_argv(self):
        cmd = build_cmd(
            "apptainer", 2000, 0, "/path/to/carla.sif",
            "/home/carla/CarlaUE4.sh")
        self.assertEqual(cmd[:4], ["apptainer", "exec", "--nv", "/path/to/carla.sif"])
        self.assertIn("-carla-rpc-port=2000", cmd)
        self.assertIn("-graphicsadapter=1", cmd)
        self.assertIn("-RenderOffScreen", cmd)
        self.assertNotIn("--carla-server", cmd)

    def test_docker_uses_host_network(self):
        cmd = build_cmd(
            "docker", 2100, 1, "carlasim/carla:0.9.15",
            "/home/carla/CarlaUE4.sh")
        self.assertEqual(cmd[0], "docker")
        self.assertIn("--ipc=host", cmd)
        self.assertEqual(
            cmd[cmd.index("--name") + 1], docker_container_name(2100))
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

    def test_hang_warmup_skips_strikes_until_first_listen(self):
        self.assertTrue(hang_warmup_allows_miss(False))
        self.assertFalse(hang_warmup_allows_miss(True))

    def test_port_step_must_be_at_least_three(self):
        args = parse_args([
            "--runtime", "docker", "--image", "img", "--port-step", "2"])
        error = launcher_config_error(args)
        self.assertIsNotNone(error)
        self.assertIn("port-step", error)
        args_ok = parse_args([
            "--runtime", "docker", "--image", "img", "--port-step", "3"])
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
            os.path.join(ROOT, "examples", "docker", "train.sh"),
            os.path.join(ROOT, "examples", "apptainer", "train.slurm"),
        ]
        for script in scripts:
            result = subprocess.run(
                ["bash", "-n", script],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(
                result.returncode, 0,
                "{}: {}".format(script, result.stderr.decode("utf-8", "replace")))


class ModelContractTests(unittest.TestCase):
    def test_obs_to_tensors_and_model_shapes(self):
        try:
            import numpy as np
            import torch
            from a3c_core import obs_to_tensors
            from model import build_model
        except ImportError:
            self.skipTest('torch or carla is not installed')
        spec = observation_spec(SimpleNamespace(res=64))
        obs = {
            "image": np.zeros((3, 64, 64), dtype=np.float32),
            "speed": np.array([0.2], dtype=np.float32),
            "maneuver": np.array(2, dtype=np.int64),
        }
        device = torch.device("cpu")
        tensors = obs_to_tensors(obs, device)
        self.assertEqual(tuple(tensors["image"].shape), (1, 3, 64, 64))
        self.assertEqual(tuple(tensors["speed"].shape), (1, 1))
        self.assertEqual(tuple(tensors["maneuver"].shape), (1,))
        self.assertEqual(tensors["maneuver"].dtype, torch.int64)
        policy, value = build_model(spec, N_ACTIONS, device)(tensors)
        self.assertEqual(tuple(policy.shape), (1, N_ACTIONS))
        self.assertEqual(tuple(value.shape), (1, 1))


if __name__ == "__main__":
    unittest.main()
