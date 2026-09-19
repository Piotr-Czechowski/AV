"""Smoke checks that do not need a CARLA server."""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from carla_multiserver_launcher import (  # noqa: E402
    build_cmd, docker_gpu_device, graphics_adapter_for, resolve_binary)
from rl_configuration import Actions  # noqa: E402


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
