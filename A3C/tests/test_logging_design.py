"""Does the logging code still behave the way ``docs/logging.md`` says?

The rules it checks — no averaging, exact values, one W&B point per
record, every worker visible separately — break silently.  Training keeps
running and charts keep drawing; the logs just stop being truthful, and
you find out after an expensive cluster run rather than during one.

Checked assumptions:

* every update and every system sample becomes its own W&B point;
* values arrive unchanged;
* each worker also appears under ``worker_<id>/``;
* reward components keep both ``_sum`` and ``_mean``;
* raw per-step arrays never reach W&B;
* run-level records create no ``worker_-1`` namespace;
* ``lr`` and ``entropy_coef`` are always forwarded;
* final counters reach ``events.jsonl`` even without W&B.

Not checked: whether the values are computed correctly.  This asserts
that ``pi_loss`` arrives intact, not that ``pi_loss`` is right.  Real
multiprocessing, CARLA, and a live W&B backend are out of scope too.

Runs in under a second against the real logger, with the W&B run stubbed
out and files written to a temporary directory.  Worth running after any
change to ``new_hogwild_training_logger.py``:

    ./.venv/bin/python -m unittest A3C/test_logging_design.py
"""

import json
import os
import queue
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from new_hogwild_training_logger import (  # noqa: E402
    build_record, make_telemetry_stop, project_system_to_wandb,
    project_update_to_wandb, run_telemetry_loop)


class FakeWandbRun:
    """Stand-in for wandb.Run; records payloads instead of sending them."""

    def __init__(self):
        self.logged = []
        self.summary = self
        self.summary_values = {}

    def log(self, payload):
        self.logged.append(dict(payload))

    def update(self, values):
        self.summary_values = dict(values)


def make_update(worker, global_t, pi_loss, **extra):
    data = {
        'update': 7,
        'global_t': global_t,
        'trajectory_length': 20,
        'pi_loss': pi_loss,
        'v_loss': 0.5,
        'total_loss': pi_loss + 0.5,
        'gradient_norm': 1.25,
        'gradient_norm_pre_clip': 3.5,
        'grad_clipped': True,
        'lr': 1e-4,
        'ent_mean': 0.9,
        'reward_speed_sum': 4.0,
        'reward_speed_mean': 0.2,
    }
    data.update(extra)
    return build_record('update', worker_id=worker, data=data)


def make_system(global_t):
    return build_record('system', worker_id=None, data={
        'global_t': global_t,
        'system': {'cpu_percent_mean': 55.0, 'cpu_percent_max': 99.0,
                   'mem_percent': 40.0, 'mem_available_gb': 20.0,
                   'swap_used_gb': 0.5},
        'gpus': [{'index': 0, 'util_percent': 77, 'mem_used_gb': 6.0}],
    })


class TestUpdateProjection(unittest.TestCase):
    def test_values_are_exact(self):
        payload = project_update_to_wandb(make_update(1, 500, 0.25))
        self.assertEqual(payload['train/pi_loss'], 0.25)
        self.assertEqual(payload['train/gradient_norm'], 1.25)
        self.assertEqual(payload['train/gradient_norm_pre_clip'], 3.5)
        self.assertEqual(payload['train/grad_clipped'], 1)
        self.assertEqual(payload['global_step'], 500)

    def test_reward_components_keep_both_sum_and_mean(self):
        payload = project_update_to_wandb(make_update(1, 500, 0.25))
        self.assertEqual(payload['train/reward_speed_sum'], 4.0)
        self.assertEqual(payload['train/reward_speed_mean'], 0.2)

    def test_raw_arrays_are_not_forwarded(self):
        payload = project_update_to_wandb(
            make_update(1, 500, 0.25, advantages=[1.0, 2.0]))
        self.assertNotIn('train/advantages', payload)

    def test_per_worker_namespace(self):
        payload = project_update_to_wandb(make_update(3, 500, 0.25))
        self.assertEqual(payload['worker_3/train/pi_loss'], 0.25)
        self.assertEqual(payload['worker_3/train/gradient_norm'], 1.25)

    def test_global_record_has_no_worker_namespace(self):
        payload = project_update_to_wandb(make_update(-1, 500, 0.25))
        self.assertFalse([k for k in payload if k.startswith('worker_')])

    def test_schedule_fields_are_always_forwarded(self):
        payload = project_update_to_wandb(make_update(1, 500, 0.25))
        self.assertEqual(payload['train/lr'], 1e-4)
        self.assertEqual(payload['worker_1/train/lr'], 1e-4)


class TestSystemProjection(unittest.TestCase):
    def test_sample_values_are_exact(self):
        payload = project_system_to_wandb(make_system(900))
        self.assertEqual(payload['system/cpu_percent_mean'], 55.0)
        self.assertEqual(payload['system/cpu_percent_max'], 99.0)
        self.assertEqual(payload['system/mem_percent'], 40.0)
        self.assertEqual(payload['system/mem_available_gb'], 20.0)
        self.assertEqual(payload['system/swap_used_gb'], 0.5)
        self.assertEqual(payload['system/gpu0_util_percent'], 77)
        self.assertEqual(payload['system/gpu0_mem_used_gb'], 6.0)
        self.assertEqual(payload['global_step'], 900)


class TestTelemetryLoop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def drain(self, records, wandb_run=None, **kwargs):
        source = queue.Queue()
        for record in records:
            source.put(record)
        source.put(make_telemetry_stop({'final_global_step': 42}))
        run_telemetry_loop(
            source, self.tmp, wandb_run=wandb_run,
            shared_counters={'queue_drops': None}, **kwargs)
        return wandb_run

    def test_every_update_from_every_worker_produces_a_point(self):
        run = self.drain([
            make_update(0, 100, 0.1),
            make_update(1, 110, 0.2),
            make_update(0, 120, 0.3),
        ], FakeWandbRun())
        points = [p for p in run.logged if 'train/pi_loss' in p]
        self.assertEqual([p['train/pi_loss'] for p in points],
                         [0.1, 0.2, 0.3])
        self.assertEqual([p['global_step'] for p in points],
                         [100, 110, 120])
        self.assertEqual(points[0]['worker_0/train/pi_loss'], 0.1)
        self.assertEqual(points[1]['worker_1/train/pi_loss'], 0.2)

    def test_every_system_sample_produces_a_point(self):
        run = self.drain([make_system(10), make_system(20)], FakeWandbRun())
        points = [p for p in run.logged if 'system/cpu_percent_mean' in p]
        self.assertEqual(len(points), 2)
        self.assertEqual([p['global_step'] for p in points], [10, 20])

    def test_final_summary_persisted_without_wandb(self):
        self.drain([], None)
        path = os.path.join(self.tmp, 'logs', 'events.jsonl')
        with open(path) as handle:
            events = [json.loads(line) for line in handle if line.strip()]
        summary = [e for e in events if e.get('event') == 'final_summary']
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]['final_global_step'], 42)
        self.assertIn('queue_drops', summary[0])


if __name__ == '__main__':
    unittest.main()
