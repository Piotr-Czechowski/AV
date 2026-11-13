"""TEST CODE: verification of the metrics and their logging.

This file is not used by training.  It is a test suite that checks that the
metrics are computed and logged correctly:

- ``reward_stats``: per-worker windows of the last 100 episodes, run minimum
  and maximum, global aggregates over the workers, reward thresholds,
  checkpoint round trip, training clock, lock, concurrent processes;
- the path to the outputs: local JSONL records, W&B projections and run
  summary, telemetry loop, and that every value sent to W&B is also in the
  local files under the same name.

Needs no torch, no CARLA, and no W&B.  Run it from the template directory:

    python -m unittest discover -s test
"""

import contextlib
import io
import json
import multiprocessing
import os
import queue
import re
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import reward_stats  # noqa: E402
from reward_stats import (  # noqa: E402
    RewardTracker, clean_thresholds, threshold_key, threshold_label,
    threshold_summary)
from training_logger import (  # noqa: E402
    TrainingLogger, build_record, make_telemetry_stop,
    project_episode_to_wandb, project_episode_to_wandb_summary,
    project_update_to_wandb, run_telemetry_loop)

INF = float('inf')
EPISODE_STAT_KEYS = {
    'is_new_best', 'reward_mean_100', 'goal_distance_mean_100', 'reward_min',
    'reward_max', 'goal_distance_min', 'goal_distance_max',
    'global/reward_mean_100', 'global/goal_distance_mean_100',
    'global/reward_min', 'global/reward_max', 'global/goal_distance_min',
    'global/goal_distance_max', 'global/elapsed_training_s'}
THRESHOLD_FIELDS = ('global_step', 'elapsed_training_s', 'global_episode',
                    'value')


# -- helpers ---------------------------------------------------------------

def small_window(size):
    """Decorator: run one test with a reward window of ``size`` episodes."""
    return mock.patch.object(reward_stats, 'REWARD_WINDOW', size)


def at_time(now):
    """Patch ``time.time`` so the tracker's clock reads ``now``."""
    return mock.patch('reward_stats.time.time', return_value=now)


def hits_of(tracker):
    return tracker.state_dict()['hits']


def record_many(tracker, worker_id, rewards, start_episode=1):
    """Record ``rewards`` in order; return the new hits of all calls."""
    hits = []
    for offset, reward in enumerate(rewards):
        episode = start_episode + offset
        _, new_hits = tracker.record_episode(
            worker_id, reward, global_t=episode * 10,
            global_episode=episode)
        hits.extend(new_hits)
    return hits


def signals(hits):
    return [(hit['signal'], hit['global_episode']) for hit in hits]


class FakeSummary(dict):
    """Dict with ``update`` like ``wandb_run.summary``; keeps every call."""

    def __init__(self, fail=False):
        super().__init__()
        self.fail = fail
        self.calls = []

    def update(self, items):
        if self.fail:
            raise RuntimeError('summary down')
        self.calls.append(dict(items))
        super().update(items)


class FakeRun:
    """Stands in for a W&B run: ``log()`` and ``summary.update()``."""

    def __init__(self, fail_summary=False):
        self.logged = []
        self.summary = FakeSummary(fail=fail_summary)

    def log(self, payload):
        self.logged.append(payload)


def drain(records, run_dir, wandb_run, shared_counters=None,
          final_summary=None):
    """Run the telemetry loop over ``records``; return events.jsonl."""
    telemetry_queue = queue.Queue()
    for record in records:
        telemetry_queue.put(record)
    telemetry_queue.put(make_telemetry_stop(final_summary or {'final': 1}))
    with contextlib.redirect_stdout(io.StringIO()):
        run_telemetry_loop(
            telemetry_queue, run_dir, wandb_run=wandb_run,
            shared_counters=shared_counters)
    with open(os.path.join(run_dir, 'logs', 'events.jsonl')) as handle:
        return [json.loads(line) for line in handle]


def threshold_event():
    return build_record('event', worker_id=1, data={
        'event': 'reward_threshold_reached', 'signal': 'reward_mean_100',
        'threshold': 25.0, 'value': 26.5, 'global_step': 8000,
        'global_episode': 77, 'elapsed_training_s': 1234.5, 'worker': 1})


def episode_record(**fields):
    """An episode record as ``TrainingLogger.log_episode`` writes it."""
    data = {
        'global_step': 3400, 'episode/id': 12, 'reward': 7.5,
        'episode_length': 150, 'episode/reward_per_step': 0.05,
        'episode/duration_s': 4.5, 'episode/success': True,
        'episode/max_speed_kmh': 31.0, 'episode/min_route_distance': 0.5,
        'goal_distance': 42.0, 'episode/collisions': 2,
        'episode/action_0_fraction': 0.75, 'episode/action_1_fraction': 0.25,
        'reward_mean_100': 5.0, 'goal_distance_mean_100': 40.0,
        'reward_min': -8.0, 'reward_max': 11.0,
        'goal_distance_min': 3.0, 'goal_distance_max': 90.0,
        'global/reward_mean_100': 4.0, 'global/goal_distance_mean_100': 45.0,
        'global/reward_min': -20.0, 'global/reward_max': 15.0,
        'global/goal_distance_min': 1.0, 'global/goal_distance_max': 120.0,
        'is_new_best': False, 'global/elapsed_training_s': 987.5,
        'port': 2000,
    }
    data.update(fields)
    return build_record('episode', worker_id=2, data=data)


def _record_in_process(tracker, worker_id, count):
    for index in range(count):
        tracker.record_episode(
            worker_id, worker_id * 1000 + index,
            global_t=index, global_episode=worker_id * 1000 + index)


# -- windows, minimum and maximum, global aggregates -----------------------

class WindowTests(unittest.TestCase):
    @small_window(3)
    def test_worker_windows_and_global_mean_of_means(self):
        tracker = RewardTracker(2)
        record_many(tracker, 0, [1, 2, 3, 4])
        stats, hits = tracker.record_episode(1, 10.0, 50, 5)
        self.assertEqual(set(stats), EPISODE_STAT_KEYS)
        self.assertEqual(hits, [])
        # worker 0 keeps its last 3 episodes, worker 1 has one
        self.assertEqual(stats['reward_mean_100'], 10.0)
        self.assertEqual(tracker.state_dict()['workers'],
                         [[2.0, 3.0, 4.0], [10.0]])
        # the global mean is the mean of the worker means, (3 + 10) / 2,
        # not the mean of the pooled episodes
        self.assertEqual(stats['global/reward_mean_100'], 6.5)
        # wrap: five more episodes of worker 1, worker 0 stays untouched
        record_many(tracker, 1, [11, 12, 13, 14, 15], start_episode=6)
        self.assertEqual(tracker.state_dict()['workers'],
                         [[2.0, 3.0, 4.0], [13.0, 14.0, 15.0]])
        # a worker without episodes does not count in the global mean
        stats, _ = RewardTracker(3).record_episode(1, 4.0, 1, 1)
        self.assertEqual(stats['global/reward_mean_100'], 4.0)

    @small_window(2)
    def test_run_minimum_and_maximum_outlive_the_windows(self):
        tracker = RewardTracker(2)
        stats, _ = tracker.record_episode(0, 5.0, 1, 1, goal_dist=40.0)
        self.assertTrue(stats['is_new_best'])
        self.assertEqual(
            (stats['reward_min'], stats['reward_max']),
            (5.0, 5.0))
        stats, _ = tracker.record_episode(1, -3.0, 2, 2, goal_dist=70.0)
        self.assertFalse(stats['is_new_best'])
        # global: the lowest worker minimum and the highest maximum
        self.assertEqual(
            (stats['reward_min'], stats['reward_max'],
             stats['global/reward_min'], stats['global/reward_max']),
            (-3.0, -3.0, -3.0, 5.0))
        self.assertEqual(
            (stats['global/goal_distance_min'], stats['global/goal_distance_max']),
            (40.0, 70.0))
        # worker 0's window forgets 5.0; its run minimum and maximum do not
        record_many(tracker, 0, [1, 1, 1], start_episode=3)
        stats, _ = tracker.record_episode(0, 6.0, 10, 6, goal_dist=2.0)
        self.assertTrue(stats['is_new_best'])
        self.assertEqual(tracker.state_dict()['workers'][0], [1.0, 6.0])
        self.assertEqual(
            (stats['reward_min'], stats['reward_max']),
            (1.0, 6.0))
        self.assertEqual(
            (stats['goal_distance_min'], stats['goal_distance_max']),
            (2.0, 40.0))
        self.assertEqual(stats['global/goal_distance_min'], 2.0)
        # an equal reward is not a new best
        stats, _ = tracker.record_episode(1, 6.0, 11, 7)
        self.assertFalse(stats['is_new_best'])
        # before any episode a minimum or maximum is None, not inf
        stats, _ = RewardTracker(2).record_episode(0, 1.0, 1, 1)
        self.assertIsNone(stats['goal_distance_min'])
        self.assertIsNone(stats['global/goal_distance_max'])

    @small_window(2)
    def test_goal_distance_window_per_worker(self):
        tracker = RewardTracker(2)
        for distance in (10.0, 20.0, 30.0):
            stats, _ = tracker.record_episode(0, 1.0, 1, 1,
                                              goal_dist=distance)
        self.assertEqual(stats['goal_distance_mean_100'], 25.0)
        self.assertEqual(stats['goal_distance_min'], 10.0)
        # a missing or non-finite distance is not counted
        for distance in (None, INF, float('nan')):
            stats, _ = tracker.record_episode(0, 1.0, 2, 2,
                                              goal_dist=distance)
            self.assertEqual(stats['goal_distance_mean_100'], 25.0)
            self.assertEqual(stats['goal_distance_max'], 30.0)
        # worker 1 has its own window, empty until its first distance
        stats, _ = tracker.record_episode(1, 1.0, 3, 3)
        self.assertIsNone(stats['goal_distance_mean_100'])
        self.assertEqual(stats['global/goal_distance_mean_100'], 25.0)
        stats, _ = tracker.record_episode(1, 1.0, 4, 4, goal_dist=4.0)
        self.assertEqual(stats['goal_distance_mean_100'], 4.0)
        self.assertEqual(stats['global/goal_distance_mean_100'], 14.5)
        state = tracker.state_dict()
        self.assertEqual(state['worker_goal_dists'], [[20.0, 30.0], [4.0]])
        # a snapshot from before the distance windows
        del state['worker_goal_dists']
        restored = RewardTracker(2)
        restored.load_state_dict(state)
        self.assertEqual(restored.state_dict()['worker_goal_dists'],
                         [[], []])


# -- thresholds ------------------------------------------------------------

class ThresholdTests(unittest.TestCase):
    def test_validation_and_label_encoding(self):
        self.assertEqual(clean_thresholds([5, 1.0, 5.0, -2, 0.0, -0.0]),
                         [-2.0, 0.0, 1.0, 5.0])
        self.assertEqual(RewardTracker(1, [5, 1.0, 5.0]).thresholds,
                         [1.0, 5.0])
        for bad in (float('nan'), INF, -INF):
            with self.assertRaises(ValueError):
                clean_thresholds([bad])
        with self.assertRaises(ValueError):  # same label: keys would collide
            clean_thresholds([1000000, 1000001])
        for threshold, label in (
                (0.5, '0p5'), (-50, 'neg50'), (100, '100'), (1e6, '1e06'),
                (-2.5e-05, 'neg2p5eneg05')):
            self.assertEqual(threshold_label(threshold), label)
        pattern = re.compile(r'^[A-Za-z0-9_/]+$')
        for threshold in (0.5, -50, 100, 1e6, -2.5e-05, 1e-9):
            self.assertRegex(
                threshold_key('worker_0/reward_mean_100', threshold,
                              'global_step'),
                pattern)

    def test_episode_reward_fires_once_per_worker(self):
        tracker = RewardTracker(1, thresholds=[10.0])
        hits = record_many(tracker, 0, [3, 10, 50, 12])
        self.assertEqual(hits, [{
            'signal': 'worker_0/episode_reward', 'threshold': 10.0,
            'value': 10.0, 'global_step': 20, 'global_episode': 2,
            'elapsed_training_s': 0.0, 'worker': 0}])
        self.assertEqual(hits_of(tracker), hits)

    @small_window(2)
    def test_worker_signals_fire_once_per_worker(self):
        tracker = RewardTracker(2, thresholds=[5.0, 100.0])
        hits = record_many(tracker, 0, [6])
        self.assertEqual(signals(hits), [('worker_0/episode_reward', 1)])
        # worker 1 has its own signals; its mean fires once its own window
        # of 2 is full
        hits = record_many(tracker, 1, [7, 8], start_episode=2)
        self.assertEqual(signals(hits), [
            ('worker_1/episode_reward', 2), ('worker_1/reward_mean_100', 3)])
        # worker 0's window [6, 4] is full now: its mean and the global mean
        # (5 + 7.5) / 2 fire; its episode signal is spent
        hits = record_many(tracker, 0, [4], start_episode=4)
        self.assertEqual(signals(hits), [
            ('reward_mean_100', 4), ('worker_0/reward_mean_100', 4)])
        self.assertEqual(record_many(tracker, 0, [9], start_episode=5), [])
        self.assertEqual(record_many(tracker, 1, [9], start_episode=6), [])
        summary = tracker.summary()
        prefix = 'threshold/worker_1/reward_mean_100/5/'
        self.assertIs(summary[prefix + 'reached'], True)
        self.assertEqual(summary[prefix + 'global_step'], 30)
        self.assertEqual(summary[prefix + 'value'], 7.5)
        # the worker is in the key, so the summary has no worker field
        self.assertNotIn(prefix + 'worker', summary)
        self.assertEqual(summary['threshold/reward_mean_100/5/value'], 6.25)
        self.assertIs(
            summary['threshold/worker_0/episode_reward/100/reached'], False)

    @small_window(2)
    def test_global_mean_waits_for_every_worker_window(self):
        tracker = RewardTracker(2, thresholds=[5.0])
        # worker 0's window is full, worker 1's is not: no global hit yet
        self.assertEqual(signals(record_many(tracker, 0, [9, 9])), [
            ('worker_0/episode_reward', 1), ('worker_0/reward_mean_100', 2)])
        self.assertEqual(record_many(tracker, 1, [1], start_episode=3), [])
        # both full: (9 + 1) / 2 = 5 reaches 5
        hits = record_many(tracker, 1, [1], start_episode=4)
        self.assertEqual(signals(hits), [('reward_mean_100', 4)])
        self.assertEqual(hits[0]['value'], 5.0)

    @small_window(2)
    def test_summary_reached_unreached_and_minmax(self):
        tracker = RewardTracker(1, thresholds=[1.0, 100.0])
        tracker.set_clock(1000.0, 5.0)
        with at_time(1010.0):
            tracker.record_episode(0, 4.0, 30, 1, goal_dist=12.0)
        with at_time(1020.0):
            tracker.record_episode(0, 2.0, 40, 2, goal_dist=8.0)
        summary = tracker.summary()
        self.assertEqual(
            {key: value for key, value in summary.items()
             if not key.startswith('threshold/')},
            {'worker_0/reward_min': 2.0, 'worker_0/reward_max': 4.0,
             'worker_0/goal_distance_min': 8.0,
             'worker_0/goal_distance_max': 12.0,
             'global/reward_min': 2.0, 'global/reward_max': 4.0,
             'global/goal_distance_min': 8.0,
             'global/goal_distance_max': 12.0})
        prefix = 'threshold/worker_0/episode_reward/1/'
        self.assertIs(summary[prefix + 'reached'], True)
        self.assertEqual(summary[prefix + 'global_step'], 30)
        self.assertEqual(summary[prefix + 'elapsed_training_s'], 15.0)
        self.assertEqual(summary[prefix + 'global_episode'], 1)
        self.assertEqual(summary[prefix + 'value'], 4.0)
        # a pair that was never reached has None fields
        prefix = 'threshold/worker_0/episode_reward/100/'
        self.assertIs(summary[prefix + 'reached'], False)
        for name in THRESHOLD_FIELDS:
            self.assertIsNone(summary[prefix + name])
        # the same shapes come from threshold_summary directly
        self.assertEqual(
            threshold_summary('worker_0/episode_reward', 100.0),
            {key: value for key, value in summary.items()
             if key.startswith(prefix)})
        # nothing recorded yet: no minimum or maximum, every pair unreached
        empty = RewardTracker(2, thresholds=[3.0]).summary()
        self.assertFalse([key for key in empty
                          if not key.startswith('threshold/')])
        self.assertIs(empty['threshold/reward_mean_100/3/reached'], False)


# -- checkpoint, lock, processes -------------------------------------------

class CheckpointTests(unittest.TestCase):
    @small_window(3)
    def test_round_trip_keeps_state_hits_and_elapsed(self):
        tracker = RewardTracker(2, thresholds=[1.0])
        with at_time(500.0):  # before set_clock: no offset, no time
            self.assertEqual(tracker.elapsed_training_s(), 0.0)
        tracker.set_clock(1000.0, 100.0)
        with at_time(1010.0):
            stats, _ = tracker.record_episode(0, 1.0, 1, 1)
        self.assertEqual(stats['global/elapsed_training_s'], 110.0)
        tracker = RewardTracker(2, thresholds=[1.0])
        with at_time(1050.0):
            tracker.set_clock(1000.0, 100.0)
            record_many(tracker, 0, [1, 2, 3, 4])
            record_many(tracker, 1, [9, -9], start_episode=5)
            tracker.record_episode(1, 0.0, 70, 7, goal_dist=5.0)
            state = tracker.state_dict()
        self.assertEqual(state['elapsed_training_s'], 150.0)
        self.assertEqual(state['worker_reward_max'], [4.0, 9.0])
        self.assertEqual(state['worker_goal_dist_min'], [INF, 5.0])
        restored = RewardTracker(2, thresholds=[1.0])
        elapsed = restored.load_state_dict(state)
        self.assertEqual(elapsed, 150.0)
        # the clock of the new session is not set yet, so compare the rest
        self.assertEqual(
            dict(restored.state_dict(), elapsed_training_s=elapsed), state)
        # reached thresholds do not fire again
        self.assertTrue(state['hits'])
        self.assertEqual(record_many(restored, 0, [50, 50, 50], 8), [])
        # the next session continues from the stored time
        restored.set_clock(2000.0, elapsed)
        with at_time(2010.0):
            self.assertEqual(restored.elapsed_training_s(), 160.0)

    @small_window(2)
    def test_load_with_fewer_and_more_workers(self):
        tracker = RewardTracker(3)
        for worker_id in (0, 1, 2):
            record_many(tracker, worker_id, [worker_id + 1] * 2,
                        start_episode=1 + 2 * worker_id)
        small = RewardTracker(2)
        small.load_state_dict(tracker.state_dict())
        state = small.state_dict()
        self.assertEqual(state['workers'], [[1.0, 1.0], [2.0, 2.0]])
        self.assertEqual(state['worker_reward_max'], [1.0, 2.0])

        single = RewardTracker(1)
        record_many(single, 0, [1, 2, 3])
        big = RewardTracker(3)
        big.load_state_dict(single.state_dict())
        state = big.state_dict()
        self.assertEqual(state['workers'], [[2.0, 3.0], [], []])
        self.assertEqual(state['worker_reward_min'], [1.0, INF, INF])
        self.assertEqual(state['worker_reward_max'][1:], [-INF] * 2)
        stats, _ = big.record_episode(2, 5.0, 9, 4)
        self.assertEqual(stats['reward_mean_100'], 5.0)
        self.assertEqual(stats['global/reward_mean_100'], 3.75)  # (2.5 + 5) / 2

    @small_window(3)
    def test_threshold_and_format_changes_on_load(self):
        tracker = RewardTracker(1, thresholds=[1.0, 2.0])
        record_many(tracker, 0, [2.0])
        state = tracker.state_dict()
        self.assertEqual(len(state['hits']), 2)
        # only thresholds configured now are restored
        restored = RewardTracker(1, thresholds=[2.0, 3.0])
        restored.load_state_dict(state)
        self.assertEqual([h['threshold'] for h in hits_of(restored)], [2.0])
        new_hits = record_many(restored, 0, [3.0], start_episode=2)
        self.assertEqual([h['threshold'] for h in new_hits], [3.0])
        # a snapshot with unknown keys still loads; without elapsed time it
        # reports None
        state.update(window=100, num_workers=1)
        del state['elapsed_training_s']
        self.assertIsNone(RewardTracker(1).load_state_dict(state))

    def test_snapshot_from_before_the_run_minimum_and_maximum(self):
        # the format with run-wide windows, best and worst values, and
        # hits of run-wide signals that no longer exist
        old = {
            'global': [5.0], 'long': [5.0], 'best_reward': 9.0,
            'worst_reward': -4.0,
            'workers': [[1.0, 5.0], [-4.0]],
            'worker_goal_dists': [[30.0, 10.0], []],
            'worker_best': [9.0, -4.0], 'worker_worst': [-1.0, -4.0],
            'hits': [{'signal': 'reward_mean', 'threshold': 1.0,
                      'value': 2.0, 'global_t': 9, 'global_episode': 3,
                      'elapsed_training_s': 1.0, 'worker': 0},
                     {'signal': 'worker_0/episode_reward', 'threshold': 1.0,
                      'value': 5.0, 'global_t': 20, 'global_episode': 4,
                      'elapsed_training_s': 2.0, 'worker': 0}],
            'elapsed_training_s': 12.0}
        tracker = RewardTracker(2, thresholds=[1.0])
        self.assertEqual(tracker.load_state_dict(old), 12.0)
        state = tracker.state_dict()
        self.assertEqual(state['worker_reward_min'], [-1.0, -4.0])
        self.assertEqual(state['worker_reward_max'], [9.0, -4.0])
        # goal-distance minimum and maximum come from the saved window
        self.assertEqual(state['worker_goal_dist_min'], [10.0, INF])
        self.assertEqual(state['worker_goal_dist_max'], [30.0, -INF])
        # the dropped run-wide signal is gone; the old step name is read
        self.assertEqual(state['hits'], [{
            'signal': 'worker_0/episode_reward', 'threshold': 1.0,
            'value': 5.0, 'global_step': 20, 'global_episode': 4,
            'elapsed_training_s': 2.0, 'worker': 0}])

    def test_reads_do_not_hang_on_a_held_lock(self):
        tracker = RewardTracker(1, thresholds=[1.0])
        tracker.record_episode(0, 2.0, 1, 1)
        self.assertTrue(tracker.lock.acquire())  # a killed worker's lock
        try:
            with mock.patch.object(reward_stats, 'LOCK_TIMEOUT_S', 0.05):
                for read in (tracker.state_dict, tracker.summary):
                    out = io.StringIO()
                    with contextlib.redirect_stdout(out):
                        result = read()
                    self.assertEqual(
                        out.getvalue().count('[REWARD_STATS] lock busy'), 1)
                    self.assertTrue(result)
        finally:
            tracker.lock.release()
        # a free lock is released again after a read
        tracker.state_dict()
        self.assertTrue(tracker.lock.acquire(timeout=1.0))
        tracker.lock.release()

    def test_concurrent_processes_through_one_tracker(self):
        workers, count = 3, 40
        tracker = RewardTracker(workers, thresholds=[1500.0])
        processes = [
            multiprocessing.Process(
                target=_record_in_process, args=(tracker, worker_id, count))
            for worker_id in range(workers)]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=60)
            self.assertEqual(process.exitcode, 0)
        state = tracker.state_dict()
        # 40 episodes fit in each worker window of 100
        for worker_id in range(workers):
            self.assertEqual(
                state['workers'][worker_id],
                [float(worker_id * 1000 + i) for i in range(count)])
        self.assertEqual(state['worker_reward_min'], [0.0, 1000.0, 2000.0])
        self.assertEqual(state['worker_reward_max'], [39.0, 1039.0, 2039.0])
        # only worker 2 reaches 1500, and its threshold fires once
        self.assertEqual(signals(state['hits']),
                         [('worker_2/episode_reward', 2000)])


# -- logging ---------------------------------------------------------------

class LoggingTests(unittest.TestCase):
    def test_wandb_episode_projection(self):
        payload = project_episode_to_wandb(episode_record())
        self.assertEqual(payload, {
            'global_step': 3400,
            # this one episode
            'episode/id': 12, 'episode/reward_per_step': 0.05,
            'episode/duration_s': 4.5, 'episode/success': 1,
            'episode/max_speed_kmh': 31.0,
            'episode/min_route_distance': 0.5, 'episode/collisions': 2,
            'episode/action_0_fraction': 0.75,
            'episode/action_1_fraction': 0.25,
            # this worker
            'worker_2/reward': 7.5, 'worker_2/episode_length': 150,
            'worker_2/goal_distance': 42.0,
            'worker_2/reward_mean_100': 5.0,
            'worker_2/goal_distance_mean_100': 40.0,
            # aggregates over the workers
            'global/reward_mean_100': 4.0,
            'global/goal_distance_mean_100': 45.0,
            'global/elapsed_training_s': 987.5,
        })
        # an interrupted episode stays local
        self.assertEqual(
            project_episode_to_wandb(episode_record(interrupted=True)), {})
        # a run-level record has no worker keys
        run_level = build_record('episode', worker_id=-1, data={
            'global_step': 1, 'goal_distance': 3.0, 'reward_mean_100': 1.0})
        self.assertFalse([key for key in project_episode_to_wandb(run_level)
                          if key.startswith('worker_')])

    def test_wandb_episode_summary_projection(self):
        self.assertEqual(project_episode_to_wandb_summary(episode_record()), {
            'worker_2/reward_min': -8.0, 'worker_2/reward_max': 11.0,
            'worker_2/goal_distance_min': 3.0,
            'worker_2/goal_distance_max': 90.0,
            'global/reward_min': -20.0, 'global/reward_max': 15.0,
            'global/goal_distance_min': 1.0,
            'global/goal_distance_max': 120.0})
        # no distance yet: those keys are left out
        no_dist = episode_record(goal_distance_min=None,
                                 goal_distance_max=None)
        self.assertNotIn('worker_2/goal_distance_min',
                         project_episode_to_wandb_summary(no_dist))
        self.assertEqual(project_episode_to_wandb_summary(
            episode_record(interrupted=True)), {})

    def test_wandb_update_projection_has_one_place_per_value(self):
        record = build_record('update', worker_id=1, data={
            'global_step': 500, 'update': 25, 'is_terminal': False,
            'train/trajectory_length': 20, 'train/pi_loss': 0.1,
            'train/v_loss': 0.2, 'train/total_loss': 0.3,
            'train/gradient_norm': 1.5, 'train/gradient_norm_pre_clip': 2.0,
            'train/grad_clipped': True, 'train/lr': 1e-4,
            'train/entropy_coef': 0.02, 'train/advantages_mean': 0.4,
            'train/advantages_std': 0.5, 'train/value_mean': 0.6,
            'train/value_std': 0.7, 'train/reward_mean': 0.8,
            'train/reward_sum': 16.0, 'train/entropy': 2.1})
        self.assertEqual(project_update_to_wandb(record), {
            'global_step': 500,
            # schedules shared by all workers
            'train/lr': 1e-4, 'train/entropy_coef': 0.02,
            # this worker's rollout
            'worker_1/train/pi_loss': 0.1, 'worker_1/train/v_loss': 0.2,
            'worker_1/train/total_loss': 0.3,
            'worker_1/train/entropy': 2.1,
            'worker_1/train/advantages_mean': 0.4,
            'worker_1/train/advantages_std': 0.5,
            'worker_1/train/value_mean': 0.6,
            'worker_1/train/value_std': 0.7,
            'worker_1/train/reward_mean': 0.8,
        })

    def test_telemetry_loop_threshold_event_reaches_summary_and_file(self):
        run = FakeRun()
        with tempfile.TemporaryDirectory() as tmp:
            events = drain([episode_record(), threshold_event()], tmp, run)
        hit = next(e for e in events
                   if e['event'] == 'reward_threshold_reached')
        self.assertEqual(hit['signal'], 'reward_mean_100')
        self.assertEqual(hit['global_step'], 8000)
        self.assertEqual(hit['worker'], 1)  # the event keeps the worker
        prefix = 'threshold/reward_mean_100/25/'
        self.assertEqual(run.summary[prefix + 'reached'], True)
        self.assertEqual(run.summary[prefix + 'global_step'], 8000)
        self.assertEqual(run.summary[prefix + 'elapsed_training_s'], 1234.5)
        self.assertEqual(run.summary[prefix + 'global_episode'], 77)
        self.assertEqual(run.summary[prefix + 'value'], 26.5)
        self.assertNotIn(prefix + 'worker', run.summary)
        self.assertEqual(len(run.logged), 1)  # the episode metric point

    def test_telemetry_loop_minmax_summary_and_local_health(self):
        run = FakeRun()
        nan_event = build_record('event', worker_id=0, data={
            'event': 'nan_gradient', 'global_step': 50, 'nan_count': 1})
        counter = multiprocessing.Value('l', 3)
        with tempfile.TemporaryDirectory() as tmp:
            events = drain(
                [episode_record(), episode_record(global_step=3500),
                 episode_record(**{'global_step': 3600, 'reward_max': 12.0,
                                   'global/reward_max': 16.0}),
                 nan_event],
                tmp, run, shared_counters={'queue_drops': counter},
                final_summary={'final_global_step': 3600})
        minmax_calls = [call for call in run.summary.calls
                        if 'final_global_step' not in call]
        # the second record changed nothing, the third only two maxima
        self.assertEqual(len(minmax_calls), 2)
        self.assertEqual(minmax_calls[1], {'worker_2/reward_max': 12.0,
                                           'global/reward_max': 16.0})
        self.assertEqual(run.summary['global/reward_min'], -20.0)
        # health stays local: no W&B point, no W&B summary key
        self.assertEqual(len(run.logged), 3)  # the three episodes
        self.assertEqual(run.summary.calls[-1], {'final_global_step': 3600})
        final = events[-1]
        self.assertEqual(final['event'], 'final_summary')
        self.assertEqual(final['health/nan_updates_total'], 1)
        self.assertEqual(final['queue_drops'], 3)
        self.assertEqual(final['final_global_step'], 3600)

    def test_telemetry_loop_without_wandb_and_with_failing_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = drain([threshold_event()], tmp, None)
        self.assertEqual([e['event'] for e in events],
                         ['reward_threshold_reached', 'final_summary'])
        counter = multiprocessing.Value('l', 0)
        with tempfile.TemporaryDirectory() as tmp:
            events = drain(
                [threshold_event()], tmp, FakeRun(fail_summary=True),
                shared_counters={'wandb_errors': counter})
        self.assertIn(
            'reward_threshold_reached', [e['event'] for e in events])
        self.assertEqual(counter.value, 1)

    def test_tracker_stats_flow_into_record_and_payload(self):
        tracker = RewardTracker(2, thresholds=[3.0])
        tracker.set_clock(100.0, 10.0)
        with tempfile.TemporaryDirectory() as tmp:
            logger = TrainingLogger(tmp, worker_id=1)
            with at_time(130.0):
                stats, hits = tracker.record_episode(
                    1, 4.0, 90, 1, goal_dist=12.0)
            record = logger.log_episode(
                global_episode=1, global_t=90, total_reward=4.0, steps=20,
                goal_dist=12.0, **stats)
            interrupted = logger.log_episode(
                global_episode=2, global_t=95, total_reward=1.0, steps=5,
                goal_dist=30.0, interrupted=True)
            logger.close()
            path = os.path.join(tmp, 'logs', 'worker_1', 'episodes.jsonl')
            with open(path) as handle:
                lines = [json.loads(line) for line in handle]
        self.assertEqual([hit['signal'] for hit in hits],
                         ['worker_1/episode_reward'])
        self.assertEqual(lines[0]['global/elapsed_training_s'], 40.0)
        self.assertEqual(lines[0]['goal_distance_mean_100'], 12.0)
        # worker 0 has no episodes: its inf minimum is not written
        self.assertEqual(lines[0]['global/reward_min'], 4.0)
        self.assertTrue(lines[1]['interrupted'])
        payload = project_episode_to_wandb(record)
        self.assertEqual(payload['global/elapsed_training_s'], 40.0)
        self.assertEqual(payload['worker_1/goal_distance'], 12.0)
        self.assertEqual(payload['worker_1/goal_distance_mean_100'], 12.0)
        self.assertEqual(payload['global/reward_mean_100'], 4.0)
        self.assertEqual(
            project_episode_to_wandb_summary(record)['worker_1/reward_max'],
            4.0)
        self.assertEqual(project_episode_to_wandb(interrupted), {})

    def test_wandb_names_are_the_local_field_names(self):
        tracker = RewardTracker(2, thresholds=[1.0])
        stats, _ = tracker.record_episode(1, 4.0, 90, 1, goal_dist=12.0)
        with tempfile.TemporaryDirectory() as tmp:
            logger = TrainingLogger(tmp, worker_id=1)
            episode = logger.log_episode(
                global_episode=1, global_t=90, total_reward=4.0, steps=20,
                duration_s=3.0, max_speed_kmh=30.0, min_route_dist=0.5,
                goal_dist=12.0, collisions=0, reached_goal=True,
                action_counts=[5, 15], **stats)
            checks = [(episode, project_episode_to_wandb(episode)),
                      (episode, project_episode_to_wandb_summary(episode))]
            try:
                import numpy  # noqa: F401  (log_update needs it)
                update = logger.log_update(
                    update_count=1, global_t=95, trajectory_length=2,
                    is_terminal=False, pi_loss=0.1, v_loss=0.2,
                    total_loss=0.3, gradient_norm=1.0, lr=1e-4,
                    entropy_coef=0.02, advantages=[1.0, 2.0],
                    values=[0.5, 0.5], rewards=[1.0, 0.0],
                    entropies=[2.0, 2.0])
                checks.append((update, project_update_to_wandb(update)))
            except ImportError:
                pass
            logger.close()
        for record, payload in checks:
            self.assertTrue(payload)
            for key, value in payload.items():
                # worker_1/<field> is <field> in worker 1's file
                field = key.split('/', 1)[1] \
                    if key.startswith('worker_') else key
                self.assertEqual(float(record[field]), float(value), key)

    def test_steps_stay_local(self):
        telemetry_queue = queue.Queue()
        with tempfile.TemporaryDirectory() as tmp:
            logger = TrainingLogger(tmp, worker_id=3, log_steps=True,
                                    telemetry_queue=telemetry_queue,
                                    publish_metrics=True)
            logger.log_step(global_t=7, local_t=1, global_episode=1,
                            step_in_ep=1, action=0, value=0.0, entropy=1.0,
                            reward=0.25, done=False)
            logger.close()
            path = os.path.join(tmp, 'logs', 'worker_3', 'steps.jsonl')
            with open(path) as handle:
                self.assertEqual(json.loads(handle.readline())['reward'],
                                 0.25)
        # Steps go to steps.jsonl only, never to telemetry (W&B).
        self.assertTrue(telemetry_queue.empty())


if __name__ == '__main__':
    unittest.main()
