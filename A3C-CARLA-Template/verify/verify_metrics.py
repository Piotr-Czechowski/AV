"""Checks of the reward statistics, thresholds, and their logging.

Covers ``reward_stats`` (windows, best/worst, thresholds, checkpoint
round trip, clock, lock, concurrent processes) and the path to the outputs
(W&B projection, telemetry loop, episode records).  Needs no torch, no
CARLA, and no W&B.
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
    REWARD_WINDOW, RewardTracker, clean_thresholds, threshold_key,
    threshold_label, threshold_summary)
from training_logger import (  # noqa: E402
    TrainingLogger, build_record, make_telemetry_stop,
    project_episode_to_wandb, run_telemetry_loop)

INF = float('inf')
EPISODE_STAT_KEYS = {
    'is_new_best', 'local_mean_reward', 'local_reward_min_100',
    'local_reward_max_100', 'local_best_reward', 'local_worst_reward',
    'global_mean_reward_100', 'global_reward_min_100',
    'global_reward_max_100', 'global_mean_reward', 'global_reward_min',
    'global_reward_max', 'best_reward', 'worst_reward',
    'elapsed_training_s'}


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
    """Dict with ``update`` like ``wandb_run.summary``."""

    def __init__(self, fail=False):
        super().__init__()
        self.fail = fail

    def update(self, items):
        if self.fail:
            raise RuntimeError('summary down')
        super().update(items)


class FakeRun:
    """Stands in for a W&B run: ``log()`` and ``summary.update()``."""

    def __init__(self, fail_summary=False):
        self.logged = []
        self.summary = FakeSummary(fail=fail_summary)

    def log(self, payload):
        self.logged.append(payload)


def drain(records, run_dir, wandb_run, shared_counters=None):
    """Run the telemetry loop over ``records``; return events.jsonl."""
    telemetry_queue = queue.Queue()
    for record in records:
        telemetry_queue.put(record)
    telemetry_queue.put(make_telemetry_stop({'final': 1}))
    with contextlib.redirect_stdout(io.StringIO()):
        run_telemetry_loop(
            telemetry_queue, run_dir, wandb_run=wandb_run,
            shared_counters=shared_counters)
    with open(os.path.join(run_dir, 'logs', 'events.jsonl')) as handle:
        return [json.loads(line) for line in handle]


def threshold_event():
    return build_record('event', worker_id=1, data={
        'event': 'reward_threshold_reached', 'signal': 'reward_mean_100',
        'threshold': 25.0, 'value': 26.5, 'global_t': 8000,
        'global_episode': 77, 'elapsed_training_s': 1234.5})


def episode_record(**fields):
    data = {
        'global_episode': 12, 'global_t': 3400, 'total_reward': 7.5,
        'steps': 150, 'mean_reward': 0.05, 'duration_s': 4.5,
        'reached_goal': True,
        'max_speed_kmh': 31.0, 'min_route_dist': 0.5, 'goal_dist': 42.0,
        'collisions': 2, 'action_counts': [3, 1],
        'reward_components': {'progress': 2.0, 'total': 7.5},
        'local_mean_reward': 5.0, 'local_reward_min_100': -3.0,
        'local_reward_max_100': 9.0, 'local_best_reward': 11.0,
        'local_worst_reward': -8.0,
        'global_mean_reward_100': 4.0, 'global_reward_min_100': -4.0,
        'global_reward_max_100': 10.0, 'global_mean_reward': 3.5,
        'global_reward_min': -6.0, 'global_reward_max': 12.0,
        'best_reward': 15.0, 'worst_reward': -20.0,
        'elapsed_training_s': 987.5,
    }
    data.update(fields)
    return build_record('episode', worker_id=2, data=data)


def _record_in_process(tracker, worker_id, count):
    for index in range(count):
        tracker.record_episode(
            worker_id, worker_id * 1000 + index,
            global_t=index, global_episode=worker_id * 1000 + index)


# -- windows and best/worst ------------------------------------------------

class WindowTests(unittest.TestCase):
    @small_window(3)
    def test_window_scopes_wrap_and_independence(self):
        tracker = RewardTracker(2)
        record_many(tracker, 0, [1, 2, 3, 4])
        stats, hits = tracker.record_episode(1, 10.0, 50, 5)
        self.assertEqual(set(stats), EPISODE_STAT_KEYS)
        self.assertEqual(hits, [])
        # worker 1 has one episode
        self.assertEqual(stats['local_mean_reward'], 10.0)
        self.assertEqual(stats['local_reward_min_100'], 10.0)
        self.assertEqual(stats['local_reward_max_100'], 10.0)
        # run window: last 3 of 1,2,3,4,10
        self.assertEqual(stats['global_mean_reward_100'], 17 / 3)
        self.assertEqual(stats['global_reward_min_100'], 3.0)
        self.assertEqual(stats['global_reward_max_100'], 10.0)
        # long window holds 3 x 2 = 6 episodes, so all five
        self.assertEqual(stats['global_mean_reward'], 4.0)
        self.assertEqual(stats['global_reward_min'], 1.0)
        self.assertEqual(stats['global_reward_max'], 10.0)
        state = tracker.state_dict()
        self.assertEqual(state['workers'], [[2.0, 3.0, 4.0], [10.0]])

        # wrap: five more episodes of worker 1, worker 0 stays untouched
        record_many(tracker, 1, [11, 12, 13, 14, 15], start_episode=6)
        state = tracker.state_dict()
        self.assertEqual(state['workers'], [[2.0, 3.0, 4.0],
                                            [13.0, 14.0, 15.0]])
        self.assertEqual(state['global'], [13.0, 14.0, 15.0])
        self.assertEqual(
            state['long'], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0])

    @small_window(2)
    def test_best_and_worst_outlive_the_windows(self):
        tracker = RewardTracker(2)
        stats, _ = tracker.record_episode(0, 5.0, 1, 1)
        self.assertTrue(stats['is_new_best'])
        stats, _ = tracker.record_episode(1, -3.0, 2, 2)
        self.assertFalse(stats['is_new_best'])
        self.assertEqual(stats['local_best_reward'], -3.0)
        self.assertEqual(stats['local_worst_reward'], -3.0)
        record_many(tracker, 0, [1, 1, 1], start_episode=3)
        stats, _ = tracker.record_episode(0, 6.0, 10, 6)
        self.assertTrue(stats['is_new_best'])
        self.assertEqual(stats['best_reward'], 6.0)
        self.assertEqual(stats['worst_reward'], -3.0)
        self.assertEqual(stats['local_best_reward'], 6.0)
        self.assertEqual(stats['local_worst_reward'], 1.0)
        self.assertEqual(stats['global_reward_min_100'], 1.0)
        # an equal reward is not a new best
        stats, _ = tracker.record_episode(0, 6.0, 11, 7)
        self.assertFalse(stats['is_new_best'])


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
                threshold_key('reward_mean', threshold, 'global_step'),
                pattern)

    def test_episode_reward_fires_once(self):
        tracker = RewardTracker(1, thresholds=[10.0])
        hits = record_many(tracker, 0, [3, 10, 50, 12])
        self.assertEqual(hits, [{
            'signal': 'episode_reward', 'threshold': 10.0, 'value': 10.0,
            'global_t': 20, 'global_episode': 2,
            'elapsed_training_s': 0.0, 'worker': 0}])
        self.assertEqual(hits_of(tracker), hits)

    @small_window(3)
    def test_mean_signals_wait_for_a_full_window_and_fire_once(self):
        tracker = RewardTracker(2, thresholds=[5.0])
        # the mean of [10, 0] is 5.0, but the window of 3 is not full yet
        hits = record_many(tracker, 0, [10, 0])
        self.assertEqual(signals(hits), [('episode_reward', 1)])
        hits = record_many(tracker, 0, [10], start_episode=3)
        self.assertEqual(signals(hits), [('reward_mean_100', 3)])
        # the long window (3 x 2 episodes) is full after episode 6
        self.assertEqual(record_many(tracker, 1, [5, 5], start_episode=4), [])
        hits = record_many(tracker, 1, [5], start_episode=6)
        self.assertEqual(signals(hits), [('reward_mean', 6)])
        self.assertEqual(record_many(tracker, 1, [9], start_episode=7), [])

    @small_window(2)
    def test_summary_reached_and_unreached(self):
        tracker = RewardTracker(1, thresholds=[1.0, 100.0])
        tracker.set_clock(1000.0, 5.0)
        with at_time(1010.0):
            tracker.record_episode(0, 4.0, 30, 1)
        with at_time(1020.0):
            tracker.record_episode(0, 2.0, 40, 2)
        summary = tracker.summary()
        self.assertEqual(
            {key: summary[key] for key in (
                'best_reward', 'worst_reward', 'final_global_mean_reward',
                'final_reward_min', 'final_reward_max',
                'final_reward_mean_100', 'final_reward_min_100',
                'final_reward_max_100')},
            {'best_reward': 4.0, 'worst_reward': 2.0,
             'final_global_mean_reward': 3.0, 'final_reward_min': 2.0,
             'final_reward_max': 4.0, 'final_reward_mean_100': 3.0,
             'final_reward_min_100': 2.0, 'final_reward_max_100': 4.0})
        prefix = 'threshold/episode_reward/1/'
        self.assertIs(summary[prefix + 'reached'], True)
        self.assertEqual(summary[prefix + 'global_step'], 30)
        self.assertEqual(summary[prefix + 'elapsed_training_s'], 15.0)
        self.assertEqual(summary[prefix + 'global_episode'], 1)
        self.assertEqual(summary[prefix + 'value'], 4.0)
        self.assertEqual(summary[prefix + 'worker'], 0)
        # a pair that was never reached has None fields
        self.assertIs(summary['threshold/episode_reward/100/reached'], False)
        for name in ('global_step', 'elapsed_training_s', 'global_episode',
                     'value', 'worker'):
            self.assertIsNone(summary['threshold/episode_reward/100/' + name])
        # the same shapes come from threshold_summary directly
        self.assertEqual(
            threshold_summary('episode_reward', 100.0),
            {key: value for key, value in summary.items()
             if key.startswith('threshold/episode_reward/100/')})
        # nothing recorded yet: no best, no mean
        empty = RewardTracker(2, thresholds=[3.0]).summary()
        self.assertIsNone(empty['best_reward'])
        self.assertIsNone(empty['final_global_mean_reward'])
        self.assertIs(empty['threshold/reward_mean/3/reached'], False)


# -- checkpoint, legacy, lock, processes -----------------------------------

class CheckpointTests(unittest.TestCase):
    @small_window(3)
    def test_round_trip_keeps_state_hits_and_elapsed(self):
        tracker = RewardTracker(2, thresholds=[1.0])
        with at_time(500.0):  # before set_clock: no offset, no time
            self.assertEqual(tracker.elapsed_training_s(), 0.0)
        tracker.set_clock(1000.0, 100.0)
        with at_time(1010.0):
            stats, _ = tracker.record_episode(0, 1.0, 1, 1)
        self.assertEqual(stats['elapsed_training_s'], 110.0)
        tracker = RewardTracker(2, thresholds=[1.0])
        with at_time(1050.0):
            tracker.set_clock(1000.0, 100.0)
            record_many(tracker, 0, [1, 2, 3, 4])
            record_many(tracker, 1, [9, -9], start_episode=5)
            state = tracker.state_dict()
        self.assertEqual(state['elapsed_training_s'], 150.0)
        restored = RewardTracker(2, thresholds=[1.0])
        elapsed = restored.load_state_dict(state)
        self.assertEqual(elapsed, 150.0)
        # the clock of the new session is not set yet, so compare the rest
        self.assertEqual(
            dict(restored.state_dict(), elapsed_training_s=elapsed), state)
        # reached thresholds do not fire again
        self.assertTrue(state['hits'])
        self.assertEqual(record_many(restored, 0, [50, 50, 50], 7), [])
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
        # the long window of 4 keeps the most recent four of six
        self.assertEqual(state['long'], [2.0, 2.0, 3.0, 3.0])
        self.assertEqual(state['workers'], [[1.0, 1.0], [2.0, 2.0]])
        self.assertEqual(state['worker_best'], [1.0, 2.0])
        self.assertEqual(state['best_reward'], 3.0)

        single = RewardTracker(1)
        record_many(single, 0, [1, 2, 3])
        big = RewardTracker(3)
        big.load_state_dict(single.state_dict())
        state = big.state_dict()
        self.assertEqual(state['long'], [2.0, 3.0])
        self.assertEqual(state['workers'], [[2.0, 3.0], [], []])
        self.assertEqual(state['worker_best'][1:], [-INF] * 2)
        stats, _ = big.record_episode(2, 5.0, 9, 4)
        self.assertEqual(stats['local_mean_reward'], 5.0)
        self.assertEqual(stats['global_mean_reward'], 10 / 3)

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
        # a snapshot with the dropped keys still loads; without elapsed
        # time it reports None
        state.update(window=100, num_workers=1)
        del state['elapsed_training_s']
        self.assertIsNone(RewardTracker(1).load_state_dict(state))

    @small_window(3)
    def test_legacy_migration(self):
        tracker = RewardTracker(2)
        # not wrapped: 4 of 6 slots used
        self.assertIsNone(tracker.load_legacy({
            'recent_rewards': [1.0, 2.0, 3.0, 4.0, 0.0, 0.0],
            'recent_reward_count': 4, 'recent_reward_index': 4,
            'best_reward': 4.0}))
        state = tracker.state_dict()
        self.assertEqual(state['long'], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(state['global'], [2.0, 3.0, 4.0])
        self.assertEqual(state['best_reward'], 4.0)
        self.assertEqual(state['worst_reward'], 1.0)  # window minimum
        self.assertEqual(state['workers'], [[], []])
        self.assertEqual(state['worker_worst'], [INF, INF])
        # wrapped: a buffer of 4 that took 6 episodes holds 5 6 3 4
        wrapped = RewardTracker(2)
        wrapped.load_legacy({
            'recent_rewards': [5.0, 6.0, 3.0, 4.0],
            'recent_reward_count': 4, 'recent_reward_index': 6,
            'best_reward': 6.0})
        state = wrapped.state_dict()
        self.assertEqual(state['long'], [3.0, 4.0, 5.0, 6.0])
        self.assertEqual(state['global'], [4.0, 5.0, 6.0])
        self.assertEqual(state['worst_reward'], 3.0)
        # empty
        empty = RewardTracker(1)
        self.assertIsNone(empty.load_legacy({}))
        self.assertEqual(empty.state_dict()['long'], [])

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
        expected = sorted(float(worker_id * 1000 + index)
                          for worker_id in range(workers)
                          for index in range(count))
        # 120 episodes fit in the long window of 300
        self.assertEqual(sorted(state['long']), expected)
        self.assertEqual(len(state['global']), REWARD_WINDOW)
        self.assertEqual(state['global'], state['long'][-REWARD_WINDOW:])
        for worker_id in range(workers):
            self.assertEqual(
                state['workers'][worker_id],
                [float(worker_id * 1000 + i) for i in range(count)])
        self.assertEqual(state['best_reward'], 2039.0)
        self.assertEqual(state['worst_reward'], 0.0)
        # only worker 2 reaches 1500, and the threshold fires once
        episode_hits = [h for h in state['hits']
                        if h['signal'] == 'episode_reward']
        self.assertEqual(len(episode_hits), 1)
        self.assertEqual(episode_hits[0]['worker'], 2)


# -- logging ---------------------------------------------------------------

class LoggingTests(unittest.TestCase):
    def test_wandb_episode_projection(self):
        payload = project_episode_to_wandb(episode_record())
        expected = {
            # pre-existing keys
            'global_step': 3400, 'episode/id': 12, 'episode/reward': 7.5,
            'episode/reward_per_step': 0.05, 'episode/length': 150,
            'episode/duration_s': 4.5, 'episode/success': 1,
            'episode/max_speed_kmh': 31.0,
            'episode/min_route_distance': 0.5,
            'episode/goal_distance': 42.0, 'episode/collisions': 2,
            'episode/action_0_fraction': 0.75,
            'episode/action_1_fraction': 0.25,
            'episode/reward_progress': 2.0,
            'global/reward_mean': 3.5, 'global/best_reward': 15.0,
            'worker_2/reward': 7.5, 'worker_2/episode_length': 150,
            'worker_2/reward_mean_100': 5.0,
            # new keys
            'global/reward_mean_100': 4.0, 'global/reward_min_100': -4.0,
            'global/reward_max_100': 10.0, 'global/reward_min': -6.0,
            'global/reward_max': 12.0, 'global/worst_reward': -20.0,
            'global/elapsed_training_s': 987.5,
            'worker_2/reward_min_100': -3.0,
            'worker_2/reward_max_100': 9.0, 'worker_2/best_reward': 11.0,
            'worker_2/worst_reward': -8.0, 'worker_2/goal_distance': 42.0,
        }
        self.assertEqual(payload, expected)
        # the total of the reward components is not sent twice
        self.assertNotIn('episode/reward_total', payload)
        # an interrupted episode stays local
        self.assertEqual(
            project_episode_to_wandb(episode_record(interrupted=True)), {})
        # a run-level record has no worker keys
        run_level = build_record('episode', worker_id=-1, data={
            'global_t': 1, 'goal_dist': 3.0, 'local_best_reward': 1.0})
        self.assertFalse([key for key in project_episode_to_wandb(run_level)
                          if key.startswith('worker_')])

    def test_telemetry_loop_threshold_event_reaches_summary_and_file(self):
        run = FakeRun()
        with tempfile.TemporaryDirectory() as tmp:
            events = drain([episode_record(), threshold_event()], tmp, run)
        hit = next(e for e in events
                   if e['event'] == 'reward_threshold_reached')
        self.assertEqual(hit['signal'], 'reward_mean_100')
        self.assertEqual(hit['global_t'], 8000)
        self.assertEqual(hit['worker'], 1)
        prefix = 'threshold/reward_mean_100/25/'
        self.assertEqual(run.summary[prefix + 'reached'], True)
        self.assertEqual(run.summary[prefix + 'global_step'], 8000)
        self.assertEqual(run.summary[prefix + 'elapsed_training_s'], 1234.5)
        self.assertEqual(run.summary[prefix + 'global_episode'], 77)
        self.assertEqual(run.summary[prefix + 'value'], 26.5)
        self.assertEqual(run.summary[prefix + 'worker'], 1)
        self.assertEqual(len(run.logged), 1)  # the episode metric point

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
                stats, hits = tracker.record_episode(1, 4.0, 90, 1)
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
        self.assertEqual(len(hits), 1)
        self.assertEqual(lines[0]['elapsed_training_s'], 40.0)
        self.assertTrue(lines[1]['interrupted'])
        payload = project_episode_to_wandb(record)
        self.assertEqual(payload['global/elapsed_training_s'], 40.0)
        self.assertEqual(payload['worker_1/goal_distance'], 12.0)
        self.assertEqual(payload['global/reward_max'], 4.0)
        self.assertEqual(project_episode_to_wandb(interrupted), {})


if __name__ == '__main__':
    unittest.main()
