"""Reward statistics shared by all workers (stdlib only).

One ``RewardTracker`` lives in shared memory next to the global network.
Each worker calls ``record_episode`` once per completed episode and gets
back the statistics at that moment, plus the reward thresholds this episode
was the first to reach.  Per worker it keeps ring buffers of the episode
reward and the final goal distance of the last ``REWARD_WINDOW`` episodes,
and the run minimum and maximum of both.  Run-wide values are aggregates of
the workers: the mean of their means, the lowest minimum, the highest
maximum.

This module imports nothing from the project, so the torch-free
``training_logger`` can reuse ``threshold_summary`` and ``minmax_summary``.
"""

import contextlib
import math
import multiprocessing
import time

REWARD_WINDOW = 100

# What a threshold is compared against.  Run-wide: the global mean reward,
# once every worker's window is full.  Per worker, named
# ``worker_<id>/<signal>``: the reward of the episode that just ended, and the
# mean of the worker's window once it is full.
THRESHOLD_SIGNALS = ('reward_mean_100',)
WORKER_THRESHOLD_SIGNALS = ('episode_reward', 'reward_mean_100')

# state_dict() and summary() wait this long for the tracker lock, then read
# without it (see RewardTracker._read_lock).
LOCK_TIMEOUT_S = 5.0

# Per threshold slot, in the order they are stored in shared memory.
_SLOT_FIELDS = ('global_step', 'global_episode', 'elapsed_training_s',
                'value', 'worker')
_NO_HIT = (-1.0, -1.0, 0.0, 0.0, -1.0)
# Fields of a threshold in the run summary; ``worker`` stays in the event.
_SUMMARY_FIELDS = ('global_step', 'elapsed_training_s', 'global_episode',
                   'value')

# Run minimum and maximum: run-summary values ``worker_<id>/<name>`` (the
# episode field ``<name>``) and ``global/<name>``.
MINMAX_NAMES = ('reward_min', 'reward_max', 'goal_distance_min',
                'goal_distance_max')


def threshold_label(threshold):
    """Threshold spelled with ``[A-Za-z0-9_]`` only, for use in a key.

    W&B reads a dot in a metric name as nesting, so ``{:g}`` is encoded:
    ``-`` -> ``neg``, ``.`` -> ``p``, ``+`` dropped.  0.5 -> ``0p5``,
    -50 -> ``neg50``, 1e+06 -> ``1e06``, -2.5e-05 -> ``neg2p5eneg05``.
    """
    return '{:g}'.format(threshold).replace('-', 'neg').replace(
        '.', 'p').replace('+', '')


def threshold_key(signal, threshold, field):
    """Name of one threshold entry in the W&B summary and ``final_summary``."""
    return 'threshold/{}/{}/{}'.format(
        signal, threshold_label(threshold), field)


def threshold_summary(signal, threshold, hit=None):
    """Summary entries of one (signal, threshold) pair.

    ``hit`` is the dict of a ``reward_threshold_reached`` event.  Without it
    the pair is ``reached = False`` and its other fields are ``None``, which
    also overwrites stale values in a resumed W&B run.
    """
    threshold = float(threshold)
    items = {threshold_key(signal, threshold, 'reached'): hit is not None}
    for field in _SUMMARY_FIELDS:
        items[threshold_key(signal, threshold, field)] = \
            hit.get(field) if hit is not None else None
    return items


def minmax_summary(stats, worker_id=None):
    """W&B run-summary entries of the run minimum and maximum in ``stats``.

    ``stats`` holds the fields of an episode record; missing or ``None``
    values are left out.
    """
    items = {}
    for name in MINMAX_NAMES:
        if worker_id is not None:
            items['worker_{}/{}'.format(worker_id, name)] = stats.get(name)
        items['global/' + name] = stats.get('global/' + name)
    return {key: value for key, value in items.items() if value is not None}


def _finite(value):
    """``value``, or ``None`` for the +-inf of an empty minimum or maximum."""
    return value if math.isfinite(value) else None


def _mean(values):
    """Mean of the values that are not ``None``; ``None`` if there are none."""
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _widen(low, high, index, value):
    """Extend the minimum ``low[index]`` and the maximum ``high[index]``."""
    low[index] = min(low[index], value)
    high[index] = max(high[index], value)


def _item(items, index, default):
    """``items[index]``, or ``default`` when ``items`` is shorter."""
    return items[index] if index < len(items) else default


def clean_thresholds(thresholds):
    """Deduplicate and sort thresholds as floats.

    Raises ``ValueError`` for a non-finite value, and for two values with
    the same key label (their summary keys would collide).
    """
    cleaned = set()
    for threshold in thresholds or ():
        threshold = float(threshold)
        if not math.isfinite(threshold):
            raise ValueError(
                'reward threshold must be finite, got {}'.format(threshold))
        cleaned.add(threshold + 0.0)  # -0.0 -> 0.0
    ordered = sorted(cleaned)
    labels = [threshold_label(threshold) for threshold in ordered]
    if len(set(labels)) != len(labels):
        raise ValueError(
            'reward thresholds {} are not distinct: only 6 significant '
            'digits are kept in the summary keys'.format(ordered))
    return ordered


class SharedRewardWindow:
    """Ring buffer of the last ``size`` floats in shared memory.

    Not locked on its own: the caller holds the ``RewardTracker`` lock.
    """

    def __init__(self, size):
        if int(size) <= 0:
            raise ValueError('window size must be positive')
        self.size = int(size)
        self._buffer = multiprocessing.Array('d', self.size, lock=False)
        self._index = multiprocessing.Value('l', 0, lock=False)
        self._count = multiprocessing.Value('l', 0, lock=False)

    @property
    def full(self):
        return self._count.value == self.size

    def push(self, value):
        """Append ``value``, overwriting the oldest one when full."""
        self._buffer[self._index.value] = float(value)
        self._index.value = (self._index.value + 1) % self.size
        self._count.value = min(self.size, self._count.value + 1)

    def values(self):
        """Stored values, oldest to newest."""
        count = self._count.value
        if count < self.size:
            return list(self._buffer[:count])
        index = self._index.value
        return list(self._buffer[index:]) + list(self._buffer[:index])

    def mean(self):
        """Mean of the stored values; ``None`` when empty."""
        return _mean(self.values())

    def load(self, values):
        """Replace the content; only the last ``size`` values are kept."""
        self._index.value = 0
        self._count.value = 0
        for value in list(values)[-self.size:]:
            self.push(value)


class RewardTracker:
    """Per-worker reward windows, run minimum and maximum, and thresholds.

    One lock guards everything.
    """

    def __init__(self, num_workers, thresholds=()):
        self.num_workers = int(num_workers)
        self.thresholds = clean_thresholds(thresholds)
        self.lock = multiprocessing.Lock()

        self._workers = [SharedRewardWindow(REWARD_WINDOW)
                         for _ in range(self.num_workers)]
        self._worker_goal_dists = [SharedRewardWindow(REWARD_WINDOW)
                                   for _ in range(self.num_workers)]
        # Run minimum and maximum of each worker; +-inf before its first value.
        inf = float('inf')
        self._reward_min = multiprocessing.Array(
            'd', [inf] * self.num_workers, lock=False)
        self._reward_max = multiprocessing.Array(
            'd', [-inf] * self.num_workers, lock=False)
        self._goal_dist_min = multiprocessing.Array(
            'd', [inf] * self.num_workers, lock=False)
        self._goal_dist_max = multiprocessing.Array(
            'd', [-inf] * self.num_workers, lock=False)

        # One slot per (signal, threshold). A slot holds _SLOT_FIELDS as
        # doubles, which store these integers exactly; global_step < 0 means
        # the threshold was not reached.
        self._pairs = [(signal, threshold) for signal in THRESHOLD_SIGNALS
                       for threshold in self.thresholds]
        self._pairs += [('worker_{}/{}'.format(worker_id, signal), threshold)
                        for worker_id in range(self.num_workers)
                        for signal in WORKER_THRESHOLD_SIGNALS
                        for threshold in self.thresholds]
        self._slot_of = {pair: slot for slot, pair in enumerate(self._pairs)}
        self._hits = multiprocessing.Array(
            'd', list(_NO_HIT) * len(self._pairs), lock=False)

        # Session clock; a start time of 0.0 means it was never set.
        self._clock_start = multiprocessing.Value('d', 0.0, lock=False)
        self._clock_offset = multiprocessing.Value('d', 0.0, lock=False)

    @contextlib.contextmanager
    def _read_lock(self):
        """Hold the lock for a read, but wait for it at most LOCK_TIMEOUT_S.

        A worker killed while it holds the lock would otherwise block the
        final checkpoint and summary.  On timeout the snapshot is read
        without the lock and may be slightly inconsistent.
        """
        acquired = self.lock.acquire(timeout=LOCK_TIMEOUT_S)
        if not acquired:
            print('[REWARD_STATS] lock busy; reading without it', flush=True)
        try:
            yield
        finally:
            if acquired:
                self.lock.release()

    # -- clock --

    def set_clock(self, session_start_time, elapsed_offset_s=0.0):
        """Start the session clock (a ``time.time()`` value).

        ``elapsed_offset_s`` is the active training time of earlier
        sessions of a resumed run.
        """
        with self.lock:
            self._clock_start.value = float(session_start_time)
            self._clock_offset.value = float(elapsed_offset_s)

    def elapsed_training_s(self):
        """Active training time, cumulative over resumed sessions.

        Before ``set_clock`` this is just the offset (zero on a new run).
        """
        if self._clock_start.value == 0.0:
            return self._clock_offset.value
        return self._clock_offset.value + (
            time.time() - self._clock_start.value)

    # -- threshold slots --

    @staticmethod
    def _slot_range(slot):
        return slice(slot * len(_SLOT_FIELDS), (slot + 1) * len(_SLOT_FIELDS))

    def _read_hit(self, slot):
        """Hit dict of ``slot``, or None if its threshold is not reached."""
        stored = dict(zip(_SLOT_FIELDS, self._hits[self._slot_range(slot)]))
        if stored['global_step'] < 0:
            return None
        signal, threshold = self._pairs[slot]
        return {
            'signal': signal,
            'threshold': threshold,
            'value': stored['value'],
            'global_step': int(stored['global_step']),
            'global_episode': int(stored['global_episode']),
            'elapsed_training_s': stored['elapsed_training_s'],
            'worker': int(stored['worker']),
        }

    def _write_hit(self, slot, hit):
        """Store a hit dict; ``None`` marks the slot as not reached."""
        values = _NO_HIT if hit is None else [
            float(hit[field]) for field in _SLOT_FIELDS]
        self._hits[self._slot_range(slot)] = values

    def _all_hits(self):
        """Reached thresholds, earliest first. The caller holds the lock."""
        hits = [self._read_hit(slot) for slot in range(len(self._pairs))]
        hits = [hit for hit in hits if hit is not None]
        hits.sort(
            key=lambda hit: (hit['global_step'], hit['global_episode']))
        return hits

    # -- recording --

    def _stats(self, worker_id):
        """Statistics of one worker (W&B names) and the run (``global/*``).

        A run-wide mean is the mean of the worker means, without workers that
        have none; a run-wide minimum or maximum is the lowest or highest of
        the workers.  The caller holds the lock.
        """
        reward_means = [window.mean() for window in self._workers]
        dist_means = [window.mean() for window in self._worker_goal_dists]
        return {
            'reward_mean_100': reward_means[worker_id],
            'goal_distance_mean_100': dist_means[worker_id],
            'reward_min': _finite(self._reward_min[worker_id]),
            'reward_max': _finite(self._reward_max[worker_id]),
            'goal_distance_min': _finite(self._goal_dist_min[worker_id]),
            'goal_distance_max': _finite(self._goal_dist_max[worker_id]),
            'global/reward_mean_100': _mean(reward_means),
            'global/goal_distance_mean_100': _mean(dist_means),
            'global/reward_min': _finite(min(self._reward_min)),
            'global/reward_max': _finite(max(self._reward_max)),
            'global/goal_distance_min': _finite(min(self._goal_dist_min)),
            'global/goal_distance_max': _finite(max(self._goal_dist_max)),
        }

    def record_episode(self, worker_id, reward, global_t, global_episode,
                       goal_dist=None):
        """Count one completed episode.

        ``goal_dist`` is the final goal distance; None or inf is skipped.

        Returns ``(stats, new_hits)``.  ``stats`` is a flat dict whose keys
        become fields of the episode record; ``new_hits`` lists the
        thresholds this episode was the first to reach.
        """
        reward = float(reward)
        with self.lock:
            # Higher than every earlier episode of the run.
            is_new_best = reward > max(self._reward_max)
            self._workers[worker_id].push(reward)
            _widen(self._reward_min, self._reward_max, worker_id, reward)
            if goal_dist is not None and math.isfinite(goal_dist):
                goal_dist = float(goal_dist)
                self._worker_goal_dists[worker_id].push(goal_dist)
                _widen(self._goal_dist_min, self._goal_dist_max, worker_id,
                       goal_dist)

            elapsed = self.elapsed_training_s()
            stats = self._stats(worker_id)
            stats['global/elapsed_training_s'] = elapsed
            stats['is_new_best'] = is_new_best

            all_full = all(window.full for window in self._workers)
            signal_values = {
                'reward_mean_100':
                    stats['global/reward_mean_100'] if all_full else None,
                'worker_{}/episode_reward'.format(worker_id): reward,
                'worker_{}/reward_mean_100'.format(worker_id):
                    stats['reward_mean_100']
                    if self._workers[worker_id].full else None,
            }
            new_hits = []
            for signal, value in signal_values.items():
                if value is None:
                    continue
                for threshold in self.thresholds:
                    slot = self._slot_of[(signal, threshold)]
                    if value < threshold or self._read_hit(slot) is not None:
                        continue
                    self._write_hit(slot, {
                        'global_step': global_t,
                        'global_episode': global_episode,
                        'elapsed_training_s': elapsed, 'value': value,
                        'worker': worker_id})
                    new_hits.append(self._read_hit(slot))
        return stats, new_hits

    # -- summary --

    def summary(self):
        """Flat entries for ``final_summary`` and the W&B run summary.

        The run minimum and maximum (see ``minmax_summary``) and every
        configured threshold.
        """
        summary = {}
        with self._read_lock():
            for worker_id in range(self.num_workers):
                summary.update(
                    minmax_summary(self._stats(worker_id), worker_id))
            hits = [self._read_hit(slot) for slot in range(len(self._pairs))]
        for (signal, threshold), hit in zip(self._pairs, hits):
            summary.update(threshold_summary(signal, threshold, hit))
        return summary

    # -- checkpoint --

    def state_dict(self):
        """Plain-Python snapshot for the checkpoint.

        ``elapsed_training_s`` is taken at the same moment as the rest.
        """
        with self._read_lock():
            return {
                'workers': [window.values() for window in self._workers],
                'worker_goal_dists': [
                    window.values() for window in self._worker_goal_dists],
                'worker_reward_min': list(self._reward_min),
                'worker_reward_max': list(self._reward_max),
                'worker_goal_dist_min': list(self._goal_dist_min),
                'worker_goal_dist_max': list(self._goal_dist_max),
                'hits': self._all_hits(),
                'elapsed_training_s': self.elapsed_training_s(),
            }

    def load_state_dict(self, state):
        """Restore a ``state_dict`` snapshot; return its training time.

        The return value is ``elapsed_training_s`` of the snapshot, or
        ``None`` if it has none.  A different number of workers is
        tolerated: worker lists are zipped, and workers without saved data
        start empty.  A snapshot from before the run minimum and maximum
        takes the reward ones from ``worker_worst`` / ``worker_best`` and the
        goal-distance ones from the saved window.  Only thresholds configured
        now are restored; the others stay in the old ``events.jsonl``.
        """
        inf = float('inf')
        saved_rewards = list(state.get('workers', ()))
        saved_dists = list(state.get('worker_goal_dists', ()))
        reward_min = list(state.get('worker_reward_min',
                                    state.get('worker_worst', ())))
        reward_max = list(state.get('worker_reward_max',
                                    state.get('worker_best', ())))
        dist_min = list(state.get('worker_goal_dist_min', ()))
        dist_max = list(state.get('worker_goal_dist_max', ()))
        with self.lock:
            for index in range(self.num_workers):
                self._workers[index].load(_item(saved_rewards, index, ()))
                self._worker_goal_dists[index].load(
                    _item(saved_dists, index, ()))
                rewards = self._workers[index].values()
                dists = self._worker_goal_dists[index].values()
                self._reward_min[index] = _item(
                    reward_min, index, min(rewards, default=inf))
                self._reward_max[index] = _item(
                    reward_max, index, max(rewards, default=-inf))
                self._goal_dist_min[index] = _item(
                    dist_min, index, min(dists, default=inf))
                self._goal_dist_max[index] = _item(
                    dist_max, index, max(dists, default=-inf))
            for slot in range(len(self._pairs)):
                self._write_hit(slot, None)
            for hit in state.get('hits', ()):
                # An older snapshot names the step ``global_t``.
                hit = dict(hit, global_step=hit.get(
                    'global_step', hit.get('global_t')))
                try:
                    slot = self._slot_of[
                        (hit['signal'], float(hit['threshold']))]
                    self._write_hit(slot, hit)
                except (KeyError, TypeError, ValueError):
                    continue
        elapsed = state.get('elapsed_training_s')
        return None if elapsed is None else float(elapsed)
