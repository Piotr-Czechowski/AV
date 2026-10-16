"""Run-wide reward statistics shared by all workers (stdlib only).

One ``RewardTracker`` lives in shared memory next to the global network.
Each worker calls ``record_episode`` once per completed episode and gets
back the rolling statistics at that moment, plus the reward thresholds this
episode was the first to reach.  Windows are ring buffers of episode
rewards: the last ``REWARD_WINDOW`` episodes of all workers, the last
``REWARD_WINDOW * num_workers`` episodes of all workers, and the last
``REWARD_WINDOW`` episodes of each worker.

This module imports nothing from the project, so the torch-free
``training_logger`` can reuse ``threshold_summary``.
"""

import contextlib
import math
import multiprocessing
import time

REWARD_WINDOW = 100

# What a threshold is compared against: the reward of the episode that just
# ended, the run-wide mean of the last REWARD_WINDOW episodes, or the mean of
# the last REWARD_WINDOW * num_workers episodes.  The two means count only
# once their window is full.
THRESHOLD_SIGNALS = ('episode_reward', 'reward_mean_100', 'reward_mean')

# state_dict() and summary() wait this long for the tracker lock, then read
# without it (see RewardTracker._read_lock).
LOCK_TIMEOUT_S = 5.0

# Per threshold slot, in the order they are stored in shared memory.
_SLOT_FIELDS = ('global_t', 'global_episode', 'elapsed_training_s',
                'value', 'worker')
_NO_HIT = (-1.0, -1.0, 0.0, 0.0, -1.0)
# Summary name of each slot field.
_SUMMARY_NAMES = (('global_t', 'global_step'),
                  ('elapsed_training_s', 'elapsed_training_s'),
                  ('global_episode', 'global_episode'),
                  ('value', 'value'),
                  ('worker', 'worker'))


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
    for field, name in _SUMMARY_NAMES:
        items[threshold_key(signal, threshold, name)] = \
            hit.get(field) if hit is not None else None
    return items


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

    def stats(self):
        """``mean``, ``min``, ``max``; ``None`` when empty."""
        values = self.values()
        if not values:
            return {'mean': None, 'min': None, 'max': None}
        return {'mean': sum(values) / len(values),
                'min': min(values), 'max': max(values)}

    def load(self, values):
        """Replace the content; only the last ``size`` values are kept."""
        self._index.value = 0
        self._count.value = 0
        for value in list(values)[-self.size:]:
            self.push(value)


class RewardTracker:
    """Shared reward windows, best/worst rewards, and reward thresholds.

    One lock guards everything.
    """

    def __init__(self, num_workers, thresholds=()):
        self.num_workers = int(num_workers)
        self.thresholds = clean_thresholds(thresholds)
        self.lock = multiprocessing.Lock()

        self._global = SharedRewardWindow(REWARD_WINDOW)
        self._long = SharedRewardWindow(REWARD_WINDOW * self.num_workers)
        self._workers = [SharedRewardWindow(REWARD_WINDOW)
                         for _ in range(self.num_workers)]

        inf = float('inf')
        self._best = multiprocessing.Value('d', -inf, lock=False)
        self._worst = multiprocessing.Value('d', inf, lock=False)
        self._worker_best = multiprocessing.Array(
            'd', [-inf] * self.num_workers, lock=False)
        self._worker_worst = multiprocessing.Array(
            'd', [inf] * self.num_workers, lock=False)

        # One slot per (signal, threshold). A slot holds _SLOT_FIELDS as
        # doubles, which store these integers exactly; global_t < 0 means
        # the threshold was not reached.
        self._pairs = [(signal, threshold) for signal in THRESHOLD_SIGNALS
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
        if stored['global_t'] < 0:
            return None
        signal, threshold = self._pairs[slot]
        return {
            'signal': signal,
            'threshold': threshold,
            'value': stored['value'],
            'global_t': int(stored['global_t']),
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
        hits.sort(key=lambda hit: (hit['global_t'], hit['global_episode']))
        return hits

    # -- recording --

    def record_episode(self, worker_id, reward, global_t, global_episode):
        """Count one completed episode.

        Returns ``(stats, new_hits)``.  ``stats`` is a flat dict whose keys
        become fields of the episode record; ``new_hits`` lists the
        thresholds this episode was the first to reach.
        """
        reward = float(reward)
        with self.lock:
            self._workers[worker_id].push(reward)
            self._global.push(reward)
            self._long.push(reward)

            is_new_best = reward > self._best.value
            if is_new_best:
                self._best.value = reward
            if reward < self._worst.value:
                self._worst.value = reward
            if reward > self._worker_best[worker_id]:
                self._worker_best[worker_id] = reward
            if reward < self._worker_worst[worker_id]:
                self._worker_worst[worker_id] = reward

            local = self._workers[worker_id].stats()
            run_100 = self._global.stats()
            run_long = self._long.stats()
            elapsed = self.elapsed_training_s()

            stats = {
                'is_new_best': is_new_best,
                'local_mean_reward': local['mean'],
                'local_reward_min_100': local['min'],
                'local_reward_max_100': local['max'],
                'local_best_reward': self._worker_best[worker_id],
                'local_worst_reward': self._worker_worst[worker_id],
                'global_mean_reward_100': run_100['mean'],
                'global_reward_min_100': run_100['min'],
                'global_reward_max_100': run_100['max'],
                'global_mean_reward': run_long['mean'],
                'global_reward_min': run_long['min'],
                'global_reward_max': run_long['max'],
                'best_reward': self._best.value,
                'worst_reward': self._worst.value,
                'elapsed_training_s': elapsed,
            }

            signal_values = {
                'episode_reward': reward,
                'reward_mean_100':
                    run_100['mean'] if self._global.full else None,
                'reward_mean': run_long['mean'] if self._long.full else None,
            }
            new_hits = []
            for slot, (signal, threshold) in enumerate(self._pairs):
                value = signal_values[signal]
                if value is None or value < threshold or \
                        self._read_hit(slot) is not None:
                    continue
                self._write_hit(slot, {
                    'global_t': global_t, 'global_episode': global_episode,
                    'elapsed_training_s': elapsed, 'value': value,
                    'worker': worker_id})
                new_hits.append(self._read_hit(slot))
        return stats, new_hits

    # -- summary --

    def summary(self):
        """Flat entries for ``final_summary`` and the W&B run summary."""
        with self._read_lock():
            run_100 = self._global.stats()
            run_long = self._long.stats()
            best, worst = self._best.value, self._worst.value
            hits = [self._read_hit(slot) for slot in range(len(self._pairs))]
        summary = {
            'best_reward': best if math.isfinite(best) else None,
            'worst_reward': worst if math.isfinite(worst) else None,
            'final_global_mean_reward': run_long['mean'],
            'final_reward_min': run_long['min'],
            'final_reward_max': run_long['max'],
            'final_reward_mean_100': run_100['mean'],
            'final_reward_min_100': run_100['min'],
            'final_reward_max_100': run_100['max'],
        }
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
                'global': self._global.values(),
                'long': self._long.values(),
                'workers': [window.values() for window in self._workers],
                'best_reward': self._best.value,
                'worst_reward': self._worst.value,
                'worker_best': list(self._worker_best),
                'worker_worst': list(self._worker_worst),
                'hits': self._all_hits(),
                'elapsed_training_s': self.elapsed_training_s(),
            }

    def load_state_dict(self, state):
        """Restore a ``state_dict`` snapshot; return its training time.

        The return value is ``elapsed_training_s`` of the snapshot, or
        ``None`` if it has none.  A different number of workers is
        tolerated: worker lists are zipped (workers without saved data start
        empty) and the long window keeps only its most recent values.  Only
        thresholds configured now are restored; the others stay in the old
        ``events.jsonl``.
        """
        inf = float('inf')
        saved_workers = list(state.get('workers', ()))
        saved_best = list(state.get('worker_best', ()))
        saved_worst = list(state.get('worker_worst', ()))
        with self.lock:
            self._global.load(state.get('global', ()))
            self._long.load(state.get('long', ()))
            for index, window in enumerate(self._workers):
                window.load(saved_workers[index]
                            if index < len(saved_workers) else ())
                self._worker_best[index] = float(
                    saved_best[index] if index < len(saved_best) else -inf)
                self._worker_worst[index] = float(
                    saved_worst[index] if index < len(saved_worst) else inf)
            self._best.value = float(state.get('best_reward', -inf))
            self._worst.value = float(state.get('worst_reward', inf))
            for slot in range(len(self._pairs)):
                self._write_hit(slot, None)
            for hit in state.get('hits', ()):
                try:
                    slot = self._slot_of[
                        (hit['signal'], float(hit['threshold']))]
                except (KeyError, TypeError, ValueError):
                    continue
                self._write_hit(slot, hit)
        elapsed = state.get('elapsed_training_s')
        return None if elapsed is None else float(elapsed)

    def load_legacy(self, checkpoint_state):
        """Migrate a checkpoint written before ``RewardTracker`` existed.

        The old state is one ring buffer (``recent_rewards``,
        ``recent_reward_count``, ``recent_reward_index``) and
        ``best_reward``.  The run-wide worst reward becomes the lowest value
        in the restored window.  Per-worker windows and per-worker best and
        worst values, thresholds, and the training time start empty.
        Returns ``None`` (no training time), as ``load_state_dict`` does for
        a snapshot without one.
        """
        buffer = [float(value)
                  for value in checkpoint_state.get('recent_rewards', ())]
        size = len(buffer)
        count = min(int(checkpoint_state.get('recent_reward_count', 0)), size)
        if size and count == size:
            start = int(checkpoint_state.get('recent_reward_index', 0)) % size
            values = buffer[start:] + buffer[:start]
        else:
            values = buffer[:count]
        with self.lock:
            self._long.load(values)
            self._global.load(values[-REWARD_WINDOW:])
            self._best.value = float(
                checkpoint_state.get('best_reward', -float('inf')))
            self._worst.value = min(values) if values else float('inf')
        return None
