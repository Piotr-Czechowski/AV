"""Wall-clock timer for profiling where time goes in the training loop.

Ported from async-rl/timing_utils.py.

Usage:
    timer = TimingAccumulator()
    t0 = timer.start()
    do_stuff()
    timer.record('stuff', t0)
    timer.log_and_reset('W0')
"""
import time
from collections import defaultdict
from datetime import datetime


class TimingAccumulator:
    __slots__ = ('_totals', '_counts')

    def __init__(self):
        self._totals = defaultdict(float)
        self._counts = defaultdict(int)

    def start(self):
        return time.perf_counter()

    def record(self, name, t0):
        elapsed = time.perf_counter() - t0
        self._totals[name] += elapsed
        self._counts[name] += 1
        return elapsed

    def get_stats(self):
        stats = {}
        for name in sorted(self._totals):
            total = self._totals[name]
            count = self._counts[name]
            avg_ms = (total / count * 1000) if count else 0
            stats[name] = (avg_ms, count, total)
        return stats

    def log_and_reset(self, prefix=''):
        stats = self.get_stats()
        parts = []
        for name, (avg_ms, count, _) in stats.items():
            parts.append('{}:{:.2f}ms(n={})'.format(name, avg_ms, count))
        if parts:
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            print('[{}] [TIMING] {} {}'.format(ts, prefix, ' '.join(parts)),
                  flush=True)
        self._totals.clear()
        self._counts.clear()
        return stats
