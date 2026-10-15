"""Evaluation entry point (not implemented yet).

Evaluation will be a separate program, not a flag of ``train_a3c.py``.
Training has no test mode: a training run only trains, and its directory
holds only training logs and checkpoints.

Planned command line:

    python evaluation.py --checkpoint runs/<run>/checkpoint.pth \\
        --episodes 50 --scenario 14 --map-name Town03 \\
        --port 2000 --outdir runs/eval_<name>

Planned behaviour:

- It reads only the weights (``checkpoint['model']``) and never updates the
  network.
- The policy is deterministic: ``argmax`` over the action logits instead of
  sampling.
- Routes repeat between runs (``select: cycle`` scenarios, fixed seed), so
  two checkpoints can be compared.
- After N episodes it writes ``summary.json`` to ``--outdir``: success rate
  (``reached_goal``), mean reward, collisions, episode length, minimum route
  distance, and goal distance. A per-episode JSONL file and camera frames
  are optional.
- It writes nothing to the training logs or to the training W&B run, and
  ``--outdir`` is never the directory of a training run.

As in training, the CARLA server must already run when evaluation starts.
"""

import sys


def main():
    print('evaluation.py is not implemented yet.', file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
