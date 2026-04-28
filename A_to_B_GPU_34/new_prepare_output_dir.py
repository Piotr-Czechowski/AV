"""Set up the run output directory + snapshot code state. Ported from async-rl."""

import os
import tempfile
import json
import subprocess


def prepare_output_dir(args, user_specified_dir=None, resume=False):
    """Create the output directory and dump args + git state into it.

    args can be a dict or argparse.Namespace. On resume=True, suffix is
    added so first-run artefacts are not clobbered.
    """
    if user_specified_dir is not None:
        if os.path.exists(user_specified_dir):
            if not os.path.isdir(user_specified_dir):
                raise RuntimeError(
                    '{} is not a directory'.format(user_specified_dir))
        else:
            os.makedirs(user_specified_dir)
        outdir = user_specified_dir
    else:
        outdir = tempfile.mkdtemp(prefix='a3c_run_')

    suffix = '_resume' if resume else ''

    args_dict = args if isinstance(args, dict) else vars(args)
    with open(os.path.join(outdir, 'args{}.txt'.format(suffix)), 'w') as f:
        json.dump(args_dict, f, indent=2, default=str)

    for name, cmd in (
        ('git-status', 'git status'),
        ('git-log',    'git log --max-count=50'),
        ('git-diff',   'git diff'),
    ):
        try:
            out = subprocess.getoutput(cmd)
        except Exception as e:
            out = 'error: {}'.format(e)
        path = os.path.join(outdir, '{}{}.txt'.format(name, suffix))
        try:
            with open(path, 'w') as f:
                f.write(out)
        except OSError:
            pass

    return outdir
