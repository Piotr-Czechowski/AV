"""Set up the run output directory and persist launch arguments."""

import os
import tempfile
import json


def prepare_output_dir(args, user_specified_dir=None, resume=False):
    """Create the output directory and dump the launch arguments into it.

    args can be a dict or argparse.Namespace. On resume=True, the args file
    gets a suffix so first-run files are not clobbered.
    """
    if user_specified_dir is not None:
        if os.path.exists(user_specified_dir):
            if not os.path.isdir(user_specified_dir):
                raise RuntimeError(
                    '{} is not a directory'.format(user_specified_dir))
        else:
            os.makedirs(user_specified_dir)
        run_output_dir = user_specified_dir
    else:
        run_output_dir = tempfile.mkdtemp(prefix='a3c_run_')

    suffix = '_resume' if resume else ''

    args_dict = args if isinstance(args, dict) else vars(args)
    with open(os.path.join(run_output_dir,
                           'args{}.txt'.format(suffix)), 'w') as f:
        json.dump(args_dict, f, indent=2, default=str)

    return run_output_dir
