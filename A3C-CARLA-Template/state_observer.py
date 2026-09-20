"""Latest camera frame holder for CarlaEnv / wrapper."""

import os


class StateObserver:
    """Hold the latest camera image. JPEG dumps go through ``carla.Image.save_to_disk``."""

    def __init__(self, output_dir=None):
        self.image = None
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'images')

    def reset(self):
        self.image = None
