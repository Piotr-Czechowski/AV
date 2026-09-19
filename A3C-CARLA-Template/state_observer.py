"""Optional on-disk snapshots of camera frames and overlay text."""

import os
import cv2
import numpy as np


class StateObserver:
    """Hold the latest camera image and write episode frames under ``output_dir``."""

    def __init__(self, output_dir=None):
        self.snapshot_date = None
        self.action = None
        self.reward = None
        self.image = None
        self.episode = None
        self.step = None
        self.manouver = None
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'images')

    def _episode_dir(self, *parts):
        """Return a path under ``output_dir``, creating it if needed."""
        directory = os.path.join(self.output_dir, *parts)
        os.makedirs(directory, exist_ok=True)
        return directory

    def save_to_disk(self):
        """Write the current camera image as JPEG."""
        self.image.save_to_disk(os.path.join(
            self._episode_dir(str(self.episode)), f"{int(self.step)}.jpeg"))

    def draw_related_values(self, episode=None, step=None):
        """Render a blue overlay with reward, action, timestamp, and maneuver."""
        image_sub = np.zeros((256, 256, 3), dtype=np.uint8)
        image_sub[:] = (255, 0, 0)  # BGR blue

        font = cv2.FONT_HERSHEY_SIMPLEX
        if self.reward:
            cv2.putText(image_sub, "Reward:", (3, 10), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image_sub, f"{self.reward}", (3, 20), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(image_sub, "Action:", (3, 30), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image_sub, f"{self.action}", (3, 40), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(image_sub, "Timnestamp:", (3, 50), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image_sub, f"{self.image.timestamp}", (3, 60), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(image_sub, "Manouver:", (3, 70), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image_sub, f"{self.manouver}", (3, 80), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        self.image_sub = np.vstack((image_sub,))

        episode_dir = self._episode_dir(str(self.episode))
        cv2.imwrite(os.path.join(
            episode_dir, f"{int(self.step)}_s.jpeg"), image_sub)

    def save_together(self):
        """Stack the camera frame and overlay vertically into one JPEG."""
        episode_dir = self._episode_dir(str(self.episode))
        image1 = cv2.imread(os.path.join(
            episode_dir, f"{int(self.step)}.jpeg"))
        image2 = cv2.imread(os.path.join(
            episode_dir, f"{int(self.step)}_s.jpeg"))

        if image1 is None or image2 is None:
            raise ValueError("Failed to load one of the images. Check that both files exist.")

        if image1.shape[1] != image2.shape[1]:
            width = min(image1.shape[1], image2.shape[1])
            image1 = cv2.resize(image1, (width, int(image1.shape[0] * width / image1.shape[1])))
            image2 = cv2.resize(image2, (width, int(image2.shape[0] * width / image2.shape[1])))

        combined_image = np.vstack((image1, image2))
        combined_dir = self._episode_dir('combined', str(self.episode))
        cv2.imwrite(os.path.join(
            combined_dir, f"{int(self.step)}_combined.jpeg"), combined_image)

    def reset(self):
        self.snapshot_date = None
        self.action = None
        self.reward = None
        self.image = None
        self.episode = None
        self.step = None
