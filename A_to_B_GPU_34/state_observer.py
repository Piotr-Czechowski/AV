from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np


class StateObserver:
    def __init__(self):
        self.snapshot_date = None
        self.action = None
        self.reward = None
        self.episode_step = None
    def save_to_disk(self, image, episode, step):
        # # Convert the image to a format that OpenCV can work with (BGR format)
        # img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
        # img_array = img_array.reshape((image.height, image.width, 4))  # RGBA format
        # img = img_array[:, :, :3]  # Convert RGBA to BGR (OpenCV uses BGR format)
        
        # text = "Sample Text on Frame"
        # position = (50, 50)  # Position where the text will appear
        # font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
        # font_scale = 1  # Font size
        # color = (255, 255, 255)  # White color in BGR
        # thickness = 2  # Line thickness

        # # Put text on the image
        # cv2.putText(img, text, position, font, font_scale, color, thickness)
        image.save_to_disk(f"A_to_B_GPU_34/images/{episode}/{int(step)}.jpeg")

    def draw_related_values(self):
        pass
