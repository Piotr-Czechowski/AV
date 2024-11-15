from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np


class StateObserver:
    def __init__(self):
        self.snapshot_date = None
        self.action = None
        self.reward = None
        self.episode_step = None
        self.image = None

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
        self.image = image
        self.image.save_to_disk(f"A_to_B_GPU_34/images/{episode}/{int(step)}.jpeg")

    def draw_related_values(self, episode, step):
        # Tworzenie obrazu image2 o wymiarach 80x80 (kolor niebieski)
        image_sub = np.zeros((80, 80, 3), dtype=np.uint8)
        image_sub[:] = (255, 0, 0)  # Kolor niebieski w formacie BGR

        # Dodawanie tekstu "Image1" na obraz image1
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Dodawanie tekstu "Image1" na obraz image1
        if self.reward:
            cv2.putText(image_sub, "Reward:", (3, 10), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image_sub, f"{self.reward}", (60, 10), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        # Dodawanie tekstu "Image2" na obraz image2
        cv2.putText(image_sub, "Action:", (3, 30), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image_sub, f"{self.action}", (60, 30), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)


        cv2.putText(image_sub, "Timnestamp:", (3, 50), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image_sub, f"{self.image.timestamp}", (3, 60), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        image_sub = np.vstack((image_sub,))

        cv2.imwrite(f"A_to_B_GPU_34/images/{episode}/{int(step)}_s.jpeg", image_sub)
