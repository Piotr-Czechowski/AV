import os
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np


class StateObserver:
    def __init__(self):
        self.snapshot_date = None
        self.action = None
        self.reward = None
        self.image = None
        self.episode = None
        self.step = None
        self.manouver = None

    def save_to_disk(self):
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
        self.image.save_to_disk(f"A_to_B_GPU_34/images/{self.episode}/{int(self.step)}.jpeg")
        # self.episode = self.episode
        # self.step = self.step

    def draw_related_values(self, episode=None, step=None):
        # Tworzenie obrazu image2 o wymiarach 80x80 (kolor niebieski)
        image_sub = np.zeros((256, 256, 3), dtype=np.uint8)
        image_sub[:] = (255, 0, 0)  # Kolor niebieski w formacie BGR

        # Dodawanie tekstu "Image1" na obraz image1
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Dodawanie tekstu "Image1" na obraz image1
        if self.reward:
            cv2.putText(image_sub, "Reward:", (3, 10), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image_sub, f"{self.reward}", (3, 20), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        # Dodawanie tekstu "Image2" na obraz image2
        cv2.putText(image_sub, "Action:", (3, 30), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image_sub, f"{self.action}", (3, 40), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)


        cv2.putText(image_sub, "Timnestamp:", (3, 50), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image_sub, f"{self.image.timestamp}", (3, 60), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(image_sub, "Manouver:", (3, 70), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image_sub, f"{self.manouver}", (3, 80), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        self.image_sub = np.vstack((image_sub,))

        cv2.imwrite(f"A_to_B_GPU_34/images/{self.episode}/{int(self.step)}_s.jpeg", image_sub)

    def save_together(self):
        
        # Wczytanie dwóch obrazów JPG
        image1 = cv2.imread(f"A_to_B_GPU_34/images/{self.episode}/{int(self.step)}.jpeg")
        image2 = cv2.imread(f"A_to_B_GPU_34/images/{self.episode}/{int(self.step)}_s.jpeg")

        # Sprawdzanie, czy obrazy zostały poprawnie wczytane
        if image1 is None or image2 is None:
            raise ValueError("Nie udało się wczytać jednego z obrazów. Upewnij się, że pliki istnieją.")

        # Dopasowanie szerokości obrazów, jeśli jest różna
        if image1.shape[1] != image2.shape[1]:
            width = min(image1.shape[1], image2.shape[1])  # Minimalna szerokość
            image1 = cv2.resize(image1, (width, int(image1.shape[0] * width / image1.shape[1])))
            image2 = cv2.resize(image2, (width, int(image2.shape[0] * width / image2.shape[1])))

        # Łączenie obrazów w pionie (góra-dół)
        combined_image = np.vstack((image1, image2))

        # Zapis i wyświetlenie połączonego obrazu
        try:
            os.mkdir("A_to_B_GPU_34/images/combined")
        except FileExistsError:
            pass
        try:
            os.mkdir(f"A_to_B_GPU_34/images/combined/{self.episode}")
        except FileExistsError:
            pass
        cv2.imwrite(f"A_to_B_GPU_34/images/combined/{self.episode}/{int(self.step)}_combined.jpeg", combined_image)
    
    def reset(self):
        self.snapshot_date = None
        self.action = None
        self.reward = None
        self.image = None
        self.episode = None
        self.step = None