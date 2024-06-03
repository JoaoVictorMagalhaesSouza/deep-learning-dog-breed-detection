import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image


class LoadImages():

    def __init__(
            self,
            folder_path: str,
            labels_path: str,
            default_size: int = 256,
            qtde_images: float = 1.0     
    ):
        self.folder_path = folder_path
        self.labels_path = labels_path
        self.labels = []
        self.images = []
        self.default_size = default_size
        self.qtde_images = qtde_images
        
    
    def load_images(self):
        all_images = os.listdir(self.folder_path)
        labels_dataset = pd.read_csv(self.labels_path + "labels.csv")
        total_images = int(len(all_images) * self.qtde_images)
        for i,image in enumerate(all_images):
            if i <= total_images:
                img = cv2.imread(self.folder_path + image)
                img = Image.fromarray(img, 'RGB')
                img = img.resize((self.default_size, self.default_size))
                self.images.append(np.array(img))
                try:
                    self.labels.append(labels_dataset.query(f"id == '{image.split('.')[0]}'")["breed"].values[0])
                except:
                    self.labels.append("unknown")

        return self.images, self.labels
