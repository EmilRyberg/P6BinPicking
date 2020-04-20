import torch
from torch.utils.data import Dataset
import os
import glob
from PIL import Image
import numpy as np


class OrientationDataset(Dataset):
    def __init__(self, path):
        self.image_with_labels = []
        for index, folder in enumerate(glob.glob(f"{path}/*/")):
            for image_path in glob.glob(f"{folder}*.jpg"):
                self.image_with_labels.append((image_path, index))

    def __getitem__(self, index):
        image_path, label = self.image_with_labels[index]
        pil_image = Image.open(image_path).convert('RGB')
        resized_image = pil_image.resize((224, 224))
        np_img = np.array(resized_image) / 255
        img_tensor = torch.from_numpy(np_img).permute(2, 0, 1).float()
        return image_path, img_tensor, torch.tensor(label).unsqueeze(0).float()

    def __len__(self):
        return len(self.image_with_labels)

