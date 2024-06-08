import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class AerialImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(self.image_dir)
        self.Hex_Classes = [
            ('Unlabeled', '#9B9B9B'),
            ('Building','#3C1098'),
            ('Land', '#8429F6'),
            ('Road', '#6EC1E4'),
            ('Vegetation', '#FEDD3A'),
            ('Water', '#E2A929'),
            ]

        self.class_rgb_values = [self.hex_to_rgb(color[1]) for color in self.Hex_Classes]
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        mask_name = self.images[idx].replace('.jpg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path)
        mask = Image.open(mask_path)


        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

    def label_mask(self, mask):
        mask = np.array(mask)
        labeled_mask = np.zeros(mask.shape[:2], dtype=np.int64)  # ignore channel dimension so :2

        for idx, color in enumerate(self.Hex_Classes):
            labeled_mask[np.all(mask == self.hex_to_rgb(color[1]), axis=-1)] = idx

        return labeled_mask

    def hex_to_rgb(self, hex):
        hex = hex.lstrip('#')
        return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))
