import os
from torch.utils.data import Dataset
from PIL import Image


class PKLotDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.empty_dir = os.path.join(root_dir, 'empty')
        self.occupied_dir = os.path.join(root_dir, 'occupied')

        self.empty_images = os.listdir(self.empty_dir)
        self.occupied_images = os.listdir(self.occupied_dir)

    def __len__(self):
        return len(self.empty_images) + len(self.occupied_images)

    def __getitem__(self, idx):
        if idx < len(self.empty_images):
            img_name = os.path.join(self.empty_dir, self.empty_images[idx])
            label = 0  # Empty space label
        else:
            idx = idx - len(self.empty_images)
            img_name = os.path.join(self.occupied_dir, self.occupied_images[idx])
            label = 1  # Occupied space label

        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image, label
