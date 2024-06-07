import os
from torch.utils.data import Dataset
from PIL import Image


class SplitFileDataset(Dataset):
    def __init__(self, file_path, root_dir, transform=None):
        self.classes = []
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        with open(file_path, 'r') as f:
            for line in f:
                path, label = line.split()
                self.samples.append((path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(os.path.join(self.root_dir, path)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_class_distribution(self):
        class_distribution = {}
        for _, label in self.samples:
            if label in class_distribution:
                class_distribution[label] += 1
            else:
                class_distribution[label] = 1

        class_distribution['empty'] = class_distribution.pop(0)
        class_distribution['occupied'] = class_distribution.pop(1)

        return class_distribution
