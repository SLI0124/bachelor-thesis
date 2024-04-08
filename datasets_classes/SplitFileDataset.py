import os
from torch.utils.data import Dataset
from PIL import Image


class SplitFileDataset(Dataset):
    def __init__(self, root_dir, split_file, transform=None):
        raise NotImplementedError("Implement me!")

    def __len__(self):
        raise NotImplementedError("Implement me!")

    def __getitem__(self, idx):
        raise NotImplementedError("Implement me!")
