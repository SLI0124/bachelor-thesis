import os
from torch.utils.data import Dataset


# TODO: implement this class, read the split file and load the images and labels, this might work for the PKLot
#  dataset since it has also text files with the labels for the split images

class CNRParkDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        raise NotImplementedError("This class is not implemented yet")

    def __len__(self):
        raise NotImplementedError("This class is not implemented yet")

    def __getitem__(self, idx):
        raise NotImplementedError("This class is not implemented yet")
