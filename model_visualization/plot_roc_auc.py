import os
from typing import Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
from torchvision import transforms

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
DATASET_DIR = '../data/datasets/'
PATH_TO_MODELS = '../data/models/80_20_split/'
PATH_TO_DATASETS = '../data/splits/'
SAVE_DIR = '../data/graphs/roc_auc/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def model_paths() -> str:
    for root, _, files in os.walk(PATH_TO_MODELS):
        for file in files:
            if file.endswith('.pth'):
                yield os.path.join(root, file).replace("\\", "/")


def dataset_paths() -> str:
    for root, _, files in os.walk(PATH_TO_DATASETS):
        if 'acmps' in root:
            continue
        for file in files:
            if file.endswith('_test_80_20.txt'):
                yield os.path.join(root, file).replace("\\", "/")


def test_model(model: torch.nn.Module, dataset: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_labels = []
    all_thresholds = []
    with torch.no_grad():
        for images, labels in dataset:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_thresholds.extend(probabilities[:, 1])  # Assume binary classification with class 1 probabilities
    return np.array(all_labels), np.array(all_thresholds)


def load_dataset(path) -> DataLoader:
    transform_valid = transforms.Compose([
        transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])
    dataset = SplitFileDataset.SplitFileDataset(file_path=path, root_dir=DATASET_DIR, transform=transform_valid)
    dataset_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    return dataset_loader


def plot_roc_auc(labels: np.ndarray, thresholds: np.ndarray, train_model_string: str, train_dataset_string: str,
                 train_view_string: str, test_dataset_string: str, test_view_string: str):
    fpr, tpr, _ = roc_curve(labels, thresholds)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC křivka (plocha = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC křivka: {train_model_string} {train_dataset_string} {train_view_string} - '
              f'{test_dataset_string}_{test_view_string}')
    plt.legend(loc="lower right")

    save_path = f'{SAVE_DIR}{train_model_string}_{train_dataset_string}_{train_view_string}/' \
                f'{test_dataset_string}_{test_view_string}.png'

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    plt.savefig(save_path)
    plt.show()


def main():
    print('Loading model and dataset. You can change the model and dataset path in the main function.')
    model_path = '../data/models/80_20_split/cnr/cnr_park/5_epochs/shufflenet.pth'  # pick a model to test
    dataset_path = '../data/splits/cnr/cnr_park_ext_test_80_20.txt'  # pick a dataset to test the model on

    model_train = load_model(model_path)
    dataset_train = load_dataset(dataset_path)

    train_model_string = model_path.split('/')[7].split('.')[0]
    train_dataset_string = model_path.split('/')[4]
    train_view_string = model_path.split('/')[5]

    test_dataset_string = dataset_path.split('/')[3]
    test_view_string = dataset_path.split('/')[4].replace('_test_80_20.txt', '')

    print(f'Train Model: {train_model_string}, Train Dataset: {train_dataset_string}, Train View: {train_view_string}')
    print(f'Test Dataset: {test_dataset_string}, Test View: {test_view_string}')

    labels, thresholds = test_model(model_train, dataset_train)
    plot_roc_auc(labels, thresholds, train_model_string, train_dataset_string, train_view_string, test_dataset_string,
                 test_view_string)


if __name__ == "__main__":
    import sys

    sys.path.append('../')
    from test_models import load_model
    import datasets_classes.SplitFileDataset as SplitFileDataset

    main()
