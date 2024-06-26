from typing import Tuple

from matplotlib import pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import shufflenet_v2_x0_5, mobilenet_v2, squeezenet1_0
import torch
import numpy as np
import os
import sys
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

PATH_TO_MODELS = '../data/models/80_20_split/'
PATH_TO_DATASETS = '../data/splits/'
SAVE_DIR = '../data/model_results/'
DATASET_DIR = '../data/datasets/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def denormalize(image, mean, std) -> np.ndarray:
    image = image.clone().cpu().numpy().transpose((1, 2, 0))
    image = image * std + mean  # denormalize
    image = np.clip(image, 0, 1)  # clip the pixel values
    return image


def load_model(path) -> torch.nn.Module:
    model_string = path.split('/')[-1].split('.')[0]
    state_dict = torch.load(path, map_location=device)

    if model_string == 'mobilenet':
        model = mobilenet_v2(weights=None)
    elif model_string == 'squeezenet':
        model = squeezenet1_0(weights=None)
    elif model_string == 'shufflenet':
        model = shufflenet_v2_x0_5(weights=None)
    else:
        raise ValueError('Invalid model. Please choose between mobilenet, squeezenet, shufflenet.')

    model.load_state_dict(state_dict)
    model.to(device)
    return model


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


def load_dataset(path) -> DataLoader:
    transform_valid = transforms.Compose([
        transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])
    dataset = SplitFileDataset.SplitFileDataset(file_path=path, root_dir=DATASET_DIR, transform=transform_valid)
    dataset_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    return dataset_loader


def test_model(model, dataset) -> Tuple[float, np.ndarray, float, float, np.ndarray]:
    model.eval()
    correct_predictions = []
    incorrect_predictions = []
    all_labels = []
    all_predictions = []
    all_thresholds = []
    with torch.no_grad():
        for images, labels in dataset:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = torch.sigmoid(outputs).argmax(dim=1)
            correct_predictions.extend(predictions.eq(labels.view_as(predictions)).cpu().numpy())
            incorrect_predictions.extend(predictions.ne(labels.view_as(predictions)).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_thresholds.extend(torch.sigmoid(outputs).cpu().numpy())

            if len(all_labels) % 100 == 0:
                print(f'{len(all_labels)} samples processed out of {len(dataset.dataset)}')

    accuracy = np.mean(correct_predictions)
    confusion = confusion_matrix(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    roc_auc = roc_auc_score(all_labels, all_predictions)
    return accuracy, confusion, f1, roc_auc, all_labels


def main() -> None:
    model_paths_generator = model_paths()
    dataset_paths_generator = dataset_paths()

    total_models = len(list(model_paths()))
    total_datasets = len(list(dataset_paths()))

    index = 0

    for model_path in model_paths_generator:
        model = load_model(model_path)
        torch.cuda.empty_cache()

        dataset_paths_generator = dataset_paths()  # Reset dataset generator for each model

        for dataset_path in dataset_paths_generator:
            dataset = load_dataset(dataset_path)
            index += 1

            model_name = model_path.split('/')[-1].split('.')[0]
            model_dataset_name = model_path.split('/')[4]
            model_dataset_view = model_path.split('/')[5]
            model_epochs = model_path.split('/')[6].split('_')[0]

            print(f'Model: {model_name}, Dataset: {model_dataset_name}, View: {model_dataset_view},'
                  f' Epochs: {model_epochs}')

            test_dataset_name = dataset_path.split('/')[3]
            test_dataset_view = dataset_path.split('/')[4].split('_test_80_20.txt')[0]

            print(f'Test Dataset: {test_dataset_name}, View: {test_dataset_view}')

            print('*' * 75)

            save_path = os.path.join(SAVE_DIR, f'{test_dataset_name}_{test_dataset_view}',
                                     f'{model_dataset_name}_{model_dataset_view}_{model_name}_{model_epochs}.txt')
            print(f'Path to save results: {save_path}')

            # check if text file with results already exists
            if os.path.exists(save_path):
                print(f'{save_path} already exists, skipping. {index}/{total_models * total_datasets} models tested')
                continue
            else:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

            torch.cuda.empty_cache()

            accuracy, confusion, f1, roc_auc, all_labels = test_model(model, dataset)

            true_positives = confusion[1][1]
            false_positives = confusion[0][1]
            true_negatives = confusion[0][0]
            false_negatives = confusion[1][0]

            with open(save_path, 'w') as file:
                file.write(f'Train Model: {model_name}\n')
                file.write(f'Train Dataset: {model_dataset_name}\n')
                file.write(f'Train View: {model_dataset_view}\n')
                file.write(f'Train Epochs: {model_epochs}\n')

                file.write(f'Test Dataset: {test_dataset_name}\n')
                file.write(f'Test View: {test_dataset_view}\n')

                file.write(f'Accuracy: {accuracy}\n')
                file.write(f'F1: {f1}\n')
                file.write(f'ROC AUC: {roc_auc}\n')
                file.write(f'True Positives: {true_positives}\n')
                file.write(f'False Positives: {false_positives}\n')
                file.write(f'True Negatives: {true_negatives}\n')
                file.write(f'False Negatives: {false_negatives}\n')
                file.write(f'Confusion Matrix:\n{confusion}\n')

            print(f'Accuracy: {accuracy}, F1: {f1}, ROC AUC: {roc_auc}, True Positives: {true_positives},'
                  f' False Positives: {false_positives}, True Negatives: {true_negatives},'
                  f' False Negatives: {false_negatives}')

            print('*' * 75)

            print(f'{index}/{total_models * total_datasets} models tested')

            print('*' * 75)


if __name__ == "__main__":
    sys.path.append('../')
    import datasets_classes.SplitFileDataset as SplitFileDataset

    main()
