import os
import sys
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import torch.nn.functional as func
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

DATASET_DIR = '../data/datasets/'

NUMBER_OF_PLOTS_PER_DATASET = 5
TOTAL_DATASETS = 5
TOTAL_PLOTS = NUMBER_OF_PLOTS_PER_DATASET * TOTAL_DATASETS

MODEL_PATHS = {
    'CNR Park Ext': "../data/models/80_20_split/cnr/cnr_park_ext/5_epochs/mobilenet.pth",
    'CNR Park': "../data/models/80_20_split/cnr/cnr_park/5_epochs/mobilenet.pth",
    'PKLot': "../data/models/80_20_split/pklot/all/5_epochs/mobilenet.pth",
    'ACPDS': "../data/models/80_20_split/acpds/all/5_epochs/mobilenet.pth",
    'SPKL': "../data/models/80_20_split/spkl/all/5_epochs/mobilenet.pth"
}

class_names = {
    'CNR Park Ext': ['0', '1'],
    'CNR Park': ['0', '1'],
    'PKLot': ['0', '1'],
    'ACPDS': ['0', '1'],
    'SPKL': ['0', '1']
}


def load_dataset(path) -> DataLoader:
    transform_valid = transforms.Compose([
        transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])

    dataset = SplitFileDataset.SplitFileDataset(file_path=path, root_dir=DATASET_DIR, transform=transform_valid)
    dataset_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    return dataset_loader


def main() -> None:
    dataset_paths = [
        "../data/splits/cnr/cnr_park_ext_test_80_20.txt",
        "../data/splits/cnr/cnr_park_test_80_20.txt",
        "../data/splits/pklot/all_test_80_20.txt",
        "../data/splits/acpds/all_test_80_20.txt",
        "../data/splits/spkl/all_test_80_20.txt"
    ]

    dataset_names = ['CNR Park Ext', 'CNR Park', 'PKLot', 'ACPDS', 'SPKL']

    fig, axes = plt.subplots(TOTAL_DATASETS, NUMBER_OF_PLOTS_PER_DATASET, figsize=(15, 15))
    fig.subplots_adjust(hspace=0.5)  # Increase vertical spacing between subplots

    for idx, (dataset_path, dataset_name) in enumerate(zip(dataset_paths, dataset_names)):
        dataset_loader = load_dataset(dataset_path)
        model_path = MODEL_PATHS[dataset_name]
        model = test_model.load_model(model_path)
        model.to(device)
        model.eval()

        incorrect_images = []

        for plot_idx in range(NUMBER_OF_PLOTS_PER_DATASET):
            for images, labels in dataset_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)

                probabilities = func.softmax(outputs, dim=1)
                confidences, predicted = torch.sigmoid(probabilities).max(dim=1)

                incorrect = (predicted != labels).nonzero(as_tuple=True)[0]
                for index in incorrect:
                    image = images[index].cpu()
                    incorrect_images.append((image, labels[index].item(), predicted[index].item(),
                                             confidences[index].item(), class_names[dataset_name]))
                    if len(incorrect_images) == NUMBER_OF_PLOTS_PER_DATASET:
                        break
                if len(incorrect_images) == NUMBER_OF_PLOTS_PER_DATASET:
                    break
            if len(incorrect_images) == NUMBER_OF_PLOTS_PER_DATASET:
                break

        for col_idx, (image_data) in enumerate(incorrect_images):
            ax = axes[idx, col_idx]
            image, true_label, predicted_label, confidence, classes = image_data
            image = image.permute(1, 2, 0).numpy()
            image = image * NORMALIZE_STD + NORMALIZE_MEAN
            image = image.clip(0, 1)
            ax.imshow(image)
            ax.set_title(f'Skutečnost: {classes[true_label]}\n Predikce: {classes[predicted_label]}')
            ax.axis('off')

        axes[idx, 0].annotate(dataset_name, xy=(0, 0.5), xytext=(-axes[idx, 0].yaxis.labelpad - 5, 0),
                              xycoords=axes[idx, 0].yaxis.label, textcoords='offset points',
                              size='large', ha='right', va='center')

    plt.tight_layout()

    save_path = '../data/incorrectly_identified_images/all_datasets.png'

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    sys.path.append('../')

    import model_visualization.test_models as test_model
    import datasets_classes.SplitFileDataset as SplitFileDataset

    main()
