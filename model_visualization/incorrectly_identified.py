import os
import sys
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as func
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

DATASET_DIR = '../data/datasets/'

NUMBER_OF_PLOTS = 9


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
    # dataset_path = "../data/splits/cnr/cnr_park_ext_test_80_20.txt"
    # model_path = "../data/models/80_20_split/cnr/cnr_park_ext/5_epochs/mobilenet.pth"

    dataset_path = "../data/splits/cnr/cnr_park_test_80_20.txt"
    model_path = "../data/models/80_20_split/cnr/cnr_park/5_epochs/mobilenet.pth"

    # dataset_path = "../data/splits/pklot/all_test_80_20.txt"
    # model_path = "../data/models/80_20_split/pklot/all/5_epochs/mobilenet.pth"

    # dataset_path = "../data/splits/acpds/all_test_80_20.txt"
    # model_path = "../data/models/80_20_split/acpds/all/5_epochs/mobilenet.pth"

    # dataset_path = "../data/splits/spkl/all_test_80_20.txt"
    # model_path = "../data/models/80_20_split/spkl/all/5_epochs/mobilenet.pth"

    incorrect_images = []

    model = test_model.load_model(model_path)
    model.to(device)
    model.eval()

    model_name = model_path.split('/')[-1].split('.')[0]
    model_dataset_name = model_path.split('/')[4]
    model_dataset_view = model_path.split('/')[5]

    test_dataset_name = dataset_path.split('/')[3]
    test_dataset_view = dataset_path.split('/')[4].split('_test_80_20.txt')[0]

    dataset_loader = load_dataset(dataset_path)
    idx = 0
    batch_count = 0

    for images, labels in dataset_loader:
        batch_count += 1
        print(f"Processing batch {batch_count}...\n\tNumber of incorrect images collected: "
              f"{len(incorrect_images)}/{NUMBER_OF_PLOTS}")

        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        probabilities = func.softmax(outputs, dim=1)
        confidences, predicted = torch.sigmoid(probabilities).max(dim=1)

        incorrect = (predicted != labels).nonzero(as_tuple=True)[0]
        for index in incorrect:
            image = images[index].cpu()
            incorrect_images.append((image, labels[index].item(), predicted[index].item(), confidences[index].item()))
            idx += 1
            if idx == NUMBER_OF_PLOTS:
                break
        if idx == NUMBER_OF_PLOTS:
            break

    fig = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(hspace=0.5)
    for i in range(NUMBER_OF_PLOTS):
        ax = fig.add_subplot(math.ceil(math.sqrt(NUMBER_OF_PLOTS)), math.ceil(math.sqrt(NUMBER_OF_PLOTS)), i + 1)
        image, true_label, predicted_label, confidence = incorrect_images[i]
        image = image.permute(1, 2, 0).numpy()
        image = image * NORMALIZE_STD + NORMALIZE_MEAN
        image = image.clip(0, 1)
        ax.imshow(image)
        ax.set_title(f"Skutečnost: {true_label}\nPředpověď: {predicted_label}", fontsize=15)
        ax.axis('off')

    plt.tight_layout()

    save_path = (f'../data/incorrectly_identified_images/{model_name}_{model_dataset_name}_{model_dataset_view}/'
                 f'{test_dataset_name}_{test_dataset_view}')

    print(save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(os.path.join(save_path, 'incorrectly_identified.png'))
    plt.show()


if __name__ == "__main__":
    sys.path.append('../')

    import model_visualization.test_models as test_model
    import datasets_classes.SplitFileDataset as SplitFileDataset

    main()
