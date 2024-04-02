import argparse
import os
import time

import numpy as np
import torch
import torchvision.models as models
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from datasets_classes.PKLot_Dataset import PKLotDataset


# TODO: visualize more metrics, like loss, accuracy, confusion matrix, etc. and save them as graphs or something similar

def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (i + 1) % 100 == 0:
            print(f'Batch [{i + 1}/{len(dataloader)}], Train Loss: {running_loss / (i + 1):.4f},'
                  f' Train Acc: {(correct / total):.4f}')

    return running_loss / len(dataloader), correct / total


def test_model(model, dataloader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    print(f'Test Accuracy: {(sum(np.array(all_labels) == np.array(all_predictions)) / len(all_labels)):.4f}')
    print('Confusion Matrix:')
    print(confusion_matrix(all_labels, all_predictions))


def train_and_evaluate(train_dataloader, test_dataloader, model, criterion, optimizer, device, num_epochs, dataset_name,
                       model_name, eighty_twenty_split=False):
    best_acc = 0.0
    epoch_time = time.time()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')

        train_loss, train_acc = train_model(model, train_dataloader, criterion, optimizer, device)
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')

        test_model(model, test_dataloader, device)

        if train_acc > best_acc:
            best_acc = train_acc
            if eighty_twenty_split:
                save_path = f'../../data/models/80_20_split/pklot_{dataset_name.lower()}_{model_name}_default.pth'
            else:
                save_path = f'../../data/models/pklot_{dataset_name.lower()}_{model_name}_default.pth'
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            torch.save(model.state_dict(), save_path)
            print(f'Saved the new best model to {save_path}')

        print(f'Epoch time: {(time.time() - epoch_time) / 60} minutes and {(time.time() - epoch_time) % 60} seconds')

    print(f'Best Train Acc: {best_acc:.4f}')


def main():
    # TODO: make some arguments global for better readability
    # TODO: make some arguments optional or set their default values

    parser = argparse.ArgumentParser(description='Train a model on the PKLot dataset.')

    parser.add_argument('--dataset', type=str, help='The dataset to use. Choose between PUC, UFPR04, and UFPR05.')
    parser.add_argument('--model', type=str, help='The model to use. Choose between alexnet, and TODO.')
    parser.add_argument('--epochs', type=int, help='The number of epochs to train the model for.')
    parser.add_argument('--eighty_twenty_split', type=bool,
                        help='Whether to use an 80/20 split for training and testing.')
    parser.add_argument('--train_txt', type=str, help='The path to the text file containing the training data.',
                        default="Empty")
    parser.add_argument('--test_txt', type=str, help='The path to the text file containing the testing data.',
                        default="Empty")

    # TODO: add options from split text file for different datasets and ADD logic for it

    # TODO: add options and logic for different models
    # TODO: optional: somehow add weights for different models

    # TODO: optional: add option for different batch sizes
    # TODO: optional: add option for different learning rates

    # TODO: create datasets from CNR and CNR+Ext split for PKLot and add option for parameter
    """
    comment: CNR will only have text files for training and testing, so the split will be done in the same way as for
    the PKLot dataset if the parameter is set to False aka train_txt and test_txt are not empty hence the argument eight_twenty_split
    will be no longer needed
    """

    args = parser.parse_args()

    argument = args.dataset
    if argument == 'puc':
        dataset_name = 'PUC'
        dataset_dir = f'../data/datasets/pklot/PKLotFormatted/split_by_views_without_weather/{dataset_name}'
    elif argument == 'ufpr04':
        dataset_name = 'UFPR04'
        dataset_dir = f'../data/datasets/pklot/PKLotFormatted/split_by_views_without_weather/{dataset_name}'
    elif argument == 'ufpr05':
        dataset_name = 'UFPR05'
        dataset_dir = f'../data/datasets/pklot/PKLotFormatted/split_by_views_without_weather/{dataset_name}'
    elif argument == 'all':
        dataset_name = 'all'
        dataset_dir = f'../data/datasets/pklot/PKLotFormatted/all_in_one'
    else:
        raise ValueError('Invalid dataset. Please choose between PUC, UFPR04, and UFPR05.')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = PKLotDataset.PKLotDataset(root_dir=dataset_dir, transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_puc_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_puc_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_argument = args.model
    if model_argument == 'alexnet':
        model_alexnet = models.alexnet(weights="DEFAULT")
        model_alexnet.to(device)
    else:
        raise ValueError('Invalid model. Please choose alexnet, ...')

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model_alexnet.parameters(), lr=0.001)

    num_epochs = args.epochs

    is_eighty_twenty_split = args.eighty_twenty_split

    total_time = time.time()
    train_and_evaluate(train_puc_dataloader, test_puc_dataloader, model_alexnet, criterion, optimizer, device,
                       num_epochs, dataset_name, model_argument, is_eighty_twenty_split)
    print(
        f'Total training time: {(time.time() - total_time) / 60} minutes and {(time.time() - total_time) % 60} seconds')


if __name__ == "__main__":
    main()
