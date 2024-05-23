import argparse
import os
import time
import logging

import numpy as np
import torch
import torchvision.models as models
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, recall_score
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from datasets_classes import SplitFileDataset

logger = logging.getLogger()

BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_OF_WORKERS = 4
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
ROTATION_ANGLE = 10
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
NUM_OF_EPOCHS_DEFAULT = 5

PATH_TO_SPLIT_FILES = '../data/splits'
PATH_TO_IMAGES = '../data/datasets'


def print_and_log(message):
    print(message)
    logger.info(message)


def calculate_metrics(labels, predictions):
    accuracy = np.mean(np.array(labels) == np.array(predictions))
    f1 = f1_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted', zero_division=1)

    if len(np.unique(labels)) == 2:
        roc_auc = roc_auc_score(labels, predictions)
    else:
        roc_auc = None

    return accuracy, f1, recall, roc_auc


def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    train_acc, train_f1, train_recall, train_roc_auc = 0.0, 0.0, 0.0, 0.0
    all_labels, all_predictions = [], []
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

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

        if (i + 1) % 100 == 0:
            train_acc, train_f1, train_recall, train_roc_auc = calculate_metrics(all_labels, all_predictions)

            print_and_log(f'Batch [{i + 1}/{len(dataloader)}], Train Loss: {loss.item()}, '
                          f'Train Acc: {train_acc}, Train F1: {train_f1}, Train Recall: {train_recall}, '
                          f'Train ROC AUC: {train_roc_auc}')

    print_and_log('-' * 50)
    return running_loss / len(dataloader), train_acc, train_f1, train_recall, train_roc_auc


def test_model(model, dataloader, criterion, device):
    model.eval()
    all_labels, all_predictions = [], []
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    test_loss /= len(dataloader)
    test_acc, test_f1, test_recall, test_roc_auc = calculate_metrics(all_labels, all_predictions)

    print_and_log(
        f'Test Loss: {test_loss}, Test Acc: {test_acc}, Test F1: {test_f1}, '
        f'Test Recall: {test_recall}, Test ROC AUC: {test_roc_auc}')

    cm = confusion_matrix(all_labels, all_predictions)
    print_and_log(f'Confusion Matrix:\n {cm}')

    return test_loss, test_acc, test_f1, test_recall, test_roc_auc


def train_and_evaluate(train_dataloader, test_dataloader, model, criterion, optimizer, device, num_epochs, dataset_name,
                       model_name, eighty_twenty_split, camera_view):
    best_acc = 0.0
    epoch_time = time.time()
    train_losses, train_accuracies, train_f1s, train_recalls, train_roc_aucs = [], [], [], [], []
    valid_losses, valid_accuracies, valid_f1s, valid_recalls, valid_roc_aucs = [], [], [], [], []

    for epoch in range(num_epochs):
        print_and_log(f'Epoch {epoch + 1}/{num_epochs}')

        train_loss, train_acc, train_f1, train_recall, train_roc_auc = train_model(model, train_dataloader, criterion,
                                                                                   optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        train_f1s.append(train_f1)
        train_recalls.append(train_recall)

        print_and_log(f'Train Loss: {train_loss}, Train Acc: {train_acc}, '
                      f'Train F1: {train_f1}, Train Recall: {train_recall}, Train ROC AUC: {train_roc_auc}')

        all_labels, all_probs = [], []
        with torch.no_grad():
            for images, labels in train_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = nn.functional.softmax(outputs, dim=1)[:, 1]  # probability of the positive class
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        train_roc_auc = roc_auc_score(all_labels, all_probs)
        train_roc_aucs.append(train_roc_auc)

        # valid_acc, valid_f1, valid_recall, valid_roc_auc = test_model(model, test_dataloader, device)
        valid_loss, valid_acc, valid_f1, valid_recall, valid_roc_auc = test_model(model, test_dataloader, criterion,
                                                                                  device)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_acc)
        valid_f1s.append(valid_f1)
        valid_recalls.append(valid_recall)
        valid_roc_aucs.append(valid_roc_auc)

        if train_acc > best_acc:
            best_acc = train_acc
            valid_split = 100 - eighty_twenty_split
            save_path = (f'../data/models/{dataset_name}/{camera_view}/{eighty_twenty_split}_{valid_split}_split/'
                         f'{num_epochs}_epochs/{model_name}.pth')

            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))

            torch.save(model.state_dict(), save_path)
            print_and_log(f'Saved the new best model to {save_path}')

        print_and_log(f'Epoch time: {time.time() - epoch_time} seconds.')
        epoch_time = time.time()
        print_and_log('=' * 50)

    print_and_log(f'Average Train Loss: {np.mean(train_losses)}')
    print_and_log(f'Average Train Accuracy: {np.mean(train_accuracies)}')
    print_and_log(f'Average Train F1: {np.mean(train_f1s)}')
    print_and_log(f'Average Train Recall: {np.mean(train_recalls)}')

    if train_roc_aucs:
        print_and_log(f'Average Train ROC AUC: {np.mean(train_roc_aucs)}')
    else:
        print_and_log("No Train ROC AUC available.")

    print_and_log(f'\nBest Train Accuracy: {best_acc}\n')

    print_and_log(f'Average Test Loss: {np.mean(valid_losses)}')
    print_and_log(f'Average Test Accuracy: {np.mean(valid_accuracies)}')
    print_and_log(f'Average Test F1: {np.mean(valid_f1s)}')
    print_and_log(f'Average Test Recall: {np.mean(valid_recalls)}')

    if valid_roc_aucs:
        print_and_log(f'Average Test ROC AUC: {np.mean(valid_roc_aucs)}\n')
    else:
        print_and_log("No Test ROC AUC available.\n")


def main():
    # parsing the arguments for the script from the command line
    parser = argparse.ArgumentParser(description='Train a model on a dataset.')

    parser.add_argument('--dataset', type=str,
                        help='The dataset to use. Choose between pklot, cnr, acmps, acpds, spkl')
    parser.add_argument('--camera_view', type=str,
                        help='For the PKLot dataset, choose between puc, ufpr04, ufpr05, and all, '
                             'for CNRPark choose between cnr_park (camera A and B), cnr_park_ext'
                             ' (camera 0-9) and all (all cameras), '
                             'for ACPDS default and only option is all, '
                             'for ACPMS default and only option is all, '
                             'for SPKL default and only option is all.')
    parser.add_argument('--model', type=str,
                        help='The model to use. Choose between alexnet, mobilenet, squeezenet and shufflenet.')
    parser.add_argument('--train_split', type=int,
                        help='The percentage of the dataset to use for training. The rest will be used for testing.')
    parser.add_argument('--num_epochs', type=int,
                        help='The number of epochs to train the model.', default=NUM_OF_EPOCHS_DEFAULT)

    # parse the arguments and store them in args variable, then assign them to the corresponding variables
    args = parser.parse_args()
    dataset_argument = args.dataset
    camera_view_argument = args.camera_view
    train_size_split_argument = args.train_split
    model_argument = args.model
    num_epochs_argument = args.num_epochs

    # print all the arguments
    print_and_log(f'Dataset: {dataset_argument}')
    print_and_log(f'Camera view: {camera_view_argument}')
    print_and_log(f'Train size split: {train_size_split_argument}')
    print_and_log(f'Test size split: {100 - train_size_split_argument}')
    print_and_log(f'Model: {model_argument}')
    print_and_log(f'Number of epochs: {num_epochs_argument}')
    print_and_log('*' * 50)

    # create the log file
    train_ratio = train_size_split_argument
    test_ratio = 100 - train_size_split_argument
    log_file_name = (f'../data/logs/{train_ratio}_{test_ratio}_split/'
                     f'{dataset_argument}/{camera_view_argument}/{num_epochs_argument}_epochs/{model_argument}.txt')

    # create the directories for the log file
    if not os.path.exists(os.path.dirname(log_file_name)):
        os.makedirs(os.path.dirname(log_file_name))

    if os.path.exists(log_file_name):
        print(f'Log file {log_file_name} already exists. Deleting it.')
        os.remove(log_file_name)

    # set the logger
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file_name)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    print(f'Logging to {log_file_name}')

    # create the paths to the split files
    train_split_file = (f'{PATH_TO_SPLIT_FILES}/{dataset_argument}/{camera_view_argument}'
                        f'_train_{train_size_split_argument}_{100 - train_size_split_argument}.txt')
    test_split_file = (f'{PATH_TO_SPLIT_FILES}/{dataset_argument}/'f'{camera_view_argument}'
                       f'_test_{train_size_split_argument}_{100 - train_size_split_argument}.txt')
    print_and_log(f'Train split file: {train_split_file}')
    print_and_log(f'Test split file: {test_split_file}')

    # transformations for the dataset, different for training and testing
    transform_train = transforms.Compose([
        transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(ROTATION_ANGLE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN,
                             std=NORMALIZE_STD)
    ])

    transform_test = transforms.Compose([
        transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN,
                             std=NORMALIZE_STD)
    ])

    # create the dataset and dataloader
    train_dataset = SplitFileDataset.SplitFileDataset(train_split_file, PATH_TO_IMAGES, transform=transform_train)
    test_dataset = SplitFileDataset.SplitFileDataset(test_split_file, PATH_TO_IMAGES, transform=transform_test)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_OF_WORKERS,
                                  pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_OF_WORKERS,
                                 pin_memory=True)

    # print the number of samples in the train and test datasets and the class distribution
    print_and_log(f'Train dataset size: {len(train_dataset)}')
    print_and_log(f'Train dataset class distribution: {train_dataset.get_class_distribution()}')
    print_and_log(f'Test dataset size: {len(test_dataset)}')
    print_and_log(f'Test dataset class distribution: {test_dataset.get_class_distribution()}')

    # set the device to cuda if available, otherwise to cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create match clause for the model_argument
    match model_argument:
        case 'alexnet':
            model = models.alexnet(weights=None)
        case 'mobilenet':
            model = models.mobilenet_v2(weights=None)
        case 'squeezenet':
            model = models.squeezenet1_0(weights=None)
        case 'shufflenet':
            model = models.shufflenet_v2_x0_5(weights=None)
        case _:
            raise ValueError(
                'Invalid model. Please choose between alexnet, mobilenet, squeezenet, shufflenet.')

    # setting the model to the device, setting the optimizer and the criterion
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # train the model
    print_and_log('Training the model...')
    print_and_log('*' * 50)
    total_time = time.time()

    train_and_evaluate(train_dataloader, test_dataloader, model, criterion, optimizer, device, num_epochs_argument,
                       dataset_argument, model_argument, train_size_split_argument, camera_view_argument)

    print_and_log(f'Total training time: {time.time() - total_time} seconds.')


if __name__ == "__main__":
    main()
