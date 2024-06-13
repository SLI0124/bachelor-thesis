import argparse
import os
import time
import logging

import numpy as np
import torch
import torchvision.models as models
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import KFold
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import nn
from torch import optim

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
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')

    if len(np.unique(labels)) == 2:
        roc_auc = roc_auc_score(labels, predictions)
    else:
        roc_auc = None

    return accuracy, f1, roc_auc


def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total, train_acc, train_f1, train_roc_auc = 0.0, 0, 0, 0.0, 0.0, 0.0
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

        train_acc, train_f1, train_roc_auc = calculate_metrics(all_labels, all_predictions)

        if (i + 1) % 100 == 0:
            print_and_log(f'Batch [{i + 1}/{len(dataloader)}], Train Loss: {loss.item()}, '
                          f'Train Acc: {train_acc}, Train F1: {train_f1}, Train ROC AUC: {train_roc_auc}')

    print_and_log('-' * 50)
    return running_loss / len(dataloader), train_acc, train_f1, train_roc_auc


def validate_model(model, dataloader, criterion, device):
    model.eval()
    all_labels, all_predictions = [], []
    valid_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    valid_loss /= len(dataloader)
    valid_acc, valid_f1, valid_roc_auc = calculate_metrics(all_labels, all_predictions)

    print_and_log(
        f'Valid Loss: {valid_loss}, Valid Acc: {valid_acc}, Valid F1: {valid_f1}, Valid ROC AUC: {valid_roc_auc}')

    print_and_log(f'Confusion Matrix for the validation set:\n {confusion_matrix(all_labels, all_predictions)}')

    return valid_loss, valid_acc, valid_f1, valid_roc_auc


def train_and_evaluate(train_dataloader, valid_dataloader, model, criterion, optimizer, device, num_epochs,
                       dataset_name, model_name, eighty_twenty_split, camera_view):
    best_acc = 0.0
    epoch_time = time.time()
    train_losses, train_accuracies, train_f1s, train_roc_aucs = [], [], [], []
    valid_losses, valid_accuracies, valid_f1s, valid_roc_aucs = [], [], [], []

    for epoch in range(num_epochs):
        print_and_log(f'Epoch {epoch + 1}/{num_epochs}')

        train_loss, train_acc, train_f1, train_roc_auc = train_model(model, train_dataloader, criterion, optimizer,
                                                                     device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        train_f1s.append(train_f1)
        train_roc_aucs.append(train_roc_auc)

        print_and_log(
            f'Train Loss: {train_loss}, Train Acc: {train_acc}, Train F1: {train_f1}, Train ROC AUC: {train_roc_auc}')

        valid_loss, valid_acc, valid_f1, valid_roc_auc = validate_model(model, valid_dataloader, criterion, device)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_acc)
        valid_f1s.append(valid_f1)
        valid_roc_aucs.append(valid_roc_auc)

        if valid_acc > best_acc:
            best_acc = valid_acc
            valid_split = 100 - eighty_twenty_split
            save_path = (f'../data/models/{eighty_twenty_split}_{valid_split}_split/{dataset_name}/{camera_view}/'
                         f'{num_epochs}_epochs/{model_name}.pth')

            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))

            torch.save(model.state_dict(), save_path)
            print_and_log(f'Saved the new best model to {save_path} with accuracy {best_acc}')

        print_and_log(f'Epoch time: {time.time() - epoch_time} seconds.')
        epoch_time = time.time()
        print_and_log('=' * 50)

    print_and_log(f'Average Train Loss: {np.mean(train_losses)}')
    print_and_log(f'Average Train Accuracy: {np.mean(train_accuracies)}')
    print_and_log(f'Average Train F1: {np.mean(train_f1s)}')
    print_and_log(f'Average Train ROC AUC: {np.mean(train_roc_aucs)}')

    print_and_log(f'\nBest Validation Accuracy: {best_acc}\n')

    print_and_log(f'Average Valid Loss: {np.mean(valid_losses)}')
    print_and_log(f'Average Valid Accuracy: {np.mean(valid_accuracies)}')
    print_and_log(f'Average Valid F1: {np.mean(valid_f1s)}')
    print_and_log(f'Average Valid ROC AUC: {np.mean(valid_roc_aucs)}\n')


def train_and_evaluate_k_fold(dataset, model, criterion, optimizer, device, num_epochs,
                              dataset_name, model_name, camera_view, k_fold_input):
    def reset_weights(m):
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()

    results = {}
    best_valid_acc = 0.0
    k_fold = KFold(n_splits=k_fold_input, shuffle=True, random_state=42)

    torch.manual_seed(42)

    for fold, (train_indices, valid_indices) in enumerate(k_fold.split(dataset)):
        print_and_log(f'Fold {fold + 1}/{k_fold_input}')
        fold_start_time = time.time()

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
                                      num_workers=NUM_OF_WORKERS, pin_memory=True)
        valid_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=valid_sampler,
                                      num_workers=NUM_OF_WORKERS, pin_memory=True)

        model.apply(reset_weights)

        for epoch in range(num_epochs):
            print_and_log(f'Epoch {epoch + 1}/{num_epochs}')
            epoch_start_time = time.time()

            train_loss, train_acc, train_f1, train_roc_auc = train_model(model, train_dataloader, criterion, optimizer,
                                                                         device)

            print_and_log(f'Train Loss: {train_loss}, Train Acc: {train_acc},'
                          f' Train F1: {train_f1}, Train ROC AUC: {train_roc_auc}')

            valid_loss, valid_acc, valid_f1, valid_roc_auc = validate_model(model, valid_dataloader, criterion, device)

            if epoch not in results:
                results[epoch] = {
                    'train_loss': [],
                    'train_acc': [],
                    'train_f1': [],
                    'train_roc_auc': [],
                    'valid_loss': [],
                    'valid_acc': [],
                    'valid_f1': [],
                    'valid_roc_auc': []
                }

            results[epoch]['train_loss'].append(train_loss)
            results[epoch]['train_acc'].append(train_acc)
            results[epoch]['train_f1'].append(train_f1)
            results[epoch]['train_roc_auc'].append(train_roc_auc)
            results[epoch]['valid_loss'].append(valid_loss)
            results[epoch]['valid_acc'].append(valid_acc)
            results[epoch]['valid_f1'].append(valid_f1)
            results[epoch]['valid_roc_auc'].append(valid_roc_auc)

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                save_path = (f'../data/models/{k_fold_input}_fold/{dataset_name}/{camera_view}/'
                             f'{num_epochs}_epochs/{model_name}.pth')

                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))

                torch.save(model.state_dict(), save_path)
                print_and_log(f'Saved the new best model to {save_path} with validation accuracy {best_valid_acc}')

            epoch_end_time = time.time()
            print_and_log(f'Epoch time: {epoch_end_time - epoch_start_time} seconds.')
            print_and_log('=' * 50)

        fold_end_time = time.time()
        print_and_log(f'Fold time: {fold_end_time - fold_start_time} seconds.')
        print_and_log('=' * 50)

    for epoch in results:
        print_and_log(f'Results for epoch {epoch + 1}/{num_epochs}')

        print_and_log(f'Average Train Loss: {np.mean(results[epoch]["train_loss"])}')
        print_and_log(f'Average Train Accuracy: {np.mean(results[epoch]["train_acc"])}')
        print_and_log(f'Average Train F1: {np.mean(results[epoch]["train_f1"])}')
        print_and_log(f'Average Train ROC AUC: {np.mean(results[epoch]["train_roc_auc"])}')

        print_and_log(f'Average Valid Loss: {np.mean(results[epoch]["valid_loss"])}')
        print_and_log(f'Average Valid Accuracy: {np.mean(results[epoch]["valid_acc"])}')
        print_and_log(f'Average Valid F1: {np.mean(results[epoch]["valid_f1"])}')
        print_and_log(f'Average Valid ROC AUC: {np.mean(results[epoch]["valid_roc_auc"])}\n')

    print_and_log('=' * 50)

    print_and_log(f'Final results for {k_fold_input}-fold cross-validation')

    print_and_log(f'Average Train Loss: {np.mean([np.mean(results[epoch]["train_loss"]) for epoch in results])}')
    print_and_log(f'Average Train Accuracy: {np.mean([np.mean(results[epoch]["train_acc"]) for epoch in results])}')
    print_and_log(f'Average Train F1: {np.mean([np.mean(results[epoch]["train_f1"]) for epoch in results])}')
    print_and_log(f'Average Train ROC AUC: {np.mean([np.mean(results[epoch]["train_roc_auc"]) for epoch in results])}')

    print_and_log(f'\nAverage Valid Loss: {np.mean([np.mean(results[epoch]["valid_loss"]) for epoch in results])}')
    print_and_log(f'Average Valid Accuracy: {np.mean([np.mean(results[epoch]["valid_acc"]) for epoch in results])}')
    print_and_log(f'Average Valid F1: {np.mean([np.mean(results[epoch]["valid_f1"]) for epoch in results])}')
    print_and_log(f'Average Valid ROC AUC: {np.mean([np.mean(results[epoch]["valid_roc_auc"]) for epoch in results])}')

    print_and_log('=' * 50)

    print_and_log(f'Best Valid Accuracy: {best_valid_acc}')
    print_and_log('=' * 50)


def main():
    # parsing the arguments for the script from the command line
    parser = argparse.ArgumentParser(description='Train a model on a dataset.')

    parser.add_argument('--dataset', type=str, required=True,
                        help='The dataset to use. Choose between pklot, cnr, acmps, acpds, spkl')
    parser.add_argument('--camera_view', type=str, required=True,
                        help='For the PKLot dataset, choose between puc, ufpr04, ufpr05, and all, '
                             'for CNRPark choose between cnr_park (camera A and B), cnr_park_ext'
                             ' (camera 0-9) and all (all cameras), '
                             'for ACPDS default and only option is all, '
                             'for ACPMS default and only option is all, '
                             'for SPKL default and only option is all.')
    parser.add_argument('--model', type=str, required=True,
                        help='The model to use. Choose between alexnet, mobilenet, squeezenet and shufflenet.')
    parser.add_argument('--train_split', type=int,
                        help='The percentage of the dataset to use for training. The rest will be used for validating.')
    parser.add_argument('--num_epochs', type=int,
                        help='The number of epochs to train the model.', default=NUM_OF_EPOCHS_DEFAULT)
    parser.add_argument('--k_fold', type=int,
                        help='The number of folds for k-fold cross-validation.', default=None)

    # parse the arguments and store them in args variable, then assign them to the corresponding variables
    args = parser.parse_args()
    dataset_argument = args.dataset
    camera_view_argument = args.camera_view
    train_size_split_argument = args.train_split
    model_argument = args.model
    num_epochs_argument = args.num_epochs
    k_fold_argument = args.k_fold

    # if train_size_split_argument is None and k_fold is None, raise exception
    if not train_size_split_argument and not k_fold_argument:
        raise ValueError('Please provide either the train_split argument or the k_fold argument.')
    # if both are provided, raise exception as well
    if train_size_split_argument and k_fold_argument:
        raise ValueError('Please provide only one of the arguments, either train_split or k_fold.')

    # print all the arguments
    print_and_log(f'Dataset: {dataset_argument}')
    print_and_log(f'Camera view: {camera_view_argument}')
    print_and_log(f'Model: {model_argument}')
    if k_fold_argument:
        print_and_log(f'K-fold cross-validation with {k_fold_argument} folds.')
    else:
        print_and_log(f'Train size split: {train_size_split_argument}')
        print_and_log(f'Valid size split: {100 - train_size_split_argument}')
        print_and_log(f'Number of epochs: {num_epochs_argument}')
    print_and_log('*' * 50)

    # create the log file
    if k_fold_argument:
        log_file_name = (f'../data/logs/{k_fold_argument}_fold/'
                         f'{dataset_argument}/{camera_view_argument}/{num_epochs_argument}_epochs/{model_argument}.txt')
    else:
        train_ratio = train_size_split_argument
        valid_ratio = 100 - train_size_split_argument
        log_file_name = (f'../data/logs/{train_ratio}_{valid_ratio}_split/'
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
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    print(f'Logging to {log_file_name}')

    # create the paths to the split files
    test_split_file = None
    if k_fold_argument:
        train_split_file = (f'{PATH_TO_SPLIT_FILES}/{dataset_argument}/{camera_view_argument}'
                            f'_train_80_20.txt')
        valid_split_file = (f'{PATH_TO_SPLIT_FILES}/{dataset_argument}/{camera_view_argument}'
                            f'_valid_80_20.txt')
        test_split_file = (f'{PATH_TO_SPLIT_FILES}/{dataset_argument}/{camera_view_argument}'
                           f'_test_80_20.txt')
    else:
        train_split_file = (f'{PATH_TO_SPLIT_FILES}/{dataset_argument}/{camera_view_argument}'
                            f'_train_{train_size_split_argument}_{100 - train_size_split_argument}.txt')
        valid_split_file = (f'{PATH_TO_SPLIT_FILES}/{dataset_argument}/{camera_view_argument}'
                            f'_valid_{train_size_split_argument}_{100 - train_size_split_argument}.txt')
    print_and_log(f'Train split file: {train_split_file}')
    print_and_log(f'Validation split file: {valid_split_file}')
    print_and_log(f'Test split file: {test_split_file}' if k_fold_argument else '')
    print_and_log('*' * 50)

    # transformations for the dataset, different for training and validation
    transform_train = transforms.Compose([
        transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(ROTATION_ANGLE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN,
                             std=NORMALIZE_STD)
    ])

    transform_valid = transforms.Compose([
        transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN,
                             std=NORMALIZE_STD)
    ])

    # create the dataset and dataloader
    train_dataset = SplitFileDataset.SplitFileDataset(train_split_file, PATH_TO_IMAGES, transform=transform_train)
    valid_dataset = SplitFileDataset.SplitFileDataset(valid_split_file, PATH_TO_IMAGES, transform=transform_valid)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_OF_WORKERS,
                                  pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_OF_WORKERS,
                                  pin_memory=True)

    whole_dataset, test_dataset = None, None
    if k_fold_argument:
        test_dataset = SplitFileDataset.SplitFileDataset(test_split_file, PATH_TO_IMAGES, transform=transform_valid)
        whole_dataset = torch.utils.data.ConcatDataset([test_dataset, valid_dataset])
        whole_dataset = torch.utils.data.ConcatDataset([whole_dataset, train_dataset])

    # print the number of samples in the train and validation dataset and the class distribution
    if not k_fold_argument:
        print_and_log(f'Train dataset size: {len(train_dataset)}')
        print_and_log(f'Train dataset class distribution: {train_dataset.get_class_distribution()}')
        print_and_log(f'Valid dataset size: {len(valid_dataset)}')
        print_and_log(f'Valid dataset class distribution: {valid_dataset.get_class_distribution()}')
    else:
        print_and_log(f'Dataset size: {len(whole_dataset)}')
        # get class distribution for the whole dataset, each split has one dictionary, so we need to merge them
        class_distribution = {}
        for dataset in [train_dataset, valid_dataset, test_dataset]:
            for key, value in dataset.get_class_distribution().items():
                if key in class_distribution:
                    class_distribution[key] += value
                else:
                    class_distribution[key] = value
        print_and_log(f'Dataset class distribution: {class_distribution}')

    # set the device to cuda if available, otherwise to cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # choose the model based on the model_argument
    match model_argument:
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

    if k_fold_argument:
        train_and_evaluate_k_fold(whole_dataset, model, criterion, optimizer, device,
                                  num_epochs_argument, dataset_argument, model_argument, camera_view_argument,
                                  k_fold_argument)
    else:
        train_and_evaluate(train_dataloader, valid_dataloader, model, criterion, optimizer, device, num_epochs_argument,
                           dataset_argument, model_argument, train_size_split_argument, camera_view_argument)

    print_and_log(f'Total training time: {time.time() - total_time} seconds.')


if __name__ == "__main__":
    import sys

    sys.path.append('../')
    import datasets_classes.SplitFileDataset as SplitFileDataset

    main()
