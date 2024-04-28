import argparse
import os
import time
import logging

import numpy as np
import torch
import torchvision.models as models
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from datasets_classes import FolderDataset

logger = logging.getLogger()

BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
NUM_OF_WORKERS = 4
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
ROTATION_ANGLE = 10


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
            print(f'Batch [{i + 1}/{len(dataloader)}], Train Loss: {running_loss / (i + 1)},'
                  f' Train Acc: {(correct / total)}')
            logger.info(f'Batch [{i + 1}/{len(dataloader)}], Train Loss: {running_loss / (i + 1)},'
                        f' Train Acc: {(correct / total)}')

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

    accuracy = np.mean(np.array(all_labels) == np.array(all_predictions))
    print(f'Test Acc: {accuracy}')
    logger.info(f'Test Acc: {accuracy}')

    cm = confusion_matrix(all_labels, all_predictions)
    print(f'Confusion Matrix:\n {cm}')
    logger.info(f'Confusion Matrix:\n {cm}')


def train_and_evaluate(train_dataloader, test_dataloader, model, criterion, optimizer, device, num_epochs, dataset_name,
                       model_name, eighty_twenty_split, camera_view):
    best_acc = 0.0
    epoch_time = time.time()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        logger.info(f'Epoch {epoch + 1}/{num_epochs}')

        train_loss, train_acc = train_model(model, train_dataloader, criterion, optimizer, device)
        print(f'Train Loss: {train_loss}, Train Acc: {train_acc}')
        logger.info(f'Train Loss: {train_loss}, Train Acc: {train_acc}')

        test_model(model, test_dataloader, device)

        if train_acc > best_acc:
            best_acc = train_acc
            if eighty_twenty_split == 80:
                save_path = f'../data/models/{dataset_name}/{camera_view}/80_20/{model_name}.pth'
            elif eighty_twenty_split == 50:
                save_path = f'../data/models/{dataset_name}/{camera_view}/50_50/{model_name}.pth'
            else:
                raise ValueError('Invalid train split. Please choose between 80 and 50.')

            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))

            torch.save(model.state_dict(), save_path)
            print(f'Saved the new best model to {save_path}')
            logger.info(f'Saved the new best model to {save_path}')

        print(f'Epoch time: {time.time() - epoch_time} seconds.')
        logger.info(f'Epoch time: {time.time() - epoch_time} seconds.')
        epoch_time = time.time()

    print(f'Best Train Acc: {best_acc}')
    logger.info(f'Best Train Acc: {best_acc}')


def main():
    # parsing the arguments for the script from the command line
    parser = argparse.ArgumentParser(description='Train a model on the PKLot dataset.')

    parser.add_argument('--dataset', type=str,
                        help='The dataset to use. Choose between pklot, cnrpark, acmps, acpds, spkl')
    parser.add_argument('--camera_view', type=str,
                        help='For the PKLot dataset, choose between puc, ufpr04, ufpr05, and all, '
                             'for CNRPark choose between cnrpark (camera A and B), cnrpark_ext (camera 0-9 and cnrpark)'
                             ' and all, '
                             'for ACPDS default and only option is all, '
                             'for ACPMS default and only option is all, '
                             'for SPKL default and only option is all.')
    parser.add_argument('--model', type=str,
                        help='The model to use. Choose between alexnet, mobilenet, squeezenet and shufflenet.')
    parser.add_argument('--train_split', type=int,
                        help='The percentage of the dataset to use for training. The rest will be used for testing.',
                        default=80)

    # parse the arguments and store them in args variable, then assign them to the corresponding variables
    args = parser.parse_args()
    dataset_argument = args.dataset
    camera_view_argument = args.camera_view
    test_size_split_argument = args.train_split
    model_argument = args.model

    # print all the arguments
    print(f'Dataset: {dataset_argument}')
    print(f'Camera view: {camera_view_argument}')
    print(f'Test size split: {test_size_split_argument}')
    print(f'Model: {model_argument}')
    logger.info(f'Dataset: {dataset_argument}')
    logger.info(f'Camera view: {camera_view_argument}')
    logger.info(f'Test size split: {test_size_split_argument}')
    logger.info(f'Model: {model_argument}')

    # set the log file name based on the arguments
    if test_size_split_argument == 80:
        log_file_name = (f'../data/logs/{test_size_split_argument}_{100 - test_size_split_argument}_split/'
                         f'{dataset_argument}/{camera_view_argument}/{model_argument}.txt')
    elif test_size_split_argument == 50:
        log_file_name = (f'../data/logs/{test_size_split_argument}_{100 - test_size_split_argument}_split/'
                         f'{dataset_argument}/{camera_view_argument}/{model_argument}.txt')
    else:
        raise ValueError('Invalid train split. Please choose between 80 and 50.')

    if not os.path.exists(os.path.dirname(log_file_name)):
        os.makedirs(os.path.dirname(log_file_name))

    if os.path.exists(log_file_name):
        print(f'Log file {log_file_name} already exists. Deleting it.')
        os.remove(log_file_name)

    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file_name)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    print(f'Logging to {log_file_name}')

    # set the dataset directory based on the arguments
    if dataset_argument == 'pklot':
        dataset_name = 'pklot'
        if camera_view_argument == 'puc':
            if test_size_split_argument == 80:
                dataset_dir = f'../data/datasets/pklot_puc/80_20/'
            elif test_size_split_argument == 50:
                dataset_dir = f'../data/datasets/pklot_puc/50_50/'
            else:
                raise ValueError('Invalid train split. Please choose between 80 and 50.')
        elif camera_view_argument == 'ufpr04':
            if test_size_split_argument == 80:
                dataset_dir = f'../data/datasets/pklot_ufpr04/80_20/'
            elif test_size_split_argument == 50:
                dataset_dir = f'../data/datasets/pklot_ufpr04/50_50/'
            else:
                raise ValueError('Invalid train split. Please choose between 80 and 50.')
        elif camera_view_argument == 'ufpr05':
            if test_size_split_argument == 80:
                dataset_dir = f'../data/datasets/pklot_ufpr05/80_20/'
            elif test_size_split_argument == 50:
                dataset_dir = f'../data/datasets/pklot_ufpr05/50_50/'
            else:
                raise ValueError('Invalid train split. Please choose between 80 and 50.')
        elif camera_view_argument == 'all':
            if test_size_split_argument == 80:
                dataset_dir = f'../data/datasets/pklot_all/80_20/'
            elif test_size_split_argument == 50:
                dataset_dir = f'../data/datasets/pklot_all/50_50/'
            else:
                raise ValueError('Invalid train split. Please choose between 80 and 50.')
        else:
            raise ValueError('Invalid camera view. Please choose between PUC, UFPR04, UFPR05, and all.')

    elif dataset_argument == 'cnrpark':
        dataset_name = 'cnrpark'
        if camera_view_argument == 'cnrpark':
            if test_size_split_argument == 80:
                dataset_dir = f'../data/datasets/cnrpark/80_20/'
            elif test_size_split_argument == 50:
                dataset_dir = f'../data/datasets/cnrpark/50_50/'
            else:
                raise ValueError('Invalid train split. Please choose between 80 and 50.')
        elif camera_view_argument == 'cnrpark_ext':
            if test_size_split_argument == 80:
                dataset_dir = f'../data/datasets/cnrpark_ext/80_20/'
            elif test_size_split_argument == 50:
                dataset_dir = f'../data/datasets/cnrpark_ext/50_50/'
            else:
                raise ValueError('Invalid train split. Please choose between 80 and 50.')
        else:
            raise ValueError('Invalid camera view. Please choose between CNRPark, CNRParkExt, and all.')

    elif dataset_argument == 'acmps':
        dataset_name = 'acmps'
        dataset_dir = f'../data/datasets/acmps/all'
    elif dataset_argument == 'acpds':
        dataset_name = 'acpds'
        dataset_dir = f'../data/datasets/acpds/all'
    elif dataset_argument == 'spkl':
        dataset_name = 'spkl'
        dataset_dir = f'../data/datasets/spkl/all'
    else:
        raise ValueError('Invalid dataset. Please choose between pklot, cnrpark, acmps, acpds, spkl.')

    # transformations for the dataset, different for training and testing
    transform_train = transforms.Compose([
        transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(ROTATION_ANGLE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset_dir = dataset_dir + 'train/'
    test_dataset_dir = dataset_dir + 'test/'

    # create the dataset and dataloader
    train_dataset = FolderDataset.FolderDataset(train_dataset_dir, transform=transform_train)
    test_dataset = FolderDataset.FolderDataset(test_dataset_dir, transform=transform_test)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_OF_WORKERS,
                                  pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_OF_WORKERS,
                                 pin_memory=True)

    # set the device to cuda if available, otherwise to cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create the model, optimizer, and criterion
    if model_argument == 'alexnet':
        model = models.alexnet(weights=None)
    elif model_argument == 'mobilenet':
        model = models.mobilenet_v2(weights=None)
    elif model_argument == 'squeezenet':
        model = models.squeezenet1_0(weights=None)
    elif model_argument == 'shufflenet':
        model = models.shufflenet_v2_x0_5(weights=None)
    else:
        raise ValueError(
            'Invalid model. Please choose between alexnet, mobilenet, squeezenet, shufflenet.')

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    criterion = nn.CrossEntropyLoss()

    # train the model
    print('Training the model...')
    logger.info('Training the model...')
    total_time = time.time()

    train_and_evaluate(train_dataloader, test_dataloader, model, criterion, optimizer, device, NUM_EPOCHS,
                       dataset_name, model_argument, test_size_split_argument, camera_view_argument)

    print(f'Total training time: {time.time() - total_time} seconds.')
    logger.info(f'Total training time: {time.time() - total_time} seconds.')


if __name__ == "__main__":
    main()
