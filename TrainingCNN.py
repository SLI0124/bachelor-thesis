import os
import sys
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

DATA_DIR = "./data/ai_folder/"
TRAIN_DIR = "./data/ai_folder/training/"
TEST_DIR = "./data/ai_folder/testing/"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = [i.name.split('/')[-1] for i in os.scandir(TRAIN_DIR) if i.is_dir()]


class ParkingLotCNNModel(nn.Module):
    def __init__(self, num_classes=6):
        super(ParkingLotCNNModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.relu1 = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(12, 20, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(20, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()

        self.fc = nn.Linear(75 * 75 * 32, num_classes)

    def forward(self, input):
        output = self.relu1(self.bn1(self.conv1(input)))
        output = self.pool(output)
        output = self.relu2(self.conv2(output))
        output = self.relu3(self.bn3(self.conv3(output)))
        output = output.view(-1, 32 * 75 * 75)
        output = self.fc(output)
        return output


def train_model(model, train_loader, test_loader, device, num_epochs, learning_rate):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    train_count, test_count = len(train_loader.dataset), len(test_loader.dataset)
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_accuracy, train_loss = 0.0, 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, prediction = torch.max(outputs.data, 1)
            train_accuracy += int(torch.sum(prediction == labels.data))

        train_accuracy = train_accuracy / train_count
        train_loss = train_loss / train_count

        model.eval()
        test_accuracy = 0.0

        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, prediction = torch.max(outputs.data, 1)
                test_accuracy += int(torch.sum(prediction == labels.data))

        test_accuracy = test_accuracy / test_count

        print(
            f'Epoch: {epoch} Train Loss: {train_loss} Train Accuracy: {train_accuracy} Test Accuracy: {test_accuracy}')

        if test_accuracy > best_accuracy:
            torch.save(model.state_dict(), 'best_checkpoint.model')
            best_accuracy = test_accuracy


def show_image(model, transformer, num_images=25):
    test_data = ImageFolder(TEST_DIR, transform=transformer)
    fig = plt.figure(figsize=(10, 10))

    for i in range(1, num_images + 1):
        img_num = np.random.randint(0, len(test_data))
        img = Variable(test_data[img_num][0].unsqueeze(0).to(DEVICE))
        output = model(img)
        _, prediction = torch.max(output.data, 1)
        img = img.cpu().squeeze(0).detach().numpy().transpose((1, 2, 0))
        img = img / 2 + 0.5
        img = np.clip(img, 0, 1)
        fig.add_subplot(5, 5, i)
        plt.imshow(img)
        plt.title(f"{classes[prediction]}")
        plt.axis('off')

    plt.show()


def main(args):
    print(torch.__version__)

    transformer = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_loader = DataLoader(ImageFolder(TRAIN_DIR, transform=transformer), batch_size=64, shuffle=True)
    test_loader = DataLoader(ImageFolder(TEST_DIR, transform=transformer), batch_size=32, shuffle=True)

    print(f"classes are: {classes}")

    model = ParkingLotCNNModel(num_classes=len(classes))
    train_model(model, train_loader, test_loader, DEVICE, num_epochs=10, learning_rate=0.001)

    print("Done training")

    show_image(model, transformer)


if __name__ == '__main__':
    main(sys.argv[1:])
