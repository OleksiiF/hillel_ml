import itertools

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from settings import (
    RANDOM_ROTATION,
    EPOCHS,
    LR,
    KERNEL_SIZE,
    PADDING,
    OPTIMIZER,
    STRIDE
)


class DataProvider:
    def __init__(self, random_rotation=10):
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.RandomRotation(random_rotation),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.train_dataset = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=train_transform
        )
        self.test_dataset = datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=val_transform
        )
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=4, shuffle=True)
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=4,
            shuffle=False
        )


class CNN(nn.Module):
    def __init__(
            self,
            num_classes=10,
            data_provider=DataProvider(),
            num_epocs=100,
            lr=0.1,
            weight_decay=5e-4,
            padding=1,
            kernel_size=3,
            stride=1,
            optimizer='Adam'
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride
            ),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(in_features=4 * 4 * 128, out_features=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = getattr(optim, optimizer)(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.num_epocs = num_epocs
        self.data_provider = data_provider
        self.__all_preds = torch.tensor([]).to(self.device)
        self.to(self.device)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_nn(self):
        for epoch in range(self.num_epocs):
            total_correct = 0
            total_loss = 0
            for imgs, labels in self.data_provider.train_dataloader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                output = self(imgs)
                loss = self.criterion(output, labels)  # Calculate Loss
                loss.backward()  # Calculate gradients
                self.optimizer.step()  # Update weights

                total_loss += loss.item()
                total_correct += output.argmax(dim=1).eq(labels).sum().item()

            print('epoch:', epoch, "total_correct:", total_correct, "loss:", total_loss)

    @torch.no_grad()
    def test_nn(self):
        for images, labels in self.data_provider.test_dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            preds = self(images)
            self.__all_preds = torch.cat((self.__all_preds, preds), dim=0)

        actual_labels = torch.Tensor(self.data_provider.test_dataset.targets).to(self.device)
        preds_correct = self.__all_preds.argmax(dim=1).eq(actual_labels).to(self.device).sum().item()

        self.accuracy = preds_correct / len(self.data_provider.test_dataset)


    def plot_confusion_matrix(
            self,
            classes=('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
            normalize=False,
            title='Confusion matrix',
            cmap=plt.cm.Blues
    ):
        cm = confusion_matrix(
            self.data_provider.test_dataset.targets,
            self.__all_preds.cpu().argmax(dim=1)
        )
        print(cm)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        plt.savefig('confusion_matrix.png')
        plt.show(block=False)
        plt.pause(5)
        plt.close()


def objective(trial):
    data_provider = DataProvider(**{
        'random_rotation': trial.suggest_int(**RANDOM_ROTATION),
    })
    model = CNN(**{
        'data_provider': data_provider,
        'num_epocs': trial.suggest_int(**EPOCHS),
        'lr': trial.suggest_float(**LR),
        'kernel_size': trial.suggest_int(**KERNEL_SIZE),
        'padding': trial.suggest_int(**PADDING),
        'stride': trial.suggest_int(**STRIDE),
        'optimizer': trial.suggest_categorical(**OPTIMIZER),
    })
    model.train_nn()
    model.test_nn()
    print(model.accuracy)

    return model.test_accuracy


if __name__ == '__main__':
    model = CNN()
    model.train_nn()
    model.test_nn()
    print(model.accuracy)
    model.plot_confusion_matrix()

    # study = optuna.create_study(
    #     study_name="PyTorch MNIST",
    #     # sampler=optuna.samplers.RandomSampler(),
    #     sampler=optuna.samplers.TPESampler(),
    #     direction='maximize',
    # )
    # study.optimize(
    #     objective,
    #     n_jobs=-1,
    #     n_trials=100
    # )
    # # after study -> tensorboard --logdir runs
    # # go to http://localhost:6006
