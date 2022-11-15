import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F


class DataProvider:
    def __init__(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.train_set = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=25, shuffle=True)

        self.test_set = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=25, shuffle=False)


class CNN(nn.Module):
    def __init__(self, num_epocs=100, data_provider=DataProvider()):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__()
        self.data_provider = data_provider
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.out = nn.Linear(in_features=84, out_features=10)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.num_epocs = num_epocs
        self.__all_preds = torch.tensor([]).to(self.device)

        self.to(self.device)

        for name, param in self.state_dict().items():
            print(name, param.size())

    def forward(self, t):
        # Layer 1
        t = t
        # Layer 2
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)  # output shape : (6,14,14)
        # Layer 3
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)  # output shape : (16,5,5)
        # Layer 4
        t = t.reshape(-1, 16 * 5 * 5)
        t = self.fc1(t)
        t = F.relu(t)  # output shape : (1,120)
        # Layer 5
        t = self.fc2(t)
        t = F.relu(t)  # output shape : (1, 84)
        # Layer 6/ Output Layer
        t = self.out(t)  # output shape : (1,10)

        return t

    def train_nn(self):
        total_corrects = []
        total_losses =[]
        for epoch in range(self.num_epocs):
            total_correct = 0
            total_loss = 0

            for imgs, labels in self.data_provider.train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                preds = self(imgs)
                loss = F.cross_entropy(preds, labels)  # Calculate Los
                self.optimizer.zero_grad()
                loss.backward()  # Calculate gradients
                self.optimizer.step()  # Update weights

                total_loss += loss.item()
                total_correct += preds.argmax(dim=1).eq(labels).sum().item()

            print('epoch:', epoch, "total_correct:", total_correct, "loss:", total_loss)
            total_corrects.append(total_correct)
            total_losses.append(total_loss)

        plt.plot(list(range(1, self.num_epocs+1)), total_corrects, label ='Total corrects')
        plt.plot(list(range(1, self.num_epocs+1)), total_losses, '-.', label ='Total losses')
        plt.legend(loc="upper left")

        plt.xlabel("Epochs")
        plt.title('Total corrects and losses')

        plt.savefig('Total corrects and losses.png')
        plt.show(block=False)
        plt.pause(5)
        plt.close()

    @torch.no_grad()
    def test_nn(self):
        for images, labels in self.data_provider.test_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            preds = self(images)
            self.__all_preds = torch.cat((self.__all_preds, preds), dim=0)

        actual_labels = torch.Tensor(self.data_provider.test_set.targets).to(self.device)
        preds_correct = self.__all_preds.argmax(dim=1).eq(actual_labels).to(self.device).sum().item()

        self.accuracy = preds_correct / len(self.data_provider.test_set)

    def plot_confusion_matrix(
            self,
            classes=('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
            normalize=False,
            title='Confusion matrix',
            cmap=plt.cm.Blues
    ):
        cm = confusion_matrix(
            self.data_provider.test_set.targets,
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


if __name__ == '__main__':
    model = CNN()
    model.train_nn()
    model.test_nn()
    print(model.accuracy)
    model.plot_confusion_matrix()
