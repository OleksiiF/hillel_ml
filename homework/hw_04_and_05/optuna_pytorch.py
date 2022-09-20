from contextlib import contextmanager

import numpy as np
import optuna
from torch import (
    optim,
    cuda,
    device,
    utils,
    rand,
    pca_lowrank,
    max as torch_max,
    nn as torch_nn,

)
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


@contextmanager
def tb_writer():
    """
    return tenserboard writer
    """
    writer = SummaryWriter()
    try:
        yield writer

    finally:
        writer.close()


class NeuralNetwork(torch_nn.Module):
    def __init__(
            self,
            epochs=27,
            dropout=.22,
            learning_rate=0.03593,
            optimizer='Adagrad',
            activation_func="glu",
            hidden_layers=2,
            batch_size=20,
            hidden_layer_dimensions=420,
    ):
        super(NeuralNetwork, self).__init__()
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.device = device("cpu")
        self.to(self.device)
        # hyperparams
        self.hidden_layer_dimensions = hidden_layer_dimensions
        self.epochs = epochs
        self.hidden_layers = hidden_layers
        self.make_hidden_layers()
        self.dropout = torch_nn.Dropout(dropout)  # prevents overwriting
        self.loss_function = torch_nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.__optimizer = optimizer
        self.optimizer = getattr(optim, self.__optimizer)(
            self.parameters(),
            lr=self.learning_rate
        )
        self.__activation_func = activation_func
        self.activation_func = getattr(
            torch_nn.functional,
            self.__activation_func
        )
        # load data
        self.num_workers = 0  # number of subprocesses to use for data loading
        self.batch_size = batch_size  # how many samples per batch to load
        self.train_data = self.get_data()
        self.test_data = self.get_data(is_train=False)
        self.train_loader = self.get_data_loader(self.train_data)
        self.test_loader = self.get_data_loader(self.test_data)

        self.test_accuracy = 0.0

    def make_hidden_layers(self):
        self.fc1 = torch_nn.Linear(28 * 28, self.hidden_layer_dimensions)
        self.fc1.to(self.device)
        for i in range(self.hidden_layers):
            tmp = torch_nn.Linear(
                self.hidden_layer_dimensions,
                self.hidden_layer_dimensions
            )  # linear layer (hidden -> hidden_n)
            tmp.to(self.device)

        # linear layer (n_hidden -> 10)
        self.last_hidden_layer_output = torch_nn.Linear(
            self.hidden_layer_dimensions,
            10
        )

    def forward(self, input):
        flatten_image_input = input.view(-1, 28 * 28).to(self.device)
        output = self.activation_func(
            self.fc1(flatten_image_input)
        )  # add hidden layer, with relu activation function
        
        return output

    def get_data(self, is_train=True):
        mnist_params = {
            'root': 'data',
            'train': is_train,
            'download': True,
            'transform': transforms.ToTensor()  # convert data to torch.FloatTensor
        }

        return datasets.MNIST(**mnist_params)

    def get_data_loader(self, data):
        return utils.data.DataLoader(
            data,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def train_model(self):
        self.train()

        for epoch in range(self.epochs):
            # train the model
            for data, target in self.train_loader:
                self.optimizer.zero_grad()  # clear the gradients of all optimized variables
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self(data)
                loss = self.loss_function(
                    output.to(self.device),
                    target.to(self.device)
                )  # calculate the loss
                loss.backward()  # backward pass: compute gradient of the loss with respect to model parameters
                self.optimizer.step()  # perform a single optimization step (parameter update)

    def pca_worker(self) -> None:
        """
        Draws visualization of the explained variance
        depending on components
        """
        train_data_array = self.train_data.data.numpy()
        nsamples, nx, ny = train_data_array.shape
        d2_train_dataset = train_data_array.reshape((nsamples, nx * ny))

        pca = PCA().fit(d2_train_dataset)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of components')
        plt.ylabel('Cumulative explained variance')
        plt.grid(True)
        plt.show()

    def write_tb_data(self) -> None:
        """
        will write data to tensorboard files,
        which will be used to visualize
        an effectivity of the hyper params
        """
        hyper_params = ({
            "learning rate": self.learning_rate,
            "hidden layer dimensions": self.hidden_layer_dimensions,
            "hidden layers": self.hidden_layers,
            "optimizer": self.__optimizer,
            "activation func": self.__activation_func,
            "dropout": self.dropout.p,
            "epochs": self.epochs,
        },
            # {"test loss": test_loss},
            {"test accuracy": self.test_accuracy},
        )

        with tb_writer() as writer:
            writer.add_hparams(*hyper_params)

    def test_model(self) -> None:
        test_loss_counter = 0.0
        class_correct = [0.] * 10
        class_total = [0.] * 10
        self.eval()

        for data, target in self.test_loader:
            output = self(data)  # forward pass: compute predicted outputs by passing inputs to the model
            loss = self.loss_function(output, target.to(self.device))  # calculate the loss
            test_loss_counter += loss.item() * data.size(0)  # update test loss
            _, pred = torch_max(output, 1)  # convert output probabilities to predicted class

            correct = np.squeeze(
                pred.eq(
                    target.data.view_as(
                        pred.to(self.device)
                    ).to(self.device)
                ).to(self.device)
            )  # compare predictions to true label
            # calculate test accuracy for each object class
            for i in range(self.batch_size):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

        self.test_accuracy = sum(class_correct) / sum(class_total)
        # test_loss = test_loss_counter / len(self.test_loader.dataset)


class OptunaSettings:
    dropout = {
        'name': 'dropout',
        'low': 0.1,
        'high': 0.3,
        'step': 0.01
    }
    epochs = {
        'name': 'epochs',
        'low': 1,
        'high': 30,
    }
    hidden_layers = {
        'name': 'hidden layers',
        'low': 1,
        'high': 3,
    }
    hidden_layer_dimensions = {
        'name': 'hidden layer dimensions',
        'low': 200,
        'high': 500,
        'step': 20
    }
    lr = {
        'name': 'learning rate',
        'low': 1e-5,
        'high': 1e-1,
        'step': 1e-5
    }
    optimizer = {
        "name": 'optimizer',
        'choices': [
            "Adam",
            "RMSprop",
            "SGD",
            "Adagrad",
            "Adadelta",
            "Adamax",
            "ASGD"
        ]
    }
    activation_func = {
        "name": 'activation_func',
        'choices': [
            'sigmoid',
            'logsigmoid',
            'softsign',
            'hardtanh',
            'hardshrink',
            'log_softmax',
            'hardsigmoid',
            'normalize',
            'mish',
            'leaky_relu',
            'gelu',
            'relu',
            'glu',
            'elu',
            'selu',
            'celu',
            'leaky_relu',
            'rrelu',
        ]
    }


def objective(trial):
    model = NeuralNetwork(**{
        'epochs': trial.suggest_int(**OptunaSettings.epochs),
        'hidden_layers': trial.suggest_int(**OptunaSettings.hidden_layers),
        'dropout': trial.suggest_float(**OptunaSettings.dropout),
        'learning_rate': trial.suggest_float(**OptunaSettings.lr),
        'optimizer': trial.suggest_categorical(**OptunaSettings.optimizer),
        'activation_func': trial.suggest_categorical(**OptunaSettings.activation_func),
        'hidden_layer_dimensions': trial.suggest_int(**OptunaSettings.hidden_layer_dimensions),
    })
    model.pca_worker()
    # model.train_model()
    # model.test_model()
    # model.write_tb_data()

    return model.test_accuracy


if __name__ == '__main__':
    if cuda.is_available():
        print("CUDA is available")
        print(f"GPU name is {cuda.get_device_name(0)}")

    study = optuna.create_study(
        study_name="PyTorch MNIST",
        sampler=optuna.samplers.RandomSampler(),
        direction='maximize',
    )
    study.optimize(objective, n_jobs=1)
    # after study -> tensorboard --logdir runs
    # go to http://localhost:6006
