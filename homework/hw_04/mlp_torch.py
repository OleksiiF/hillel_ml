import optuna
import torch
import numpy as np
import torch.nn as nn

import torch.nn.functional as functions
from torch import optim
from torchvision import datasets
import torchvision.transforms as transforms


class NeuralNetwork(nn.Module):
    def __init__(
            self,
            epochs=5,
            dropout=.29,
            learning_rate=0.06097,
            optimizer='SGD',
            activation_func="leaky_relu",
            hidden_layers=1
    ):
        super(NeuralNetwork, self).__init__()
        self.hidden_layer_dimensions = 512
        self.epochs = epochs
        self.hidden_layers = hidden_layers
        self.make_hidden_layers()
        self.dropout = nn.Dropout(dropout)  # dropout layer (p=0.2) prevents overwriting
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = getattr(optim, optimizer)(
            self.parameters(),
            lr=learning_rate
        )
        self.activation_func = getattr(functions, activation_func)

        self.num_workers = 0  # number of subprocesses to use for data loading
        self.batch_size = 20  # how many samples per batch to load

        self.train_data = self.get_data()
        self.test_data = self.get_data(is_train=False)
        self.train_loader = self.get_data_loader(self.train_data)
        self.test_loader = self.get_data_loader(self.test_data)

    def make_hidden_layers(self):
        self.fc1 = nn.Linear(28 * 28, self.hidden_layer_dimensions)
        for i in range(self.hidden_layers):
            tmp = nn.Linear(
                self.hidden_layer_dimensions,
                self.hidden_layer_dimensions
            )  # linear layer (n_hidden -> hidden_2)

        # linear layer (n_hidden -> 10)
        self.last_hidden_layer_output = nn.Linear(
            self.hidden_layer_dimensions,
            10
        )

    def forward(self, input):
        flatten_image_input = input.view(-1, 28 * 28)

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
        return torch.utils.data.DataLoader(
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
                loss = self.loss_function(output, target)  # calculate the loss
                loss.backward()  # backward pass: compute gradient of the loss with respect to model parameters
                self.optimizer.step()  # perform a single optimization step (parameter update)

    def test_model(self):
        test_loss = 0.0
        class_correct = [0.] * 10
        class_total = [0.] * 10
        self.eval()

        for data, target in self.test_loader:
            output = self(data)  # forward pass: compute predicted outputs by passing inputs to the model
            loss = self.loss_function(output, target)  # calculate the loss
            test_loss += loss.item() * data.size(0)  # update test loss
            _, pred = torch.max(output, 1)  # convert output probabilities to predicted class
            correct = np.squeeze(pred.eq(target.data.view_as(pred)))  # compare predictions to true label
            # calculate test accuracy for each object class
            for i in range(self.batch_size):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

        avg_test_loss = test_loss / len(self.test_loader.dataset)
        return avg_test_loss


if __name__ == '__main__':
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
            'high': 5,
        }
        hidden_layers = {
            'name': 'hidden_layers=1',
            'low': 1,
            'high': 3,
        }
        lr = {
            'name': 'learning rate',
            'low': 1e-5,
            'high': 1e-1,
            'step': 1e-5
        }
        optimizer = {
            "name": 'optimizer',
            'choices': ["Adam", "RMSprop", "SGD"]
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
        })
        model.train_model()
        avg_test_loss = model.test_model()

        return avg_test_loss

    study = optuna.create_study(
        study_name="PyTorch MNIST",
        sampler=optuna.samplers.TPESampler(),
        direction='minimize'
    )
    study.optimize(
        objective,
        n_jobs=6, # -1 -> all CPUs
    )

    best_trial = study.best_trial
    for key, value in best_trial.params.items():
        print("{}: {}".format(key, value))
