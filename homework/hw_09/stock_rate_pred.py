import math
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from torch import nn, from_numpy, Tensor, zeros, optim, save


class Dataset:
    def __init__(self, test_size=0.1, describe_data=True):
        self.data = self.get_data("AAPL.csv")

        if describe_data:
            self.brief_data()

        self.test_size = test_size
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.data.Close = self.scaler.fit_transform(self.data.Close.values.reshape(-1, 1))

    def get_data(self, name):
        return pd.read_csv(
            name,
            index_col='Date',
            parse_dates=True,
            usecols=[
                'Date',
                'Close',
                # "Volume"
            ],
            na_values=['nan']
        )

    def load_data(self, look_back):
        data_raw = self.data.values
        data = []

        for index in range(len(data_raw) - look_back):
            data.append(data_raw[index: index + look_back])

        data = np.array(data)
        test_set_size = int(np.round(self.test_size * data.shape[0]))
        train_set_size = data.shape[0] - (test_set_size)

        x_train = data[:train_set_size, :-1, :]
        y_train = data[:train_set_size, -1, :]

        x_test = data[train_set_size:, :-1]
        y_test = data[train_set_size:, -1, :]

        x_train = from_numpy(x_train).type(Tensor)
        x_test = from_numpy(x_test).type(Tensor)
        y_train = from_numpy(y_train).type(Tensor)
        y_test = from_numpy(y_test).type(Tensor)

        return x_train, y_train, x_test, y_test

    def brief_data(self):
        print(self.data.head)
        plt.figure(figsize=(15, 6))
        plt.grid()
        plt.plot(self.data.Close)
        plt.ylabel("Stock price")
        plt.xlabel("Date")
        plt.title("AAPL Stock")
        plt.savefig('aapl_brief_data.png')
        plt.show(block=False)
        plt.pause(5)
        plt.close()


class LSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, output_dim=1, epochs=20, lr=0.01, look_back=100):
        super().__init__()
        self.dataset = Dataset()
        self.x_train, self.y_train, self.x_test, self.y_test = self.dataset.load_data(look_back=look_back)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.epochs = epochs
        self.loss_fn = nn.MSELoss()
        self.lr = lr
        self.optimiser = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        h0 = zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        return self.fc(out[:, -1, :])

    def train_nn(self, save_model=True):
        hist = np.zeros(self.epochs)

        for epoch in range(1, self.epochs+1):
            self.y_train_pred = self(self.x_train)

            if save_model:
                dir = 'rnn_models'
                try:
                    os.mkdir(dir)
                except FileExistsError:
                    pass

                save(self, f"{dir}/model_{epoch}.pt")
                print('Model trained and saved')

            loss = self.loss_fn(self.y_train_pred, self.y_train)
            if epoch % 10 == 0 and epoch != 0:
                print("Epoch ", epoch, "MSE: ", loss.item())

            hist[epoch-1] = loss.item()

            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

        plt.plot(hist, label="Training loss")
        plt.legend()
        plt.savefig('training_loss.png')
        plt.show(block=False)
        plt.pause(5)
        plt.close()

    def predict(self):
        y_test_pred = self(self.x_test)
        # invert predictions
        y_train_pred = self.dataset.scaler.inverse_transform(self.y_train_pred.detach().numpy())
        y_train = self.dataset.scaler.inverse_transform(self.y_train.detach().numpy())
        y_test_pred = self.dataset.scaler.inverse_transform(y_test_pred.detach().numpy())
        y_test = self.dataset.scaler.inverse_transform(self.y_test.detach().numpy())

        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(y_train[:, 0], y_train_pred[:, 0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(y_test[:, 0], y_test_pred[:, 0]))
        print('Test Score: %.2f RMSE' % (testScore))

        # Visualising the results
        figure, axes = plt.subplots(figsize=(15, 6))
        axes.xaxis_date()

        axes.plot(
            self.dataset.data[len(self.dataset.data) - len(y_test):].index,
            y_test,
            color='red',
            label='Real AAPL stock price'
        )
        axes.plot(
            self.dataset.data[len(self.dataset.data) - len(y_test):].index,
            y_test_pred,
            color='blue',
            label='Predicted AAPL stock price'
        )

        plt.title('AAPL stock price prediction')
        plt.xlabel('Time')
        plt.ylabel('AAPL stock price')
        plt.legend()
        plt.savefig('aapl_pred.png')
        plt.show(block=False)
        plt.pause(5)
        plt.close()


model = LSTM()
model.train_nn()
model.predict()
