import pandas as pd
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self):
        self.train_data = self.get_data("AAPL-train.csv")
        self.test_data = self.get_data("AAPL-test.csv")
        self.brief_data()

    def get_data(self, name):
        return pd.read_csv(
            name,
            index_col='Date',
            parse_dates=True,
            usecols=['Date', 'Close', "Volume"],
            na_values=['nan']
        )

    def brief_data(self):
        print(self.train_data.head)
        fig, axs = plt.subplots(2)
        fig.set_figwidth(15)
        axs[0].plot(self.train_data.Close)
        axs[1].plot(self.train_data.Volume)
        axs[0].grid()
        axs[1].grid()
        axs[0].set(xlabel='Date', ylabel='Stock price')
        axs[1].set(xlabel='Date', ylabel='Stock volume')
        axs[0].set_title("AAPL stock price")
        axs[1].set_title("AAPL stock volume")
        plt.subplots_adjust(hspace=0.5)
        plt.show()


dataset = Dataset()
