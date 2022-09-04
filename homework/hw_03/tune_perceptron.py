#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from logging.handlers import TimedRotatingFileHandler

import scipy
import streamlit as st
import numpy
import matplotlib.pyplot as plt


def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance


class Settings:
    # neural network configuration
    LEARNING_RATE = 0.0175
    INPUT_NODES = 784
    OUTPUT_NODES = 10
    HIDDEN_NODES = 200
    EPOCHS = 4

    # autotuner configuration
    DIGITS_AFTER_POINT = 4
    STEP_LEARNING_RATE = 10 ** -DIGITS_AFTER_POINT
    STEP_HIDDEN_NODES = int(HIDDEN_NODES * 0.3)
    AUTOTUNE_ATTEMPTS = 100
    FAILED_PERF_IN_A_ROW = 3

    # logging configuration
    DEFAULT_LOG_LEVEL = logging.INFO
    LOG_FORMAT = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    LOG_FILENAME = "log"
    LOG_TIMEFRAME = "H"  # 'midnight' value is available
    LOG_ROTATION_INTERVAL = 24
    LOG_BACKUP_QNT = 7

    @staticmethod
    def get_current_nn_settings() -> str:
        atributes = (
            'EPOCHS',
            'LEARNING_RATE',
            'INPUT_NODES',
            'HIDDEN_NODES',
            'OUTPUT_NODES',
        )

        return '\n'.join([
            f"{attribute}={getattr(Settings, attribute)}"
            for attribute in atributes
        ])


@singleton
class Logger:
    def __init__(self):
        logging.basicConfig(
            level=Settings.DEFAULT_LOG_LEVEL,
            format=Settings.LOG_FORMAT,
            datefmt="%m-%d %H:%M",
            filename=Settings.LOG_FILENAME,
        )
        console_handler = logging.StreamHandler()
        file_handler = TimedRotatingFileHandler(
            filename=Settings.LOG_FILENAME,
            when=Settings.LOG_TIMEFRAME,
            interval=Settings.LOG_ROTATION_INTERVAL,
            backupCount=Settings.LOG_BACKUP_QNT,
        )
        file_handler.setFormatter(
            logging.Formatter(Settings.LOG_FORMAT)
        )
        file_handler.suffix = '%Y-%m-%d_%H_%M_%S.log'
        self.__logger = logging.getLogger()
        self.__logger.handlers.clear()
        self.__logger.propagate = False
        self.__logger.addHandler(console_handler)
        self.__logger.addHandler(file_handler)

    def msg_handler(self, msg, st_log=True, python_log=True):
        if python_log:
            self.__logger.info(msg)
        if st_log:
            st.write(msg)

logger = Logger()


class NeuralNetwork:
    def __init__(
            self,
            learning_rate: float,
            hidden_nodes: int,
            epochs: int,
            input_nodes: int=Settings.INPUT_NODES,
            output_nodes: int=Settings.OUTPUT_NODES
    ):
        self.inodes: int = input_nodes
        self.hnodes: int = hidden_nodes
        self.onodes: int = output_nodes
        self.epochs: int = epochs
        self.learning_rate: float = learning_rate
        self.logger = logger
        self.train_data: list = self.get_data(file_name="mnist_train.csv")
        self.test_data: list = self.get_data(file_name="mnist_test.csv")
        self.wih = numpy.random.normal(
            0.0,
            pow(self.inodes, -0.5),
            (self.hnodes, self.inodes)
        )
        self.who = numpy.random.normal(
            0.0,
            pow(self.hnodes, -0.5),
            (self.onodes, self.hnodes)
        )
        self.activation_function = lambda x: scipy.special.expit(x)  # sigmoid
        self.test_scorecards = []
        self.train_scorecards = []
        self.logger.msg_handler('Neural network initiated')

    def _train(self, inputs_list, targets_list) -> None:
        # transform the inputs list into 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)  # inputs of the output layer
        final_outputs = self.activation_function(final_inputs)  # outputs of the output layer
        output_errors = targets - final_outputs  # error function (target - calculation)
        hidden_errors = numpy.dot(self.who.T, output_errors)  # spreads error between hidden nodes
        self.who += self.learning_rate * numpy.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)),
            numpy.transpose(hidden_outputs)
        )  # weights calculation between hiddens and outputs
        self.wih += self.learning_rate * numpy.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
            numpy.transpose(inputs)
        )  # weights calculation between inputs and hiddens.

    def _query(self, inputs_list):
        """
        Query to the neural network
        """
        inputs = numpy.array(inputs_list, ndmin=2).T  # transform the inputs list into 2d array
        hidden_inputs = numpy.dot(self.wih, inputs)  # inputs of hiddens
        hidden_outputs = self.activation_function(hidden_inputs)  # outputs of hiddens
        final_inputs = numpy.dot(self.who, hidden_outputs)  # inputs of the output layer
        final_outputs = self.activation_function(final_inputs)  # outputs of the output layer

        return final_outputs

    def get_data(self, file_name: str) -> list:
        with open(file_name, 'r') as fh:
            return fh.readlines()

    def nn_train(self) -> None:
        for e in range(self.epochs):
            self.logger.msg_handler('New epoch started')
            for record in self.train_data:
                all_values = record.split(',')
                inputs = (numpy.asfarray(
                    all_values[1:]) / 255.0 * 0.99) + 0.01  # scale and shift of src data
                output_targets = numpy.zeros(Settings.OUTPUT_NODES) + 0.01
                output_targets[int(all_values[0])] = 0.99
                self._train(inputs, output_targets)

        self.logger.msg_handler('Neural network trained')

    def _test(self, data: list, scorecards: list) -> None:
        for record in data:
            all_values: list = record.split(',')
            correct_label = int(all_values[0])
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01  # scale and shift of src data
            outputs = self._query(inputs)  # response from nn
            label = numpy.argmax(outputs)
            scorecards.append(1 if label == correct_label else 0)

    def nn_test(self) -> None:
        """
        Validation of neural network
        """
        self._test(self.test_data, self.test_scorecards)
        self._test(self.train_data, self.train_scorecards)

    def get_scorecard(self) -> tuple[float, float]:
        """
        Score calculation
        """
        train_scorecard_array = numpy.asarray(self.train_scorecards)
        train_performance = train_scorecard_array.sum() / train_scorecard_array.size

        test_scorecard_array = numpy.asarray(self.test_scorecards)
        test_performance = test_scorecard_array.sum() / test_scorecard_array.size

        return train_performance, test_performance


class NNTuner:
    def __init__(self):
        self.logger = logger

    def tune_learning_rate(self) -> tuple[list, list, list]:
        train_performances = []
        test_performances = []
        learning_rates = []
        etalon_performance = 0
        failed_perf_in_a_row = 0

        while Settings.AUTOTUNE_ATTEMPTS > 0:
            neural_network = NeuralNetwork(
                epochs=Settings.EPOCHS,
                learning_rate=Settings.LEARNING_RATE,
                hidden_nodes=Settings.HIDDEN_NODES
            )
            neural_network.nn_train()
            neural_network.nn_test()

            self.logger.msg_handler(Settings.get_current_nn_settings())
            tmp_train_performance, tmp_test_performance = neural_network.get_scorecard()
            self.logger.msg_handler(
                f"{tmp_train_performance=}, "
                f"{tmp_test_performance=}"
            )
            train_performances.append(tmp_train_performance)
            test_performances.append(tmp_test_performance)
            learning_rates.append(Settings.LEARNING_RATE)

            if etalon_performance > tmp_test_performance:  # performance is not increased
                failed_perf_in_a_row += 1

                if failed_perf_in_a_row >= Settings.FAILED_PERF_IN_A_ROW:
                    break

                if failed_perf_in_a_row == 2:
                    # give one more chance, to avoid the random mistake
                    Settings.DIGITS_AFTER_POINT += 1
                    Settings.LEARNING_RATE = round(
                        Settings.LEARNING_RATE - 2 * Settings.STEP_LEARNING_RATE,
                        Settings.DIGITS_AFTER_POINT
                    )
                    Settings.STEP_LEARNING_RATE = round(
                        Settings.STEP_LEARNING_RATE / 10,
                        Settings.DIGITS_AFTER_POINT
                    )
            else:
                failed_perf_in_a_row = 0
                etalon_performance = tmp_test_performance

            Settings.LEARNING_RATE = round(
                Settings.LEARNING_RATE + Settings.STEP_LEARNING_RATE,
                Settings.DIGITS_AFTER_POINT
            )

            Settings.AUTOTUNE_ATTEMPTS -= 1

        return train_performances, test_performances, learning_rates

    def tune_epochs(self) -> tuple[list, list, list]:
        failed_perf_in_a_row = 0
        etalon_performance = 0
        test_performances = []
        train_performances = []
        epochs = []

        while True:
            neural_network = NeuralNetwork(
                epochs=Settings.EPOCHS,
                learning_rate=Settings.LEARNING_RATE,
                hidden_nodes=Settings.HIDDEN_NODES
            )
            neural_network.nn_train()
            neural_network.nn_test()

            self.logger.msg_handler(Settings.get_current_nn_settings())
            tmp_train_performance, tmp_test_performance = neural_network.get_scorecard()
            self.logger.msg_handler(
                f"{tmp_train_performance=}, "
                f"{tmp_test_performance=}"
            )
            train_performances.append(tmp_train_performance)
            test_performances.append(tmp_test_performance)
            epochs.append(Settings.EPOCHS)

            if tmp_test_performance > etalon_performance:
                etalon_performance = tmp_test_performance
                failed_perf_in_a_row = 0

            else:
                failed_perf_in_a_row += 1

                if failed_perf_in_a_row >= Settings.FAILED_PERF_IN_A_ROW:
                    break

            Settings.EPOCHS += 1

        return train_performances, test_performances, epochs

    def tune_hiddens(self) -> tuple[list, list, list]:
        test_performances = []
        train_performances = []
        hidden_nodes = []
        etalon_performance = 0
        failed_perf_in_a_row = 0

        while True:
            neural_network = NeuralNetwork(
                epochs=Settings.EPOCHS,
                learning_rate=Settings.LEARNING_RATE,
                hidden_nodes=Settings.HIDDEN_NODES
            )
            neural_network.nn_train()
            neural_network.nn_test()

            self.logger.msg_handler(Settings.get_current_nn_settings())
            tmp_train_performance, tmp_test_performance = neural_network.get_scorecard()
            self.logger.msg_handler(
                f"{tmp_train_performance=}, "
                f"{tmp_test_performance=}"
            )
            train_performances.append(tmp_train_performance)
            test_performances.append(tmp_test_performance)
            hidden_nodes.append(Settings.HIDDEN_NODES)

            if tmp_test_performance > etalon_performance:
                etalon_performance = tmp_test_performance
                failed_perf_in_a_row = 0

            else:
                failed_perf_in_a_row += 1

                if failed_perf_in_a_row >= Settings.FAILED_PERF_IN_A_ROW:
                    break

            Settings.HIDDEN_NODES += Settings.STEP_HIDDEN_NODES

        return train_performances, test_performances, hidden_nodes

    def _drawer(self, x, y_train, y_test, title, label_x, label_y) -> None:
        fig = plt.figure(figsize=(10, 10))
        plt.plot(x, y_train, color='blue', label='train')
        plt.plot(x, y_test, color='red', label='test')
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.title(title)
        plt.legend()
        st.pyplot(fig)

    def learn_and_tune(
            self,
            tune_learning_rate=True,
            tune_epoch=True,
            tune_hiddens=True
    ):
        """
        Autotune the neural network and draw plots
        """
        if tune_learning_rate:
            self.logger.msg_handler("Tuning learning rate started")
            lr_train_performances, lr_test_performances, learning_rates = self.tune_learning_rate()
            self.logger.msg_handler("Tuning learning rate finished")
            self._drawer(
                learning_rates,
                lr_train_performances,
                lr_test_performances,
                "Depending performance from learning rate",
                "Learning rate",
                "Performances"
            )
        if tune_epoch:
            self.logger.msg_handler("Tuning epochs started")
            epochs_train_performances, epochs_test_performances, epochs = self.tune_epochs()
            self.logger.msg_handler("Tuning epochs finished")
            self._drawer(
                epochs,
                epochs_train_performances,
                epochs_test_performances,
                "Depending performance from epochs",
                "Epochs",
                "Performances"
            )
        if tune_hiddens:
            self.logger.msg_handler("Tuning hidden nodes started")
            hid_train_performances, hid_test_performances, hidden_nodes = self.tune_hiddens()
            self.logger.msg_handler("Tuning hidden nodes finished")
            self._drawer(
                hidden_nodes,
                hid_train_performances,
                hid_test_performances,
                "Depending performance from hidden nodes",
                "Hidden nodes",
                "Performances"
            )


def user_teach_nn(learning_rate: float, epochs: int, hidden_nodes: int) -> None:
    """
    Teach the nn according user settings
    """
    Settings.LEARNING_RATE = learning_rate
    Settings.EPOCHS = epochs
    Settings.HIDDEN_NODES = hidden_nodes
    neural_network = NeuralNetwork(
        learning_rate=learning_rate,
        epochs=epochs,
        hidden_nodes=hidden_nodes
    )
    neural_network.nn_train()
    neural_network.nn_test()

    logger.msg_handler(Settings.get_current_nn_settings())
    tmp_train_performance, tmp_test_performance = neural_network.get_scorecard()
    logger.msg_handler(
        f"{tmp_train_performance=}"
        f"{tmp_test_performance=}"
    )


def st_get_settings() -> tuple[float, int, int]:
    """
    Get settings from user via streamlit
    """
    degree = 10
    learning_rate = float(st.number_input(
        'Learning rate',
        value=0.0,
        step=10 ** -degree,
        format=f"%.{degree}f",
        help=("Set here the learning rate in float format")
    ))
    hidden_nodes = int(st.number_input(
        'Hidden nodes',
        value=10,
        max_value=Settings.INPUT_NODES,
        min_value=Settings.OUTPUT_NODES,
        step=1,
        help=("Set here the number of nodes in hidden layer")
    ))
    epochs = int(st.number_input(
        'Epochs',
        value=1,
        min_value=1,
        step=1,
        help=("Set here the number of epochs")
    ))

    return learning_rate, epochs, hidden_nodes


if __name__ == '__main__':
    is_autotune = st.radio(
        "Do you want autotune neural network?",
        ('Yes', 'No')
    )

    if is_autotune == 'Yes':
        st.write('Ok, we will tune it')
        tuner = NNTuner()
        st.button(
            "Tune NN",
            on_click=tuner.learn_and_tune,
        )
    else:
        st.write('Ok, you will tune it')
        learning_rate, epochs, hidden_nodes = st_get_settings()
        st.button(
            "Teach the neural network",
            on_click=user_teach_nn,
            args=(learning_rate, epochs, hidden_nodes)
        )
