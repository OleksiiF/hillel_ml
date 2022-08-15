import numpy
import scipy.special


# описание класса нейронной сети
class NeuralNetwork:

    # инициализация нейронной сети
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # задание количества узлов входного, скрытого и выходного слоя
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # связь весовых матриц, wih и who
        # вес внутри массива w_i_j, где связь идет из узла i в узел j
        # следующего слоя
        # w11 w21
        # w12 w22 и т д7
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # уровень обучения
        self.lr = learningrate

        # функция активации - сигмоид
        self.activation_function = lambda x: scipy.special.expit(x)

    # запрос к нейронной сети
    def query(self, inputs_list):
        # преобразование входного списка 2d массив
        inputs = numpy.array(inputs_list, ndmin=2).T

        # вычисление сигналов на входе в скрытый слой
        hidden_inputs = numpy.dot(self.wih, inputs)
        # вычисление сигналов на выходе из скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # вычисление сигналов на входе в выходной слой
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # вычисление сигналов на выходе из выходного слоя
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


# Задание архитектуры сети:
# количество входных, скрытых и выходных узлов
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# уровень обучения
learning_rate = 0.1

# создание экземпляра класса нейронной сети
n = NeuralNetwork(
    input_nodes,
    hidden_nodes,
    output_nodes,
    learning_rate
)
