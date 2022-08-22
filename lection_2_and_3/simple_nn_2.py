import numpy
# сигмоид expit()
import scipy.special


# описание класса нейронной сети
class neuralNetwork:

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

        # уровень обучения
        self.lr = learningrate

        # функция активации - сигмоид
        self.activation_function = lambda x: scipy.special.expit(x)

    # обучение нейронной сети
    def train(self, inputs_list, targets_list):
        # преобразование входного списка 2d массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # вычисление сигналов на входе в скрытый слой
        hidden_inputs = numpy.dot(self.wih, inputs)
        # вычисление сигналов на выходе из скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # вычисление сигналов на входе в выходной слой
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # вычисление сигналов на выходе из выходного слоя
        final_outputs = self.activation_function(final_inputs)

        # ошибка на выходе (целевое значение - рассчитанное)
        output_errors = targets - final_outputs
        # распространение ошибки по узлам скрытого слоя
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # пересчет весов между скрытым и выходным слоем
        self.who += self.lr * numpy.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)),
            numpy.transpose(hidden_outputs)
        )

        # пересчет весов между входным и скрытым слоем
        self.wih += self.lr * numpy.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
            numpy.transpose(inputs)
        )

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
n = neuralNetwork(
    input_nodes,
    hidden_nodes,
    output_nodes,
    learning_rate
)

# Загрузка тренировочного набора данных
training_data_file = open("mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# Обучение нейронной сети
# количество эпох
epochs = 1

for e in range(epochs):
    # итерирование по всем записям обучающего набора
    for record in training_data_list:
        # разделение записей по запятым ','
        all_values = record.split(',')
        # масштабирование и сдвиг исходных данных
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # создание целевых  выходов
        targets = numpy.zeros(output_nodes) + 0.01
        # элемент all_values[0] является целевым для этой записи
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

# Загрузка тестового набора данных
test_data_file = open("mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# Тестирование нейронной сети

# Создание пустого накопителя для оценки качества
scorecard = []

# итерирование по тестовому набору данных
for record in test_data_list:
    # разделение записей по запятым ','
    all_values = record.split(',')
    # правильный ответ - в первой ячейке
    correct_label = int(all_values[0])
    # масштабирование и сдвиг исходных данных
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # получение ответа от нейронной сети
    outputs = n.query(inputs)
    # получение выхода
    label = numpy.argmax(outputs)
    # добавление в список единицы, если ответ совпал с целевым значением
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)

# расчет точности классификатора
scorecard_array = numpy.asarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)
