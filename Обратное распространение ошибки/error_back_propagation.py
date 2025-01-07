import pandas as pd
from sklearn.model_selection import train_test_split
import time
import random
import math


class DatasetHandler:
    '''Класс для загрузки, разбиения и отображения данных.'''
    def __init__(self):
        '''Конструктор класса DatasetHandler.'''
        self.data = None
        self.train_data = None
        self.test_data = None

    def load_data(self, file_name: str):
        '''
        Считывает датасет из файла.
        Параметры:
            file_name (str): Путь к файлу с данными.
        '''
        self.data = pd.read_csv(file_name)

    def split_data(self, test_size: float = 0.2, random_state: int = 42):
        '''
        Разделяет датасет на обучающую и тестовую выборки.
        Параметры:
            test_size (float): Доля тестовой выборки (от 0 до 1).
            random_state (int): Случайное состояние для воспроизводимости.
        '''
        if self.data is None:
            raise ValueError("Данные не загружены. Используйте load_data().")
        features = self.data.iloc[:, :-1].values
        targets = self.data.iloc[:, -1].apply(lambda x: 1 if x == "Good" else 0).values
        self.train_data, self.test_data = train_test_split(
            list(zip(features, targets)), test_size=test_size, random_state=random_state
        )

    def show_sample(self, n: int = 10):
        '''
        Отображает несколько строк датасета.
        Параметры:
            n (int): Количество строк для отображения.
        '''
        if self.data is None:
            raise ValueError("Данные не загружены. Используйте load_data().")
        print(self.data.head(n))


class NeuralNetwork:
    '''
    Класс нейронной сети с одним скрытым слоем.
    '''
    def __init__(self, input_size: int, hidden_size: int, output_size: int, learning_rate: float = 0.01):
        '''
        Конструктор класса NeuralNetwork.
        Параметры:
            input_size (int): Размер входного слоя.
            hidden_size (int): Размер скрытого слоя.
            output_size (int): Размер выходного слоя.
            learning_rate (float): Скорость обучения.
        '''
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.weights_input_hidden = [
            [random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
        self.weights_hidden_output = [
            [random.uniform(-1, 1) for _ in range(output_size)] for _ in range(hidden_size)]
        self.bias_hidden = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.bias_output = [random.uniform(-1, 1) for _ in range(output_size)]

    @staticmethod
    def sigmoid(x: float) -> float:
        '''
        Сигмоидальная (логистическая) функция активации.
        Параметры:
            x (float): Входное значение.
        Возвращает:
            float: Результат применения функции.
        '''
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def sigmoid_derivative(x: float) -> float:
        '''
        Производная сигмоидальной функции.
        Параметры:
            x (float): Значение, к которому применяется производная.
        Возвращает:
            float: Производная функции.
        '''
        return x * (1 - x)

    def feedforward(self, inputs: list[float]) -> list[float]:
        '''
        Выполняет прямое распространение данных через сеть.
        Параметры:
            inputs (list[float]): Входные значения.
        Возвращает:
            list[float]: Выходные значения сети.
        '''
        self.inputs = inputs
        self.hidden_inputs = [
            sum(inputs[i] * self.weights_input_hidden[i][j] for i in range(self.input_size)) + self.bias_hidden[j]
            for j in range(self.hidden_size)
        ]
        self.hidden_outputs = [self.sigmoid(x) for x in self.hidden_inputs]
        self.final_inputs = [
            sum(self.hidden_outputs[j] * self.weights_hidden_output[j][k] for j in range(self.hidden_size)) + self.bias_output[k]
            for k in range(self.output_size)
        ]
        self.final_outputs = [self.sigmoid(x) for x in self.final_inputs]
        return self.final_outputs

    def backpropagate(self, targets: list[float]):
        '''
        Выполняет обратное распространение ошибки.

        Параметры:
            targets (list[float]): Целевые значения.
        '''
        output_errors = [targets[k] - self.final_outputs[k] for k in range(self.output_size)]
        output_gradients = [
            output_errors[k] * self.sigmoid_derivative(self.final_outputs[k]) for k in range(self.output_size)
        ]

        hidden_errors = [
            sum(self.weights_hidden_output[j][k] * output_gradients[k] for k in range(self.output_size))
            for j in range(self.hidden_size)
        ]
        hidden_gradients = [
            hidden_errors[j] * self.sigmoid_derivative(self.hidden_outputs[j]) for j in range(self.hidden_size)
        ]

        for j in range(self.hidden_size):
            for k in range(self.output_size):
                self.weights_hidden_output[j][k] += self.learning_rate * output_gradients[k] * self.hidden_outputs[j]

        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.weights_input_hidden[i][j] += self.learning_rate * hidden_gradients[j] * self.inputs[i]

        for k in range(self.output_size):
            self.bias_output[k] += self.learning_rate * output_gradients[k]
        for j in range(self.hidden_size):
            self.bias_hidden[j] += self.learning_rate * hidden_gradients[j]

    def train(self, training_data: list[tuple[list[float], float]], epochs: int = 100):
        '''
        Обучает нейронную сеть на предоставленных данных.
        Параметры:
            training_data (list[tuple[list[float], float]]): Обучающая выборка.
            epochs (int): Количество эпох обучения.
        '''
        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in training_data:
                outputs = self.feedforward(inputs)
                self.backpropagate([targets])
                total_loss += sum((targets - outputs[k]) ** 2 for k in range(len(outputs))) * 0.5
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    def test(self, test_data: list[tuple[list[float], float]]) -> tuple[float, float]:
        '''
        Тестирует нейронную сеть.
        Параметры:
            test_data (list[tuple[list[float], float]]): Тестовая выборка.
        Возвращает:
            tuple[float, float]: Точность (в процентах) и ошибка.
        '''
        correct_predictions = 0
        total_loss = 0
        for inputs, targets in test_data:
            outputs = self.feedforward(inputs)
            prediction = round(outputs[0])
            correct_predictions += int(prediction == targets)
            total_loss += 0.5 * (targets - outputs[0]) ** 2
        accuracy = (correct_predictions / len(test_data)) * 100
        print(f"Количество верных предсказаний: {correct_predictions}/{len(test_data)}")
        print(f"Test Accuracy: {accuracy:.2f}%, Loss: {total_loss:.4f}")
        return accuracy, total_loss


if __name__ == "__main__":
    dataset_handler = DatasetHandler()
    dataset_handler.load_data(r"C:\projects\MIREA\Проектирование и обучение нейронных сетей Ч.1\banana_quality.csv")
    dataset_handler.split_data(test_size=0.2)
    dataset_handler.show_sample()

    train_data = dataset_handler.train_data
    test_data = dataset_handler.test_data

    start = time.perf_counter()
    nn = NeuralNetwork(input_size=7, hidden_size=5, output_size=1, learning_rate=0.01)
    nn.train(train_data, epochs=100)
    print(f"Время обучения: {time.perf_counter() - start:.2f} секунд")
    nn.test(test_data)


# Size - размер плода
# Weight - вес плода
# Sweetness - сладость плода
# Softness - мягкость плода
# HarvestTime - количество времени, прошедшее с момента сбора плода
# Ripeness - спелость плода
# Acidity - кислотность фруктов 
# Quality - качество фруктов
