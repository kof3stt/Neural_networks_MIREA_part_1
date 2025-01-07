import numpy as np


class HebbianNeuron:
    '''Обучение по модели Хебба для отдельного нейрона'''

    def __init__(self, input_size: int, epochs: int = 10, learning_rate: float = 0.1):
        """
        Инициализирует нейрон Хебба с заданным количеством входов
        Args:
            input_size: Количество входов нейрона
            epochs: Количество эпох обучения
            learning_rate: Коэффициент скорости обучения
        """
        self.weights = np.random.rand(input_size) * 0.1
        self.epochs = epochs  # Инициализация количества эпох обучения
        self.b = np.random.rand() * 0.1  # Инициализация порога значением, близким к нулю
        self.learning_rate = learning_rate  # Инициализация коэффициента скорости обучения

    def activate(self, inputs) -> int:
        '''Вычисляет активацию нейрона'''
        activation = np.dot(self.weights, inputs) + self.b
        return 1 if activation >= 0 else 0

    def train(self, inputs, target):
        '''Обучает нейрон на одном примере, производит модификацию весовых коэффициентов'''
        output = self.activate(inputs)
        if output != target:
            if output == 0:  # Если выход неверен и равен 0
                self.weights[0] += self.learning_rate * inputs[0]
                self.weights[1] += self.learning_rate * inputs[1]
                self.b += self.learning_rate
            else:  # Если выход неверен и равен 1
                self.weights[0] -= self.learning_rate * inputs[0]
                self.weights[1] -= self.learning_rate * inputs[1]
                self.b -= self.learning_rate
        return output

    def train_hebbian_neuron(self, inputs, targets):
        '''Обучает нейрон Хебба на заданном наборе данных'''
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}:")
            for i in range(len(inputs)):
                output = self.train(inputs[i], targets[i])
                print(f"  Входы: {inputs[i]}, Целевое: {
                    targets[i]}, Выход: {output}")
            print("  Веса:", self.weights)
            print("  Порог:", self.b)


inputs_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets_and = np.array([0, 0, 0, 1])  # AND функция

inputs_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets_or = np.array([0, 1, 1, 1])  # OR функция

# neuron_and = HebbianNeuron(2)
# neuron_and.train_hebbian_neuron(inputs_and, targets_and)

neuron_or = HebbianNeuron(2)
neuron_or.train_hebbian_neuron(inputs_or, targets_or)
