import numpy as np


def print_number(number):
    for i in range(0, len(number), 3):
        for j in number[i: i + 3]:
            if j == '0':
                print(' ', end='')
            else:
                print('*', end='')
        print()


def get_number(number):
    dict_numbers = {k: v for k, v in zip(
        [''.join(num) for num in numbers], range(10))}
    number = ''.join(number)
    return dict_numbers[number]


class HebbianNeuron:
    '''Обучение по модели Хебба для отдельного нейрона'''

    def __init__(self, input_size: int, epochs: int = 100, learning_rate: float = 0.01):
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
        inputs = inputs.astype(np.int8)
        output = self.activate(inputs)
        if output != target:
            if output == 0:  # Если выход неверен и равен 0
                for i in range(len(self.weights)):
                    self.weights[i] += self.learning_rate * inputs[i]
                self.b += self.learning_rate
            else:  # Если выход неверен и равен 1
                for i in range(len(self.weights)):
                    self.weights[i] -= self.learning_rate * inputs[i]
                self.b -= self.learning_rate
        return output

    def train_hebbian_neuron(self, inputs, targets):
        '''Обучает нейрон Хебба на заданном наборе данных'''
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}:")
            for i in range(len(inputs)):
                output = self.train(inputs[i], targets[i])
                print(f"  Входы: {inputs[i]}, Цифра: {get_number(inputs[i])}, Целевое: {
                    targets[i]}, Выход: {output}")
            print("  Веса:", self.weights)
            print("  Порог:", self.b)

    def check_hebbian_neuron(self, inputs, targets):
        '''Проверка обученного нейрона на тестовой выборке'''
        accuracy = 0
        for i in range(len(inputs)):
            output = self.train(inputs[i], targets[i])
            accuracy += output == targets[i]
            print(f"  Входы: {inputs[i]}, Целевое: {
                targets[i]}, Выход: {output}")
        return accuracy / len(targets) * 100


numbers = np.array([list('111101101101111'), list('001001001001001'),
                    list('111001111100111'), list('111001111001111'),
                    list('101101111001001'), list('111100111001111'),
                    list('111100111101111'), list('111001001001001'),
                    list('111101111101111'), list('111101111001111')])

targets_numbers = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])

for num in numbers:
    print_number(num)
    print('----------------')

neuron = HebbianNeuron(15)
neuron.train_hebbian_neuron(numbers, targets_numbers)

print('Тестовая выборка:')
fake_numbers = np.array([list('111100111000111'), list('111100010001111'),
                         list('111100011001111'), list('110100111001111'),
                         list('110100111001011'), list('111100101001111'),
                         list('111100111001111'), list('111100111101111')])

targets_fake_numbers = np.array([1, 1, 1, 1, 1, 1, 1, 0])

for fake_num in fake_numbers:
    print_number(fake_num)
    print('----------------')

print(f'Accuracy: {neuron.check_hebbian_neuron(
    fake_numbers, targets_fake_numbers)}%')
