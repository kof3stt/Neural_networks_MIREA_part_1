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


class Perceptron:
    def __init__(self, input_size: int, error : float, learning_rate: int = 0.01):
        self.weights = np.random.rand(input_size) * 0.1
        self.b = np.random.rand() * 0.1
        self.error = error
        self.learning_rate = learning_rate

    def activate(self, inputs):
        activation = np.dot(self.weights, inputs.astype(np.int8))
        return 1 if activation >= self.b else -1

    def train_perceptron(self, inputs, targets):
        current_error = float('inf')
        epoch = 0
        while current_error > self.error:
            errors = []
            print(f"Epoch {epoch+1}:")
            for i in range(len(inputs)):
                output = self.activate(inputs[i])
                error = targets[i] - output
                errors.append(abs(error))
                self.weights += self.learning_rate * \
                    error * inputs[i].astype(np.int8)
                self.b += self.learning_rate * error
                print(f"  Входы: {inputs[i]}, Цифра: {get_number(
                    inputs[i])}, Целевое: {targets[i]}, Выход: {output}")
            current_error = np.mean(errors)
            print("  Веса:", self.weights)
            print("  Порог:", self.b)
            if current_error < self.error:
                print(f"Обучение завершено на эпохе {epoch + 1}")
                break
            epoch += 1

    def check_perceptron(self, inputs, targets):
        accuracy = 0
        for i in range(len(inputs)):
            output = self.activate(inputs[i])
            error = targets[i] - output
            self.weights += self.learning_rate * \
                error * inputs[i].astype(np.int8)
            self.b += self.learning_rate * error
            accuracy += output == targets[i]
            print(f"  Входы: {inputs[i]}, Целевое: {
                targets[i]}, Выход: {output}")
        return accuracy / len(targets) * 100


numbers = np.array([list('111101101101111'), list('001001001001001'),
                    list('111001111100111'), list('111001111001111'),
                    list('101101111001001'), list('111100111001111'),
                    list('111100111101111'), list('111001001001001'),
                    list('111101111101111'), list('111101111001111')])

targets_numbers = np.array([-1, -1, -1, -1, -1, 1, -1, -1, -1, -1])

for num in numbers:
    print_number(num)
    print('----------------')

perceptron = Perceptron(15, 0.000000001)
perceptron.train_perceptron(numbers, targets_numbers)

print('Тестовая выборка:')
fake_numbers = np.array([list('111100111000111'), list('111100010001111'),
                         list('111100011001111'), list('110100111001111'),
                         list('110100111001011'), list('111100101001111'),
                         list('111100111001111'), list('111100111101111')])

targets_fake_numbers = np.array([1, 1, 1, 1, 1, 1, 1, -1])

for fake_num in fake_numbers:
    print_number(fake_num)
    print('----------------')

print(f'Accuracy: {perceptron.check_perceptron(
    fake_numbers, targets_fake_numbers)}%')
