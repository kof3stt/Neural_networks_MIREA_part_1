import numpy as np
from numpy.random import randn


class RNN:
    '''Простая рекуррентная нейронная сеть (RNN) с архитектурой "многие-в-один".'''

    def __init__(self, input_size: int, output_size: int, hidden_size: int = 64):
        '''
        Инициализация параметров RNN.

        Параметры:
            input_size (int): Размер входных данных (размер словаря).
            output_size (int): Размер выходных данных (например, количество классов).
            hidden_size (int): Размер скрытого слоя.
        '''
        # Инициализация весов
        self.Whh = randn(hidden_size, hidden_size) / 1000  # Веса скрытого слоя
        self.Wxh = randn(hidden_size, input_size) / 1000   # Веса между входом и скрытым слоем
        self.Why = randn(output_size, hidden_size) / 1000  # Веса между скрытым слоем и выходом

        # Инициализация смещений
        self.bh = np.zeros((hidden_size, 1))  # Смещение скрытого слоя
        self.by = np.zeros((output_size, 1))  # Смещение выходного слоя

    def forward(self, inputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        '''
        Прямой проход через нейронную сеть.

        Параметры:
            inputs (list[np.ndarray]): Список входных данных, каждый элемент - one-hot вектор.

        Возвращает:
            tuple[np.ndarray, np.ndarray]: Кортеж из итогового выхода и скрытых состояний.
        '''
        h = np.zeros((self.Whh.shape[0], 1))  # Начальное скрытое состояние

        self.last_inputs = inputs  # Сохраняем входные данные
        self.last_hs = {0: h}  # Словарь для хранения скрытых состояний

        # Выполнение каждого шага RNN
        for i, x in enumerate(inputs):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)  # Обновление скрытого состояния
            self.last_hs[i + 1] = h  # Сохраняем скрытое состояние

        # Вычисление выхода
        y = self.Why @ h + self.by
        return y, h

    def backprop(self, d_y: np.ndarray, learn_rate: float = 2e-2) -> None:
        '''
        Обратное распространение ошибки для обновления весов.

        Параметры:
            d_y (np.ndarray): Градиент ошибки по выходу (dL/dy).
            learn_rate (float): Коэффициент обучения.
        '''
        n = len(self.last_inputs)

        # Вычисление градиентов по весам
        d_Why = d_y @ self.last_hs[n].T
        d_by = d_y

        # Инициализация градиентов для скрытых слоев
        d_Whh = np.zeros(self.Whh.shape)
        d_Wxh = np.zeros(self.Wxh.shape)
        d_bh = np.zeros(self.bh.shape)

        # Вычисление градиента по скрытому состоянию для последнего шага
        d_h = self.Why.T @ d_y

        # Обратное распространение ошибки через временные шаги
        for t in reversed(range(n)):
            # Промежуточное значение для вычисления градиентов
            temp = ((1 - self.last_hs[t + 1] ** 2) * d_h)

            # Градиенты по параметрам
            d_bh += temp
            d_Whh += temp @ self.last_hs[t].T
            d_Wxh += temp @ self.last_inputs[t].T

            # Обновление градиента по скрытому состоянию для предыдущего шага
            d_h = self.Whh @ temp

        # Обновление весов с помощью градиентного спуска
        for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
            np.clip(d, -1, 1, out=d)  # Обрезка градиентов для предотвращения взрывных градиентов

        # Применяем обновления весов
        self.Whh -= learn_rate * d_Whh
        self.Wxh -= learn_rate * d_Wxh
        self.Why -= learn_rate * d_Why
        self.bh -= learn_rate * d_bh
        self.by -= learn_rate * d_by
