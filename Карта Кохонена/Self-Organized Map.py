import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler


class MNISTData:
    '''Класс для загрузки и обработки данных MNIST.'''
    def __init__(self):
        '''Конструктор класса MNISTData.'''
        self.data: np.ndarray = None
        self.labels: np.ndarray = None

    def load_data(self) -> None:
        '''Загрузка данных MNIST и их нормализация.'''
        print("Загрузка данных MNIST...")
        mnist = fetch_openml('mnist_784', version=1)
        self.data = mnist.data.astype(np.float32)
        self.labels = mnist.target.astype(int)
        
        scaler = MinMaxScaler()
        self.data = scaler.fit_transform(self.data)
        print("Данные успешно загружены и нормализованы.")

    def get_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        '''
        Получение обучающей выборки.
        Возвращает:
            tuple[np.ndarray, np.ndarray]: Кортеж, содержащий данные и метки обучающей выборки.
        '''
        return self.data[:60000], self.labels[:60000]

    def get_test_data(self) -> tuple[np.ndarray, np.ndarray]:
        '''
        Получение тестовой выборки.
        Возвращает:
            tuple[np.ndarray, np.ndarray]: Кортеж, содержащий данные и метки тестовой выборки.
        '''
        return self.data[60000:], self.labels[60000:]

    def get_number_of_each_digit(self) -> pd.DataFrame:
        '''
        Получение количества изображений каждой цифры в обучающей и тестовой выборках.
        Возвращает:
            pd.DataFrame: Таблица с количеством изображений каждой цифры.
        '''
        train_counts = pd.Series(self.labels[:60000]).value_counts().sort_index()
        test_counts = pd.Series(self.labels[60000:]).value_counts().sort_index()
        table = pd.DataFrame({
            'Цифра': range(10),
            'Обучающая выборка': train_counts.values,
            'Тестовая выборка': test_counts.values
        })
        print(table)
        return table


class SOM:
    '''Класс для реализации нейронной сети Кохонена (Self-Organizing Map).'''
    def __init__(self, grid_shape: tuple[int, int], input_dim: int, learning_rate: float = 0.1, radius: float = None):
        '''
        Конструктор класса SOM.
        Параметры:
            grid_shape (tuple[int, int]): Размер сетки нейронов (строки, столбцы).
            input_dim (int): Размерность входных данных.
            learning_rate (float): Скорость обучения.
            radius (float): Радиус соседства нейронов. Если не указан, используется половина максимального размера сетки.
        '''
        self.grid_shape = grid_shape
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.radius = radius if radius is not None else max(grid_shape) / 2
        self.weights = np.random.random((grid_shape[0], grid_shape[1], input_dim))
        self.time_constant = 1000 / np.log(self.radius)

    def train(self, data: np.ndarray, num_iterations: int) -> None:
        '''
        Обучение сети Кохонена.
        Параметры:
            data (np.ndarray): Обучающие данные.
            num_iterations (int): Количество итераций обучения.
        '''
        for k in range(num_iterations):
            sample = data[np.random.randint(0, data.shape[0])]
            bmu_idx = self._find_bmu(sample)
            self._update_weights(sample, bmu_idx, k, num_iterations)
            if (k + 1) % 100 == 0 or k == 0:
                print(f"Итерация {k + 1}/{num_iterations} завершена.")

    def _find_bmu(self, sample: np.ndarray) -> tuple[int, int]:
        '''
        Поиск Best Matching Unit (BMU) — нейрона с наименьшим расстоянием до входного вектора.
        Параметры:
            sample (np.ndarray): Входной вектор.
        Возвращает:
            tuple[int, int]: Индексы BMU в сетке нейронов.
        '''
        distances = np.linalg.norm(self.weights - sample, axis=2)
        return np.unravel_index(np.argmin(distances), self.grid_shape)

    def _update_weights(self, sample: np.ndarray, bmu_idx: tuple[int, int], k: int, num_iterations: int) -> None:
        '''
        Обновление весов нейронов.
        Параметры:
            sample (np.ndarray): Входной вектор.
            bmu_idx (tuple[int, int]): Индексы BMU.
            k (int): Текущая итерация обучения.
            num_iterations (int): Общее количество итераций обучения.
        '''
        lr = self.learning_rate * np.exp(-k / num_iterations)
        radius_decay = self.radius * np.exp(-k / self.time_constant)

        for x in range(self.grid_shape[0]):
            for y in range(self.grid_shape[1]):
                distance = np.linalg.norm(np.array([x, y]) - np.array(bmu_idx))
                if distance < radius_decay:
                    influence = np.exp(-distance**2 / (2 * (radius_decay**2)))
                    self.weights[x, y] += lr * influence * (sample - self.weights[x, y])

    def visualize_weights(self) -> None:
        '''Визуализация весов нейронов в виде изображений.'''
        fig, axes = plt.subplots(self.grid_shape[0], self.grid_shape[1], figsize=(10, 10))
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                axes[i, j].imshow(self.weights[i, j].reshape(28, 28), cmap='gray')
                axes[i, j].axis('off')
        plt.show()

    def test(self, test_data: np.ndarray, test_labels: np.ndarray) -> None:
        '''
        Тестирование сети Кохонена на тестовой выборке.
        Параметры:
            test_data (np.ndarray): Тестовые данные.
            test_labels (np.ndarray): Метки тестовых данных.
        '''
        print("Тестирование на тестовой выборке...")
        bmu_positions = []
        for sample in test_data:
            bmu_idx = self._find_bmu(sample)
            bmu_positions.append(bmu_idx)

        bmu_positions = np.array(bmu_positions)
        plt.figure(figsize=(10, 10))
        plt.scatter(bmu_positions[:, 1], bmu_positions[:, 0], c=test_labels, cmap='tab10', s=5)
        plt.colorbar(label='Цифра')
        plt.gca().invert_yaxis()
        plt.title('Распределение тестовых данных на SOM')
        plt.xlabel('X координата нейрона')
        plt.ylabel('Y координата нейрона')
        plt.show()

    def count_clusters(self, data: np.ndarray, labels: np.ndarray) -> dict[int, int]:
        '''
        Подсчет количества кластеров для каждой цифры.
        Параметры:
            data (np.ndarray): Данные для кластеризации.
            labels (np.ndarray): Метки данных.
        Возвращает:
            dict[int, int]: Словарь, где ключи — цифры, а значения — количество кластеров для каждой цифры.
        '''
        print("Подсчет количества кластеров для каждой цифры...")
        cluster_counts = {}
        for digit in range(10):
            digit_data = data[labels == digit]
            bmu_positions = set()
            for sample in digit_data:
                bmu_idx = self._find_bmu(sample)
                bmu_positions.add(bmu_idx)
            cluster_counts[digit] = len(bmu_positions)
        print("Количество кластеров:", cluster_counts)
        return cluster_counts
    

if __name__ == "__main__":
    mnist_data = MNISTData()
    mnist_data.load_data()
    
    print("Таблица с количеством изображений каждой цифры:")
    mnist_data.get_number_of_each_digit()

    train_data, train_labels = mnist_data.get_training_data()
    
    som = SOM(grid_shape=(10, 10), input_dim=784, learning_rate=0.5)
    print("Обучение SOM...")
    som.train(train_data, num_iterations=1000)

    print("Визуализация весов карты Кохонена...")
    som.visualize_weights()

    test_data, test_labels = mnist_data.get_test_data()
    print("Распределение тестовых данных на SOM:")
    som.test(test_data, test_labels)

    cluster_counts = som.count_clusters(train_data, train_labels)
