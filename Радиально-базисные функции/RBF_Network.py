import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


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
        targets = self.data.iloc[:, -
                                 1].apply(lambda x: 1 if x == "Good" else 0).values
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


class RBFNetwork:
    '''
    Класс радиальной базисной сети (RBF-сети).
    '''

    def __init__(self, n_hidden_neurons: int, sigma: float = None):
        '''
        Конструктор класса RBFNetwork.
        Параметры:
            n_hidden_neurons (int): Количество нейронов скрытого слоя.
            sigma (float): Ширина окна радиальной функции. Если None, рассчитывается автоматически.
        '''
        self.n_hidden_neurons = n_hidden_neurons
        self.sigma = sigma  # Радиус функции.
        self.centers = None
        self.weights = None  # Веса выходного слоя.

    def _rbf_function(self, X: np.ndarray, center: np.ndarray) -> np.ndarray:
        '''
        Радиальная базисная функция (Гауссова функция).
        Параметры:
            X (np.ndarray): Входные данные.
            center (np.ndarray): Центр RBF.
        Возвращает:
            np.ndarray: Значения RBF для каждого входного вектора.
        '''
        return np.exp(-np.linalg.norm(X - center, axis=1)**2 / (2 * self.sigma**2))

    def fit(self, X: np.ndarray, y: np.ndarray):
        '''
        Обучение RBF-сети.
        Параметры:
            X (np.ndarray): Обучающие входные данные.
            y (np.ndarray): Обучающие метки классов.
        '''
        # Этап 1: Выбор центров RBF с помощью алгоритма K-means.
        kmeans = KMeans(n_clusters=self.n_hidden_neurons, random_state=52).fit(X)
        self.centers = kmeans.cluster_centers_

        # Этап 2: Определение ширины окна "сигма".
        if self.sigma is None:
            distances = [np.linalg.norm(c1 - c2) for i, c1 in enumerate(self.centers)
                         for c2 in self.centers[i + 1:]]
            self.sigma = np.mean(distances) / \
                np.sqrt(2 * self.n_hidden_neurons)

        # Этап 3: Формирование матрицы RBF (матрицы G).
        G = np.zeros((X.shape[0], self.n_hidden_neurons))
        for i, center in enumerate(self.centers):
            G[:, i] = self._rbf_function(X, center)

        # Этап 4: Вычисление весов выходного слоя (метод наименьших квадратов).
        self.weights = np.linalg.pinv(G) @ y

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Предсказание классов для входных данных.
        Параметры:
            X (np.ndarray): Входные данные.
        Возвращает:
            np.ndarray: Предсказанные метки классов.
        '''
        # Формирование матрицы RBF для входных данных.
        G = np.zeros((X.shape[0], self.n_hidden_neurons))
        for i, center in enumerate(self.centers):
            G[:, i] = self._rbf_function(X, center)

        # Рассчитываем выходные значения сети.
        predictions = G @ self.weights
        return (predictions >= 0.5).astype(int)

    def train(self, train_data: list[tuple[np.ndarray, int]], epochs: int = 1):
        '''
        Обучает RBF-сеть.
        Параметры:
            train_data (list[tuple[np.ndarray, int]]): Обучающая выборка.
            epochs (int): Количество эпох (для RBF обычно 1 достаточно).
        '''
        features, targets = zip(*train_data)
        X = np.array(features)
        y = np.array(targets)
        self.fit(X, y)

    def test(self, test_data: list[tuple[np.ndarray, int]]) -> None:
        '''
        Тестирует RBF-сеть и выводит результаты.
        Параметры:
            test_data (list[tuple[np.ndarray, int]]): Тестовая выборка.
        '''
        features, targets = zip(*test_data)
        X = np.array(features)
        y = np.array(targets)

        predictions = self.predict(X)

        correct_predictions = np.sum(predictions == y)
        total = len(y)
        accuracy = accuracy_score(y, predictions)

        print(f"Количество верных предсказаний: {correct_predictions}/{total}")
        print(f"Точность: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    dataset_handler = DatasetHandler()
    dataset_handler.load_data(r"C:\projects\MIREA\Проектирование и обучение нейронных сетей Ч.1\banana_quality.csv")
    dataset_handler.split_data(test_size=0.2)
    train_data = dataset_handler.train_data
    test_data = dataset_handler.test_data
    train_features, train_labels = zip(*train_data)
    test_features, test_labels = zip(*test_data)
    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    rbf_net = RBFNetwork(n_hidden_neurons=5)
    rbf_net.train(list(zip(train_features, train_labels)))

    rbf_net.test(list(zip(test_features, test_labels)))
