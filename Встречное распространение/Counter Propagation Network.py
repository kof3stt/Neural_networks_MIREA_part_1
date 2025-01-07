import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class MNISTData:
    '''Класс для загрузки и обработки данных MNIST.'''
    def __init__(self):
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
        '''Получение обучающей выборки.'''
        return self.data[:60000], self.labels[:60000]

    def get_test_data(self) -> tuple[np.ndarray, np.ndarray]:
        '''Получение тестовой выборки.'''
        return self.data[60000:], self.labels[60000:]


class CPN:
    '''Сеть встречного распространения с слоями Кохонена и Гроссберга.'''
    def __init__(self, kohonen_shape=(10, 10), input_dim=784, output_dim=10, learning_rate=0.1):
        '''
        Конструктор сети.
        Параметры:
            kohonen_shape (tuple): Размер сетки слоя Кохонена.
            input_dim (int): Размерность входного вектора.
            output_dim (int): Количество классов (меток).
            learning_rate (float): Скорость обучения.
        '''
        self.kohonen_shape = kohonen_shape
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        # Инициализация весов
        self.kohonen_weights = np.random.random((kohonen_shape[0], kohonen_shape[1], input_dim))
        self.grossberg_weights = np.zeros((kohonen_shape[0], kohonen_shape[1], output_dim))

    def train(self, X_train, y_train, num_epochs=10):
        '''Обучение сети на данных.'''
        print("Начало обучения сети встречного распространения...")
        for epoch in range(num_epochs):
            for i, sample in enumerate(X_train):
                # Слой Кохонена: находим BMU
                bmu_idx = self._find_bmu(sample)

                # Обновляем веса слоя Кохонена
                self._update_kohonen_weights(sample, bmu_idx)

                # Обновляем веса слоя Гроссберга
                self._update_grossberg_weights(y_train[i], bmu_idx)

            print(f"Эпоха {epoch + 1}/{num_epochs} завершена.")
        print("Обучение завершено!")

    def predict(self, X):
        '''Предсказание меток для входных данных.'''
        predictions = []
        for sample in X:
            # Найти BMU на слое Кохонена
            bmu_idx = self._find_bmu(sample)

            # Получить предсказание на слое Гроссберга
            grossberg_output = self.grossberg_weights[bmu_idx]
            predicted_label = np.argmax(grossberg_output)
            predictions.append(predicted_label)
        return np.array(predictions)

    def _find_bmu(self, sample):
        '''Находим нейрон с минимальным расстоянием на слое Кохонена (BMU).'''
        distances = np.linalg.norm(self.kohonen_weights - sample, axis=2)
        return np.unravel_index(np.argmin(distances), self.kohonen_shape)

    def _update_kohonen_weights(self, sample, bmu_idx):
        '''Обновление весов слоя Кохонена с использованием функции соседства.'''
        x, y = bmu_idx
        radius = 2
        for i in range(self.kohonen_shape[0]):
            for j in range(self.kohonen_shape[1]):
                distance = np.linalg.norm(np.array([i, j]) - np.array([x, y]))
                if distance < radius:
                    influence = np.exp(-distance**2 / (2 * radius**2))
                    self.kohonen_weights[i, j] += self.learning_rate * influence * (sample - self.kohonen_weights[i, j])

    def _update_grossberg_weights(self, target_label, bmu_idx):
        '''Обновление весов слоя Гроссберга.'''
        x, y = bmu_idx
        target_output = np.zeros(self.output_dim)
        target_output[target_label] = 1
        self.grossberg_weights[x, y] += self.learning_rate * (target_output - self.grossberg_weights[x, y])


if __name__ == "__main__":
    # Загрузка данных
    mnist_data = MNISTData()
    mnist_data.load_data()

    # Получение обучающих и тестовых данных
    X_train, y_train = mnist_data.get_training_data()
    X_test, y_test = mnist_data.get_test_data()

    # Создание и обучение сети встречного распространения
    cpn = CPN(kohonen_shape=(10, 10), input_dim=784, output_dim=10, learning_rate=0.1)
    cpn.train(X_train, y_train, num_epochs=10)

    # Тестирование сети
    print("Тестирование сети...")
    y_pred = cpn.predict(X_test)
    print("Точность на тестовой выборке:", accuracy_score(y_test, y_pred))
    print("Матрица ошибок:")
    print(confusion_matrix(y_test, y_pred))
    print("Отчет по классификации:")
    print(classification_report(y_test, y_pred))
