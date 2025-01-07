import numpy as np
import random
from RNN import RNN
from data import train_data, test_data


vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
vocab_size = len(vocab)
print(f'Найдено {vocab_size} уникальных слов.')

word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}


def createInputs(text: str) -> list[np.ndarray]:
    '''
    Создает список одноразрядных векторов для представления слов текста.
    Параметры:
        text (str): Входной текст.
    Возвращает:
        list[np.ndarray]: Список векторов формы (vocab_size, 1).
    '''
    inputs = []
    for w in text.split(' '):
        v = np.zeros((vocab_size, 1))
        v[word_to_idx[w]] = 1
        inputs.append(v)
    return inputs

def softmax(xs: np.ndarray) -> np.ndarray:
    '''
    Применяет функцию Softmax к входному массиву.
    Параметры:
        xs (np.ndarray): Входной массив.
    Возвращает:
        np.ndarray: Результат применения функции Softmax.
    '''
    return np.exp(xs) / np.sum(np.exp(xs), axis=0)


rnn = RNN(vocab_size, 2)


def processData(data: dict[str, bool], backprop: bool = True) -> tuple[float, float]:
    '''
    Рассчитывает потери и точность RNN для заданных данных.
    Параметры:
        data (dict[str, bool]): Словарь с текстами и их метками (True/False).
        backprop (bool): Указывает, выполнять ли обратное распространение ошибки.
    Возвращает:
        tuple[float, float]: Потери и точность.
    '''
    items = list(data.items())
    random.shuffle(items)
    loss = 0
    num_correct = 0

    for x, y in items:
        inputs = createInputs(x)
        target = int(y)

        out, _ = rnn.forward(inputs)
        probs = softmax(out)

        loss -= np.log(probs[target, 0])
        num_correct += int(np.argmax(probs) == target)

        if backprop:
            d_L_d_y = probs
            d_L_d_y[target] -= 1
            rnn.backprop(d_L_d_y)

    return loss / len(data), num_correct / len(data)


for epoch in range(1000):
    train_loss, train_acc = processData(train_data)

    if epoch % 100 == 99:
        print(f'--- Epoch {epoch + 1}')
        print(f'Train:\tLoss {float(train_loss):.3f} | Accuracy: {float(train_acc):.3f}')

        test_loss, test_acc = processData(test_data, backprop=False)
        print(f'Test:\tLoss {float(test_loss):.3f} | Accuracy: {float(test_acc):.3f}')
