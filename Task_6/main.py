from sklearn import datasets
import plotly.express as px
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
import math

from collections import Counter

import numpy as np


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Вычислим расстояния между x и всеми примерами в обучающем наборе
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Сортируем по расстоянию и возвращаем индексы первых k соседей
        k_idx = np.argsort(distances)[: self.k]
        # Извлекаем метки k ближайших соседей обучающих выборок
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        # вернуть наиболее распространенную метку класса
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]


if __name__ == '__main__':
    # todo: импортируем датасет
    df = px.data.iris()
    print(df)

    # todo: Матрица рассеяния набора:
    # В ячейке (i,j) такой матрицы отображается точечная диаграмма переменной Xi в зависимости от Xj.
    # diagonal_visible=False
    # До нормализации
    fig = px.scatter_matrix(df, dimensions=["sepal_width", "sepal_length", "petal_width", "petal_length"],
                            color="species")
    fig.show()

    # todo: нормализовываем данные
    normalizer = Normalizer()
    df_scaled = normalizer.fit_transform(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])

    # todo: После нормализации
    fig = px.scatter_matrix(df_scaled, dimensions=[0, 1, 2, 3], color=df['species_id'])
    fig.show()

    # todo: разделяем выборку на две части по species_id
    X_train, X_test, y_train, y_test = train_test_split(df_scaled,
                                                        df['species_id'].values,
                                                        test_size=0.25,
                                                        random_state=1)

    # кол-во элем в каждом измерении в тестовой выборке
    n = X_test.shape[0]
    print("n_shape: ", n)

    # todo: подбираем оптимальное K
    metricas = []
    # for i in range(1, math.ceil(math.sqrt(n))):
    # 0.9736842105263158
    # 0.9736842105263158
    for i in range(1, 37):
        # создаем объект
        knn = KNN(k=i)
        # отправляем тренировочную выборку и i (оптимальное кол-во соседей)
        knn.fit(X_train, y_train)
        # делаем предсказание
        pred = knn.predict(X_test)

        print("pred main: ", pred)
        # вычисляем точность при заданных параметрах
        metricas.append([accuracy(y_test, pred), i])

    print("metricas: ", metricas)

    k = 0
    max_accuracy = 0
    for index_elem, elem in enumerate(metricas):
        print("elem[0]", elem[0])
        print("elem[1] ", elem[1])
        if elem[0] > max_accuracy:
            print("true")
            k = elem[1]
            max_accuracy = elem[0]
    print("result k ", k)

    # todo -----------------------------------------------------

    # todo: c оптимальным значением
    clf = KNN(k=k)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("X_test ", X_test)
    print("KNN classification accuracy", accuracy(y_test, predictions))

    fig = px.scatter_matrix(X_test, dimensions=[0, 1, 2, 3], color=predictions)
    fig.show()

    # todo: с выбранным значением
    # [0.71491405 0.30207636 0.59408351 0.21145345]
    predictions = clf.predict([[0.71491405, 0.30207636, 0.59408351, 0.21145345]])
    print("X_test 2 ", [[0.71491405, 0.30207636, 0.59408351, 0.21145345]])
    # print("KNN classification accuracy", accuracy(y_test, predictions))

    print("Result: ")
    if predictions == 3:
        print("virginica")
    if predictions == 1:
        print("setosa")
    if predictions == 2:
        print("versi")

    # fig = px.scatter_matrix([[0.71491405, 0.30207636, 0.59408351, 0.21145345]], dimensions=[0, 1, 2, 3],
    #                        color=predictions)
    # fig.show()
