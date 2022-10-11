import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# region euclidean_distance
# todo: функция евклидова расстояния между двумя точками где x1(x,y) x2(x,y)
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


# endregion

class KMeans:
    # K - кол-во кластеров max_iters - максимальное кол-во иттераций для оптимизации
    def __init__(self, K=2, max_iters=100, plot_steps=False, GlobalItter=1):
        # K - кол-во кластеров
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        self.GlobalItter = GlobalItter
        self.Distance = 0

        # Список списков индексов для каждого кластера для 3х: [[], [], []]
        self.clusters = [[] for _ in range(self.K)]
        # print("__init__ self.clusters", self.clusters)
        # средний вектор признаков для каждого кластера
        self.centroids = []

    def funcItter(self):
        Gi = self.GlobalItter
        print("funcItter: ", Gi)
        return Gi

    def predict(self, X):
        # todo: точки
        self.X = X
        # print("predict self.X", self.X)
        # число выборок и ко-во функций X.shape = кол-во точек
        self.n_samples, self.n_features = X.shape
        # print("self.n_samples", self.n_samples)

        # todo: Инициализация ценроидов
        # todo: рандомно инициализируем центроиды
        random_samples_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_samples_idxs]
        # print("self.centroids", self.centroids)

        # Оптимизация (в диапазоне max itter)
        for _ in range(self.max_iters):
            self.GlobalItter = self.GlobalItter + _
            # print("self.GlobalItter: ", self.GlobalItter)
            if self.plot_steps:
                self.plot()
            # todo: Обновляем кластеры (отправляем 3 точки центроида)
            self.clusters = self._create_clusters(self.centroids)
            # todo: итог: получили индексы всех точек распределенных по ближайшим к ним кластерам
            # todo: при этом центроиды рандомны на первом шаге
            #  todo: строим снова график после обновления кластеров
            if self.plot_steps:
                self.plot()

            # todo: сохраняем старые точки(центроидов) в centroids_old
            centroids_old = self.centroids
            # todo: обновляем центроиды
            # todo: среднее значение кластеров к центроиду (для каждого кластера вычисляем среднее значения)
            # todo: отдаем полученные кластеры с индексами точек
            self.centroids = self._get_centroids(self.clusters)
            # todo: получаем новые координаты центроидов для наших кластеров
            # todo: строим снова график после обновления центроидов
            if self.plot_steps:
                self.plot()

            # todo: проверка сходимости расчитывает расстояние между каждым старым и новым центроидами для всех центроидов
            # todo: и проверяет равно ли это 0 если да выходим из цикла for
            if self._is_converged(centroids_old, self.centroids):
                # print("self.GlobalItter: ", self.GlobalItter)
                break
        # todo: вернуть метки кластера
        # print("predict -- self.clusters: ", self.clusters)
        # self.clusters это список индексов точек в self.K для каждого из кластеров
        # print("predict -- self._get_cluster_labels(self.clusters): \n", self._get_cluster_labels(self.clusters))
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        # todo: принимаем список точек для каждого из кластеров
        # todo: self.n_samples = кол-во точек
        # print("_get_cluster_labels -- self.n_samples ", self.n_samples)
        labels = np.empty(self.n_samples)
        # print("_get_cluster_labels -- labels np.empty \n", labels)
        # todo: для каждой выборки возвращаем кластер которому он был назначен
        for clusters_idx, cluster in enumerate(clusters):
            # print("_get_cluster_labels -- clusters_idx: ", clusters_idx)
            # print("_get_cluster_labels -- cluster: \n", cluster)
            # todo: проходимся по всем точкам в кластере
            for sample_idx in cluster:
                # print("_get_cluster_labels -- sample_idx: \n", sample_idx)
                # todo: присваиваем каждому индексу (пустого) массива labels  под индексом точки из кластера
                # todo: и присваиваем значение индекса(номер) кластера
                labels[sample_idx] = clusters_idx
            # print("_get_cluster_labels -- for labels: \n", labels)
        # todo: получаем массив из 200 элементов где каждое значение это индекс кластера
        # print("_get_cluster_labels -- return labels: \n", labels)
        return labels

    # todo: Создание кластеров
    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        # print("_create_clusters clusters ", clusters)

        # для индекса точки и точки перечислить точку из X
        for idx, sample in enumerate(self.X):
            # todo: индекс точки:
            # print("idx: ", idx)
            # todo: точка:
            # print("sample: ", sample)

            # todo: получить индекс ближайшего центроида к выбранной точке
            centroid_idx = self._closest_centroid(sample, centroids)
            # print("centroid_idx ", centroid_idx)
            # todo: добавляем к кластеру с центроидом centroid_idx индекс этой точки,
            # todo: где индекс кластера равен номеру(индексу) центроида
            clusters[centroid_idx].append(idx)
            # print("_create_clusters -- clusters 2 ", clusters)
        return clusters

    # todo: получить ближайший центроид к выбранной точке sample - точка, centroids - точка центроида
    def _closest_centroid(self, sample, centroids):
        # todo: расстояние от данной точки до каждого из (трех) центроида(ов)
        # _closest_centroid -- distances [5.066825296401784, 4.56377602701544, 1.454288868000231]
        # todo: получаем массив из расстояний
        distances = [euclidean_distance(sample, point) for point in centroids]
        # print("_closest_centroid -- distances", distances)
        # todo: получаем индекс с минимальным расстоянием (ближайший индекс равен numpy.argmin этих расстояний)
        closest_idx = np.argmin(distances)
        # т.к. index то с 0 начинается 1 2 => 1.454288868000231 min
        # _closest_centroid -- closest_idx 2
        # print("_closest_centroid -- closest_idx: ", closest_idx)
        return closest_idx

    # среднее значение кластеров к центроиду (для каждого кластера вычисляем среднее значения)
    def _get_centroids(self, clusters):
        # print("_get_centroids -- clusters: ", clusters)
        centroids = np.zeros((self.K, self.n_features))
        # print("_get_centroids -- self.K: ", self.K)
        # print("_get_centroids -- self.n_features: ", self.n_features)
        # print("_get_centroids -- centroids: ", centroids)
        for cluster_idx, cluster in enumerate(clusters):
            # todo: для каждого индекса выбираем кластер из массива
            # print("_get_centroids -- cluster_idx: ", cluster_idx)
            # print("_get_centroids -- cluster: ", cluster)

            # todo: назначаем текущему кластеру среднее арифметическое значение
            # print("_get_centroids -- self.X[cluster]: ", self.X[cluster])
            # todo: достаем каждую точку по индексу из кластера
            # todo: и считаем среднее арифметическое значение всех X и Y
            cluster_mean = np.mean(self.X[cluster], axis=0)
            # print("_get_centroids -- cluster_mean: ", cluster_mean)
            centroids[cluster_idx] = cluster_mean
            # print("_get_centroids -- centroids 3: ", centroids)
        # print("_get_centroids -- centroids  итог для return: ", centroids)
        # todo: получаем координаты новых центроидов в каждом кластере
        return centroids

    # todo: проверка сходимости
    def _is_converged(self, centroids_old, centroids):
        # todo: принимает координаты старых центроидов и новых
        # print("_is_converged -- centroids_old: \n", centroids_old)
        # print("_is_converged -- centroids: \n", centroids)
        # todo: координаты старого центроида первого кластера = x1
        # todo: координаты нового центроида первого кластера = x2
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        # todo: получаем расстояние между старым и новым центроидом в каждом кластере
        # print("_is_converged -- distances: ", distances)
        # print("_is_converged -- sum(distances): ", sum(distances))
        # todo: суммируем и проверяем равно ли нулю
        return sum(distances) == 0

    def plot(self):
        fig, ax = plt.subplots(figsize=(8, 4))
        # todo: перебираем кластеры
        for i, index in enumerate(self.clusters):
            # print("plot -- i: ", i)
            # print("plot -- index: ", index)
            # todo: получаем текущие точки равные собственной точке х
            point = self.X[index].T
            # print("plot -- self.X[index]: \n", self.X[index])
            # print("plot -- point: \n", self.X[index])
            # todo: рисуем все точки для каждого кластера свой цвет
            ax.scatter(*point)
            # todo: наносим все центроиды
        for point in self.centroids:
            ax.scatter(*point, marker='x', color="black", linewidth=2)
        plt.show()

    def Dist(self):
        # todo: получаем список координаты центроидов в каждом кластере
        # print("Dist -- self.centroids: ", self.centroids)
        # todo: получаем индексы точек i кластера
        for i, index in enumerate(self.clusters):
            # print("Dist -- i: ", i)
            # print("Dist -- list index: ", index)
            # print("Dist -- centroids: ", self.centroids[i])
            # todo: 2) сумма расстояний от точек до центроида в i кластере
            tempDistance = 0
            for j in index:
                # todo: для каждого индекса в i кластере
                # print("Dist -- j: ", j)
                # todo: получаем нужную точку
                # print("Dist -- self.X[j]: ", self.X[j])
                # todo: 1) получаем расстояние от точки до центроида в i кластере и суммируем с остальными
                tempDistance = tempDistance + euclidean_distance(self.X[j], self.centroids[i])
            # print("Dist -- tempDistance: ", tempDistance)
            self.Distance = self.Distance + tempDistance
        # print("Dist -- return self.Distance: ", self.Distance)
        return self.Distance


# todo: k means
if __name__ == '__main__':
    # todo: генерируем точки
    np.random.seed(42)
    X, y = make_blobs(200, random_state=0)
    # print(X.shape)
    print("кол-во точек: ", len(X))

    # default
    clusters = 3
    print("кол-во кластеров: ", clusters)
    GlobalItter = 1
    k = KMeans(K=clusters, max_iters=150, plot_steps=False, GlobalItter=GlobalItter)
    y_pred = k.predict(X)
    k.Dist()
    k.plot()

    # todo: 1) Определить число возможных кластеров
    # todo: от одного до два корня из кол-ва точек
    n = 2 * np.sqrt(len(X))
    Sum_of_distance = []
    # todo: приблизительное кол-во кластеров
    print(round(n))
    for i in range(1, round(n)):
        # todo: прогоняем по алгоритму с i кластероми
        temp_k = KMeans(K=i, max_iters=150, plot_steps=False, GlobalItter=GlobalItter)
        temp_y_pred = temp_k.predict(X)
        # todo: получаем значение суммы расстояний
        result_sum_klasters = temp_k.Dist()
        # print("result_sum_klasters: ", result_sum_klasters)
        # todo: 3.3) добавляем в список сумм
        Sum_of_distance.append(result_sum_klasters)
    print("Sum_of_distance: ", Sum_of_distance)

    # todo: 2) вычислить оптимальное значение кластеров
    Optimum_klasters = float('inf')
    print("k: ", Optimum_klasters)
    min = float('inf')
    print("min: ", min)
    for i in range(2, len(Sum_of_distance) - 1):
        tmp = abs(Sum_of_distance[i] - Sum_of_distance[i + 1]) / abs(Sum_of_distance[i - 1] - Sum_of_distance[i])
        if min > tmp:
            min = tmp
            Optimum_klasters = i
    print("Оптимальное кол-во кластеров: ", Optimum_klasters)

    # todo: считаем с оптимальным значением
    GlobalItter = 1
    t = KMeans(K=Optimum_klasters, max_iters=150, plot_steps=True, GlobalItter=GlobalItter)
    new_pred = t.predict(X)
    t.plot()
