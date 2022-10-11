import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# функция евклидова расстояния
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KMeans:
    # K - кол-во кластеров max_iters - максимальное кол-во иттераций для оптимизации
    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # Список выборочных индексов для каждого кластера
        self.clusters = [[] for _ in range(self.K)]
        # средний вектор признаков для каждого кластера
        self.centroids = []

    def predict(self, X):
        self.X = X
        # число выборок и ко-во функций
        self.n_samples, self.n_features = X.shape

        # Инициализация ценроидов
        random_samples_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_samples_idxs]

        # Оптимизация (в диапазоне max itter)
        for _ in range(self.max_iters):
            # Обновляем кластеры
            self.clusters = self._create_clusters(self.centroids)
            # строим снова график после обновления кластеров
            if self.plot_steps:
                self.plot()
            # обновляем центроиды
            centroids_old = self.centroids
            # среднее значение кластеров к центроиду (для каждого кластера вычисляем среднее значения)
            self.centroids = self._get_centroids(self.clusters)
            # строим снова график после обновления центроидов
            if self.plot_steps:
                self.plot()

            # проверка сходимости расчитывает расстояние между каждым старым и новым центроидами для всех центроидов
            # и проверяет равно ли это 0
            if self._is_converged(centroids_old, self.centroids):
                break
        # вернуть метки кластера
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        # для каждой выборки возвращаем кластер которому он был назначен
        for clusters_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = clusters_idx
        return labels

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        # для индекса и выборки перечислить точку X
        for idx, sample in enumerate(self.X):
            # получить ближайший центроид
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    # функция определения ближайшего центроида
    def _closest_centroid(self, sample, centroids):
        # расстояние выборки и каждого центроида
        distances = [euclidean_distance(sample, point) for point in centroids]
        # индекс с минимальным расстоянием (ближайший индекс равен numpy.argmin этих расстояний
        closest_idx = np.argmin(distances)
        return closest_idx

    # среднее значение кластеров к центроиду (для каждого кластера вычисляем среднее значения)
    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            # назначаем текущему кластеру среднее значение
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def plot(self):
        fig, ax = plt.subplots(figsize=(8, 4))
        # перебираем кластеры
        for i, index in enumerate(self.clusters):
            # получаем текущие точки равные собственной точке х
            point = self.X[index].T
            # рисуем все точки и для каждого кластера свой цвет
            ax.scatter(*point)
            # наносим все центроиды
        for point in self.centroids:
            ax.scatter(*point, marker='x', color="black", linewidth=2)
        plt.show()


# k means
if __name__ == '__main__':
    np.random.seed(42)
    X, y = make_blobs(200, random_state=0)
    print(X.shape)

    clusters = len(np.unique(y))
    print(clusters)

    k = KMeans(K=clusters, max_iters=150, plot_steps=True)
    y_pred = k.predict(X)
    k.plot()
