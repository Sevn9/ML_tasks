import numpy as np
from sklearn.datasets import make_blobs
import networkx as nx
from pyvis.network import Network
import webbrowser
import random


def print_hi(name):
    print(f'Hi, {name}')


class KNP:
    def __init__(self, points, k=3):
        # todo: Points - точки
        self.Points = points
        # todo: K - кол-во кластеров
        self.K = k
        # todo: граф
        self.graph = []
        random.seed(20)
        self.T = []  # список ребер остова

    # Minimum Spanning Tree
    def create_mst(self):
        # todo: генерируем дерево
        # генерируем веса
        fix = round(len(self.Points) / 2)
        print("fix: ", fix)
        for i in range(len(self.Points)):
            for j in range(len(self.Points)):
                if i != j and j < fix - 1:
                    w = random.randint(5, 100)
                    # print("Сгенерированный вес --w : ", w)
                    self.graph.append([w, i, j])
        print("Сгенерированный граф с весами: ", self.graph)

        Rs = sorted(self.graph, key=lambda x: x[0])
        U = set()  # список соединенных вершин
        D = {}  # словарь списка изолированных групп вершин

        for r in Rs:
            if r[1] not in U or r[2] not in U:  # проверка для исключения циклов в остове
                if r[1] not in U and r[2] not in U:  # если обе вершины не соединены, то
                    D[r[1]] = [r[1], r[2]]  # формируем в словаре ключ с номерами вершин
                    D[r[2]] = D[r[1]]  # и связываем их с одним и тем же списком вершин
                else:  # иначе
                    if not D.get(r[1]):  # если в словаре нет первой вершины, то
                        D[r[2]].append(r[1])  # добавляем в список первую вершину
                        D[r[1]] = D[r[2]]  # и добавляем ключ с номером первой вершины
                    else:
                        D[r[1]].append(r[2])  # иначе, все то же самое делаем со второй вершиной
                        D[r[2]] = D[r[1]]

                self.T.append(r)  # добавляем ребро в остов
                U.add(r[1])  # добавляем вершины в множество U
                U.add(r[2])

        for r in Rs:  # проходим по ребрам второй раз и объединяем разрозненные группы вершин
            if r[2] not in D[r[1]]:  # если вершины принадлежат разным группам, то объединяем
                self.T.append(r)  # добавляем ребро в остов
                gr1 = D[r[1]]
                D[r[1]] += D[r[2]]  # объединим списки двух групп вершин
                D[r[2]] += gr1
        print("ребра в остове: ", self.T)
        return self.T

    def create_clasters(self):
        num_ribs_remove = self.K - 1
        print("Небходимо удалить ребер: ", num_ribs_remove)

        for i in range(num_ribs_remove):
            min_value = 0
            idx_rib = 0
            print("---------------------------------------")
            for idx_edge, edge in enumerate(self.T):
                print("размер self.T: ", len(self.T))
                if idx_edge == 0:
                    min_value = edge[0]
                    idx_rib = 0
                print("idx_edge: ", idx_edge)
                print("edge: ", edge)
                if edge[2] <= min_value:
                    min_value = edge[0]
                    idx_rib = idx_edge
                    print("new min_value: ", min_value)
            print("result min_value: ", min_value)
            print("result idx_rib: ", idx_rib)
            # todo: удаляем найденный минимальный остов
            self.T.pop(idx_rib)
        print("result clusters rebra: ", self.T)
        return self.T


# todo: Алгоритм кратчайшего незамкнутого пути
if __name__ == '__main__':
    # todo: генерируем точки
    np.random.seed(42)
    X, y = make_blobs(10, random_state=0)
    print("кол-во точек: ", len(X))
    # print(X.shape)

    # todo: передаем точки
    graph = KNP(points=X)
    gr_ribs = graph.create_mst()
    # todo: список ребер графа (длина, вершина 1, вершина 2)
    print("main gr_ribs: ", gr_ribs)

    net = Network('1000px', directed=True)
    # todo: рисуем остовной граф
    # рисуем точки
    for i, p in enumerate(X):
        net.add_node(i, label=f'{i}', x=p[0], y=p[1])
    # рисуем ребра
    for idx, rib in enumerate(gr_ribs):
        net.add_edge(rib[1], rib[2], weight=rib[0])

    url = 'nx.html'  # open in new tab
    net.show('nx.html')
    webbrowser.open(url, new=2)

    # todo: создаем кластеры
    clusters = graph.create_clasters()

    net2 = Network('1000px', directed=True)

    # рисуем точки
    for i2, p2 in enumerate(X):
        net2.add_node(i2, label=f'{i2}', x=p2[0], y=p2[1])
    # рисуем ребра
    for idx_cl, cl in enumerate(clusters):
        net2.add_edge(cl[1], cl[2], weight=cl[0])

    url = 'nx2.html'  # open in new tab
    net2.show('nx2.html')
    webbrowser.open(url, new=2)
