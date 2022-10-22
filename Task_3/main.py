import numpy as np
import pygame
import random


# todo: right version dbscan

def near_by_points(points, idx, epsilon):
    near = []
    for point_idx in range(0, len(points)):
        if np.linalg.norm(points[idx] - points[point_idx]) < epsilon:
            near.append(point_idx)
    return near


def draw(list_col, clusters):
    for point, cluster in zip(list_col, clusters):
        if dic.get(cluster) == None:
            color = "#" + ''.join([random.choice('0123456789ABCDEF') for i in range(6)])
            print("color: ", color)
            dic[cluster] = color

            print(dic)
        radius = 8
        if cluster == -1:
            pygame.draw.circle(screen, "Black", point, radius)
        else:
            pygame.draw.circle(screen, dic.get(cluster), point, radius)


def algorithm(_points, _epsilon):
    labels = [0] * len(_points)  # создаем пустой лист на основе количестива точек
    cluster_idx = 0
    minimum_points = 3
    for i in range(0, len(_points)):
        if not (labels[i] == 0):  # если точка не равна нулю пропускаем
            continue
        near_points = near_by_points(_points, i, _epsilon)  # если равна то находим ближайшие точки
        if len(near_points) < minimum_points:  # если такая точка одна то присваиваем шум
            labels[i] = -1
        else:  # если таких точек несколько то создаем новый кластер
            cluster_idx += 1
            labels[i] = cluster_idx  # и присваеваем его к точке
            i = 0
            while i < len(near_points):  # проходи по близким точкам
                point = near_points[i]
                if labels[point] == -1:
                    labels[point] = cluster_idx

                elif labels[point] == 0:
                    labels[point] = cluster_idx
                    point_near = near_by_points(_points, point, _epsilon)
                    if len(point_near) >= minimum_points:
                        near_points = near_points + point_near
                i += 1
    return labels


pygame.init()
screen = pygame.display.set_mode((600, 400))
screen.fill('WHITE')
pygame.display.update()
FPS = 60
clock = pygame.time.Clock()
dic = {}
points = []

epsilon = 40

play = True
while play:
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            play = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            points.append(pygame.mouse.get_pos())
            screen.fill('WHITE')
            prediction = algorithm(np.array(points), epsilon)
            draw(points, prediction)

    pygame.display.update()
