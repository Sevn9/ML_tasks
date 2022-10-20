import pygame
import numpy as np
import random


def dist(pntA, pntB):
    return np.sqrt((pntA[0] - pntB[0]) ** 2 + (pntA[1] - pntB[1]) ** 2)


def dbscan(points):
    eps = 30
    #  todo: точки шум
    NOISE = []
    print("dbscan -- points: ", points)
    #  todo: найдем все core точки
    all_core_points, not_tagged_points = set_flags(points)
    #  todo: получаем индексы точек соответствующие условию по соседям all_core_points
    #  todo: и список индексов не помеченных точек not_tagged_points

    #   todo: находим все крайние точки и шум
    none_core_points = []
    for ntp_id, ntp in enumerate(not_tagged_points):
        #  todo: если хотя бы один сосед ntp2 зеленый то выдаем желтый флаг
        flag = 0
        for acp_id, acp in enumerate(all_core_points):
            if dist(points[ntp], points[acp]) < eps and points[ntp] != points[acp]:
                flag = 1
                break
            else:
                flag = 0
        if flag == 1:
            none_core_points.append(ntp)
        else:
            NOISE.append(ntp)
    print("none_core_points_2: ", none_core_points)
    print("NOISE2: ", NOISE)

    return all_core_points, none_core_points, NOISE


def set_flags(points):
    # todo: eps - радиус вокруг точки в области которого могут находиться другие точки
    eps = 30
    # todo: минимальное кол-во точек в радиусе одной точки для образования кластера
    minPts = 3
    flags = []
    others_points = []
    for i, pnt in enumerate(points):
        neighb = 0
        # todo: для каждой точки из списка проходимся по точкам из того же списка: aa ab ac и тд
        for p in points:
            #  todo: если дистанция между точками a и b меньше радиусной и точка не равна сама себе
            if dist(pnt, p) < eps and pnt != p:
                #  todo: добавляем соседа для точки a
                neighb += 1
        #  todo: если кол-во соседних точек для этой точки больше или равно минимальному кол-ву соседей
        if neighb >= minPts:
            #  todo: добавляем индекс этой точки в список помеченных как core точки
            flags.append(i)
        else:
            #  todo: если не подходит добавляем индекс этой точки в список остальных точек
            others_points.append(i)
    #  todo: возвращаем индексы точек соответствующие условию по соседям
    return flags, others_points


def draw():
    print('draw')
    # todo: радиус точки
    R = 7
    points = []
    pygame.init()
    # todo: настройка отображения
    screen = pygame.display.set_mode([800, 600])
    screen.fill(color='white')
    while True:
        for event in pygame.event.get():
            # todo: функционал выхода
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            # todo: -------------------------
            # print(event)
            if event.type == pygame.MOUSEBUTTONDOWN:
                #  todo: если нажимаем лкм
                if event.button == 1:
                    #  todo: рисуем круги
                    pygame.draw.circle(screen, 'black', event.pos, R)
                    # todo: добавляем в список координаты нарисованной точки
                    points.append(event.pos)
                    # print(type(np.array(event.pos)))
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    print(1)
                    id_clusters_points, n_core_points, noize_points = dbscan(points)
                    for ic, idc in enumerate(id_clusters_points):
                        pygame.draw.circle(screen, 'green', points[idc], R)
                    for id_ncp, ncp_p in enumerate(n_core_points):
                        pygame.draw.circle(screen, "yellow", points[ncp_p], R)
                    for id_noize, noize_p in enumerate(noize_points):
                        pygame.draw.circle(screen, "red", points[noize_p], R)
            pygame.display.update()


# todo: dbscan
if __name__ == '__main__':
    draw()
