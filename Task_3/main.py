import pygame
import numpy as np

def dist(pntA, pntB):
    return np.sqrt((pntA[0] - pntB[0]) ** 2 + (pntA[1] - pntB[1]) ** 2)

def dbscan():
    pass

def set_flags(points):
    eps = 20
    minPts = 3
    flags = np.zeros(len(points))
    for pnt, i in enumerate(points):
        neighb = 0
        for p in points:
            if dist(pnt, p) < eps and pnt != p:
                neighb += 1


def draw():
    print('draw')
    R = 7
    points = []
    pygame.init()
    screen = pygame.display.set_mode([800, 600])
    screen.fill(color='white')
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            print(event)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    pygame.draw.circle(screen, 'black', event.pos, R)
                    points.append(event.pos)
                    print(type(np.array(event.pos)))
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    print(1)
            pygame.display.update()


if __name__ == '__main__':
    draw()
