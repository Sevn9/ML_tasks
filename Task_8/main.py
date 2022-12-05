import random as r
import numpy as np


# todo: a+2b+3c+4d+5e=100
class diophantGenAlg:
    def __init__(self, coefs, answer):
        # коэффициенты уравнения
        self.coefs = np.array(coefs)
        # ответ уравнения
        self.answer = answer
        # шаблон генома
        self.args = np.zeros(len(coefs))

    def try_args(self, args):
        #  подставим каждое решение в выражение a+2b+3c+4d+5e
        return np.dot(self.coefs, np.array(args).transpose())

    # Расстояние от полученного значения до 100 и будет нужным значением
    def mistake(self, args):
        return abs(self.try_args(args) - self.answer)

    def solve(self, generation=6):
        # generation это кол-во родителей (начальная популяция)
        # для подсчета поколений
        e = 0
        # кол-во коэффициентов уравнения
        length = len(self.coefs)

        # генерируем список случайных решений
        genomes = [[r.randint(1, self.answer) for i in range(length)] for i in range(generation)]
        print("genomes: ", genomes)
        # флаг генерирующий число 1 или 0
        bin = lambda: r.randint(0, 1)

        # функция выбора родителя
        def parent(res):
            rt = r.randint(0, 10000)
            if rt < res[0]:
                return 0
            for i in range(1, len(res)):
                if res[i - 1] < rt < res[i]:
                    return i

        while True:
            # вычисляем для каждого генома коэффициенты выживаемости
            mistakes = [self.mistake(i) for i in genomes]
            print("mistakes: ", mistakes)
            sum = 0.0
            for m in range(len(mistakes)):
                # одна хромосома в конце концов достигнет коэффициента выживаемости 0, то есть станет решением.
                if mistakes[m] == 0:
                    self.args = np.array(genomes[m])
                    return genomes[m]
                # в ином случае:
                sum += 1 / mistakes[m]
            # находим "подходящесть" (Вероятность оказаться родителем) (выживаемость)
            alives = [int((1 / m) / sum * 10000) for m in mistakes]
            # print("alives: ", alives)
            res = [0 for i in range(generation)]
            # print("res: ", res)
            res[0] = alives[0]
            # выбираем родителей
            for i in range(1, len(alives)):
                res[i] = res[i - 1] + alives[i]
            parents = []
            # от 0 до 6
            for i in range(generation):
                # выбираем родителей
                first = parent(res)
                while first is None:
                    first = parent(res)
                second = first
                while second == first:
                    second = parent(res)
                    while second is None:
                        second = parent(res)
                # print("first: ", first)
                # print("second: ", second)
                # добавляем в список пар родителей (6 штук)
                parents.append((first, second))
            # выбираем геном с самой большой выживаемостью
            s = genomes[np.argmax(alives)]
            # print("alives: ", alives)
            # print("np.argmax(alives): ", np.argmax(alives))
            # print("genomes: ", genomes)
            # print("s: ", s)
            # для каждой пары родителей перезаписываем новые геномы (создаем новое поколение)
            # j % 2 остаток от деления на 2, length - кол-во коэффициентов уравнения т.е. 5
            print("parents: ", parents)
            genomes = [[genomes[f[j % 2]][j] for j in range(length)] for f in parents]
            print("genomes result: ", genomes)
            # создание мутаций
            for v in genomes:
                if bin() == 1:
                    # берем 1 геном, выбираем рандомное число из него и заменяем рандомным от 1 до 100
                    v[r.randint(0, length - 1)] = r.randint(1, self.answer)
            # присваиваем геном с самой большой выживаемостью по индексу 0
            genomes[0] = s
            e += 1


if __name__ == '__main__':
    dga = diophantGenAlg([1, 2, 3, 4, 5], 100)
    print("Ответ на a+2b+3c+4d+5e=100: ", dga.solve())
