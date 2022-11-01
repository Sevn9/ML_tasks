import pandas as pd
import numpy as np


class NaiveBayesClassifier:
    def __init__(self):
        # вероятности заболеть
        self.class_probabilities = {}
        # условные вероятности заболеть
        self.conditional_probabilities = {}

    def load(self, symptom, disease, disease_total):
        for cls_name, cls_count in disease.iterrows():
            print("load -- cls_name: ", cls_name)
            print("load -- cls_count: ", cls_count)
            # достаем значение (кол-во пациентов с данной болезнью) и делим на всего заболевших
            print("load -- cls_count[0]: ", cls_count[0])
            # получаем вероятность заболеть
            self.class_probabilities[cls_name] = cls_count[0] / disease_total
            print("load -- self.class_probabilities[cls_name]: ", self.class_probabilities[cls_name])

        print("load ---------------------------------------------------------------------------")

        for feature_name, feature_probs in symptom.iterrows():
            print("load -- feature_name: ", feature_name)
            # пронумеруем поиндексно и выведем в строку (1) Недомогание;0.97;0.96;1;0.92;0.91;1;1;1;0.9
            print("load -- feature_probs: \n", feature_probs)
            # условные вероятности
            self.conditional_probabilities[feature_name] = {}
            print("self.conditional_probabilities[feature_name]: ", self.conditional_probabilities[feature_name])

            for cls_name, feature_cls_prob in enumerate(feature_probs):
                print("load -- cls_name: ", cls_name)
                print("load -- feature_cls_prob: ", feature_cls_prob)
                self.conditional_probabilities[feature_name][cls_name] = feature_cls_prob
                print("load -- self.conditional_probabilities[feature_name][cls_name]: ",
                      self.conditional_probabilities[feature_name][cls_name])
        print("self.conditional_probabilities: \n", self.conditional_probabilities)
        print("self.class_probabilities: \n", self.class_probabilities)

    def get_prob_for_class(self, vec, cls):
        result = self.class_probabilities[cls]
        print("result: -- result", result)
        for idx, value in enumerate(vec):
            if value:
                result *= self.conditional_probabilities.get(idx, {}).get(cls, 0)
                # print("result: -- self.conditional_probabilities.get(idx, {}) ",
                #     self.conditional_probabilities.get(idx, {}))
                # print("self.conditional_probabilities.get(idx, {}).get(cls, 0) ",
                #      self.conditional_probabilities.get(idx, {}).get(cls, 0))
        return result

    # todo: функция предсказания
    def predict(self, X):
        # Получаем вектор(ы) пациента(ов) X
        result = []
        for x in X:
            max_y = None
            max_prob = 0
            # по каждому ключевому значению
            for cls in self.class_probabilities.keys():
                # print("predict -- cls: ", cls)
                # print("predict -- x: ", x)
                prob = self.get_prob_for_class(x, cls)
                # print("predict -- prob: ", prob)
                # ищем наиболее вероятное значение
                if prob >= max_prob:
                    max_y = cls
                    max_prob = prob
            result.append(max_y)

        return result


if __name__ == '__main__':
    symptom = pd.read_csv('symptom.csv', sep=';', index_col=0)
    print("Таблица с симптомами: \n", symptom)
    disease_names = symptom.columns
    symptom_names = symptom.index
    symptom.columns = list(range(len(symptom.columns)))
    symptom = symptom.reset_index(drop=True)
    print("Таблица с симптомами matrix: \n", symptom)

    disease = pd.read_csv('disease.csv', sep=';', index_col=0)
    print("Болезни: \n", disease)

    total_disease = disease.iloc[-1][0]
    # print("Всего пациентов: \n", total_disease)
    disease = disease[:9]
    # print("без итого: \n", disease)
    disease = disease.reset_index(drop=True)
    # print("С индексами: \n", disease)

    # todo: создаем тестового пациента
    X_test = np.random.randint(0, 2, size=(1, len(symptom)))
    print(X_test)
    # [[1 1 0 0 1 1 1 0 0 1 0 1 0 1 0 0 1 0 0 1 1 1 1]]
    print("болезни")
    print(symptom.iterrows())
    for feature_name, feature_probs in disease.iterrows():
        # болезни
        if (X_test[0][feature_name] == 1):
            print(feature_probs)

    # todo: создаем модель
    NBayesModel = NaiveBayesClassifier()
    NBayesModel.load(symptom, disease, total_disease)

    y_pred = NBayesModel.predict(X_test)
    print(y_pred)
    print("Ответ: ", list(map(lambda x: disease_names[x].strip(), y_pred)))
