import pandas as pd
import matplotlib.pyplot as plt


def read_csv(filename):
    data = pd.read_csv(filename)
    return data


def plot(dataset):
    genders = dataset.Sex
    filter_male = genders == 'male'
    filter_female = genders == 'female'

    maleDataset = dataset.loc[filter_male]
    femaleDataset = dataset.loc[filter_female]

    # 1
    # Распределение выживших среди мужчин по классу билета.
    mdf = maleDataset[['Survived', 'Pclass']][maleDataset['Survived'] == 1].groupby('Pclass').count()
    print("-----")
    print(mdf)
    print("-----")
    mdf.plot.bar()
    plt.show()

    # Распределение выживших среди женщин по классу билета.
    fmdf = femaleDataset[['Survived', 'Pclass']][femaleDataset['Survived'] == 1].groupby('Pclass').count()
    fmdf.plot.bar()
    plt.show()

    # 2 Построить круговую диаграмму для признака "класс пассажира" (количество людей в каждом классе)

    df = maleDataset['Pclass'].value_counts()
    print(df)
    df.plot(kind="pie", label="")
    plt.show()

    # 3 Построить распределение возрастов всех пассажиров.
    Age_sort_dataset = dataset['Age'].value_counts()
    print(Age_sort_dataset)
    Age_sort_dataset.plot.bar()
    plt.show()


if __name__ == '__main__':
    file = "train_titanic.csv"
    dataset = read_csv(file)
    plot(dataset)
