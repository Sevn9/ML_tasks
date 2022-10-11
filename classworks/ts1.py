import pandas as pd
import matplotlib.pyplot as plt


def read_csv(filename):
    dataset = pd.read_csv(filename)
    return dataset


def draw(dataset):
    dates = list(dataset.Date)

    sunspots = list(dataset['Monthly Mean Total Sunspot Number'])

    plt.figure(figsize=(60, 10))
    # print(dataset[dataset.columns[2]])
    # plt.plot(dates, sunspots)
    plt.scatter(dates, sunspots)

    result = pd.DataFrame(sunspots)
    rolling_mean = result.rolling(window=25).mean()
    plt.plot(dates, rolling_mean, color='r')
    plt.show()


if __name__ == '__main__':
    file = "Sunspots.csv"
    data = read_csv(file)
    draw(data)
