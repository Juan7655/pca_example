import pandas
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Utils import plot_rotated


def main():
    # import data
    data = pandas.read_csv("data/data-large_3d.csv")
    # data = pandas.read_csv("data/data_3d.csv")
    x_train, x_test, y_train, y_test = train_test_split(data[['x', 'y']], data.cat, test_size=0.01, random_state=0)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    # x_test = sc.transform(x_test)

    lda = LDA(n_components=2)
    transformed = pandas.DataFrame(lda.fit_transform(x_train, y_train))
    # x_test = pca.transform(x_test)
    # plot_rotated(data, lin_reg, color='b', draw=False)
    plot_rotated(transformed[transformed[0] < 0.25], lin_reg, color='b', draw=False)
    plot_rotated(transformed[transformed[0] >= 0.25], lin_reg, color='r', draw=True)


def lin_reg(df) -> (float, float):
    x, y = df[df.columns[0]], df[df.columns[1]]
    lr = LinearRegression().fit(x.values.reshape(-1, 1), y.values.reshape(-1, 1))
    return lr.coef_[0][0], lr.intercept_[0]


if __name__ == '__main__':
    main()
