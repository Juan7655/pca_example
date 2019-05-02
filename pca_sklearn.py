import pandas
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from decorators import plot_graph


def main():
    # import data
    data = pandas.read_csv("data_3d.csv")
    x_train, x_test, y_train, y_test = train_test_split(data[['x', 'y']], data.cat, test_size=0.01, random_state=0)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    # x_test = sc.transform(x_test)

    pca = PCA(n_components=2)
    transformed = pca.fit_transform(x_train).T
    # x_test = pca.transform(x_test)
    plot_original(data)
    plot_rotated(transformed)

    pca = PCA(n_components=1)
    transformed = pca.fit_transform(x_train).T
    # x_test = pca.transform(x_test)
    plot_original(data)
    plot_rotated(transformed)


@plot_graph(color='b', draw=False)
def plot_original(data):
    def lin_reg(df):
        x, y = df[df.columns[0]], df[df.columns[1]]
        lr = LinearRegression().fit(x.values.reshape(-1, 1), y.values.reshape(-1, 1))
        return lr.coef_[0][0], lr.intercept_[0]

    m, b = lin_reg(data)
    x_values = [min(data[data.columns[0]]), max(data[data.columns[0]])]
    y_values = list(map(lambda x: m * x + b, x_values))

    return (x_values, y_values), \
           (data[data.columns[0]], data[data.columns[1]])


@plot_graph(color='r', draw=True)
def plot_rotated(rotated_points):
    y_values = rotated_points[1] if rotated_points.shape[0] > 1 \
        else [0 for _ in range(rotated_points[0].shape[0])]
    return ([min(rotated_points[0]), max(rotated_points[0])], [0, 0]), \
           (rotated_points[0], y_values)


if __name__ == '__main__':
    main()
