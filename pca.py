import numpy as np
import pandas as pd

import data_normalizer
from decorators import plot_graph


def main():
    # import data
    data = pd.read_csv("data.csv")
    rotated_points = define_matrix_space(features=2, data=data)

    plot_original(data)
    plot_rotated(rotated_points)


def define_matrix_space(features, data):
    # Keep original data for later use. Copy values to apply normalization
    data_norm = data_normalizer.stretch_and_recenter(data)

    # Eigenvectors for the covariance matrix
    np_vec = np.linalg.eig(data.cov())[1]
    np_matrix = np_vec / np.array(np_vec.T[0])[:, None][::-1]
    np_matrix = np_matrix[:features].T

    # Rotating data points with respect to the eigenvectors
    rotated_points = np.matmul(data_norm, np_matrix)
    mult = np.identity(features)
    mult[0] *= -1
    # rotated_points = np.matmul(rotated_points, [[-1, 0], [0, 1]])
    # rotated_points = np.matmul(rotated_points, [-1])
    rotated_points = np.matmul(rotated_points, mult)

    return pd.DataFrame(rotated_points, columns=data.columns[:features])


@plot_graph(color='b', draw=False)
def plot_original(data):
    m, b = lin_reg(data)
    x_values = [min(data[data.columns[0]]), max(data[data.columns[0]])]
    y_values = list(map(lambda x: m * x + b, x_values))

    return (x_values, y_values), (data[data.columns[0]], data[data.columns[1]])


@plot_graph(color='r', draw=True)
def plot_rotated(df):
    if df.shape[1] > 1:
        y_values = df[df.columns[1]]
        # m, b = lin_reg(df)
        m, b = 0, 0
    else:
        y_values = df - df
        m, b = lin_reg(df.assign(name=0))

    x_values = [min(df[df.columns[0]]), max(df[df.columns[0]])]
    y_trendline = list(map(lambda x: m * x + b, x_values))

    return (x_values, y_trendline), (df[df.columns[0]], y_values)


def lin_reg(df: pd.DataFrame):
    x, y = df[df.columns[0]], df[df.columns[1]]
    x_mean, y_mean = x.mean(), y.mean()

    # m = ((x-X)*(y-Y))/(x-X)**2
    m_val = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
    b_val = y_mean - m_val * x_mean

    return m_val, b_val


if __name__ == '__main__':
    main()
