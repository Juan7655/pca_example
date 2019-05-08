import numpy as np
import pandas as pd

import data_normalizer
from Utils import plot_rotated


def main():
    # import data
    data = pd.read_csv("data-large.csv")
    rotated_points = define_matrix_space(features=2, data=data)

    # plot_rotated(data, lin_reg, color='b', draw=False)
    plot_rotated(rotated_points, lin_reg, color='r', draw=True)


def define_matrix_space(features, data) -> pd.DataFrame:
    # Keep original data for later use. Copy values to apply normalization
    data_norm = data_normalizer.stretch_to_unary(data)

    # Eigenvectors for the covariance matrix
    np_vec = np.linalg.eig(data.cov())[1]

    # Dividing row-wise to normalize vectors
    np_matrix = np_vec / np_vec[0][:, None][::-1]

    # select the number of features desired after transformation
    np_matrix = np_matrix[:features].T

    # Rotating data points with respect to the eigenvectors
    rotated_points = np.matmul(data_norm, np_matrix)

    return pd.DataFrame(rotated_points, columns=data.columns[:features])


def lin_reg(df: pd.DataFrame) -> (float, float):
    x, y = df[df.columns[0]], df[df.columns[1]]
    x_mean, y_mean = x.mean(), y.mean()

    # m = ((x-X)*(y-Y))/(x-X)**2
    m_val = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
    b_val = y_mean - m_val * x_mean

    return m_val, b_val


if __name__ == '__main__':
    main()
