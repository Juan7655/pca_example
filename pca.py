from __future__ import division
from sympy import *
import pandas
import matplotlib.pyplot as plt
import numpy as np


def plot_graph(color, draw=False):
    def wrapper(fun):
        def dec(*args):
            set1, set2 = fun(*args)

            plt.plot(set1[0], set1[1], zorder=1, c=color, linewidth=.5)
            plt.scatter(set2[0], set2[1], zorder=3, c=color, s=10)

            if draw:
                plt.xlabel('x')
                plt.ylabel('y')
                plt.show()

        return dec

    return wrapper


def main():
    # import data
    data = pandas.read_csv("data.csv")

    # Keep original data for later use. Copy values to apply normalization
    data_norm = data

    # Normalization application for every column
    for col in data_norm:
        data_norm[col] = data_norm[col] - data_norm[col].mean()

    # Eigenvectors for the covariance matrix
    vec = Matrix(data.cov()).eigenvects()
    eigenvectors_matrix = vec[0][2][0].col_insert(0, vec[1][2][0])

    # Rotating data points with respect to the eigenvectors
    rotated_points = Matrix(data_norm) * eigenvectors_matrix

    plot_original(data)
    plot_rotated(rotated_points)
    plot_rotated_flatten(rotated_points)


@plot_graph(color='b', draw=True)
def plot_original(data):
    def lin_reg(df):
        x, y = df[df.columns[0]], df[df.columns[1]]
        x_mean, y_mean = x.mean(), y.mean()

        # m = ((x-X)*(y-Y))/(x-X)**2
        m_val = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
        b_val = y_mean - m_val * x_mean

        return m_val, b_val

    m, b = lin_reg(data)
    x_values = [min(data[data.columns[0]]), max(data[data.columns[0]])]
    y_values = m * np.array(x_values) + b

    return (x_values, y_values), \
           (data[data.columns[0]], data[data.columns[1]])


@plot_graph(color='g', draw=True)
def plot_rotated(rotated_points):
    first_column = rotated_points.col(0).tolist()

    # Rotated points with principal component axis
    return ([min(first_column), max(first_column)], [0, 0]), \
           (rotated_points.col(0), rotated_points.col(1))


@plot_graph(color='r', draw=True)
def plot_rotated_flatten(rotated_points):
    first_column = rotated_points.col(0).tolist()

    return ([min(first_column), max(first_column)], [0, 0]),\
           (rotated_points.col(0), zeros(1, len(rotated_points.col(0))))


if __name__ == '__main__':
    main()
