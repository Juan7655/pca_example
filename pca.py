from __future__ import division
from sympy import *
import pandas
import matplotlib.pyplot as plt
import numpy as np

first_col = "x"
second_col = "y"


def main():
	data = pandas.read_csv("data.csv")
	# Keep original data for later use. Copy values to apply normalization
	data_norm = data
	# Normalization application
	data_norm[first_col] = data_norm[first_col] - data_norm[second_col].mean()
	data_norm[second_col] = data_norm[second_col] - data_norm[second_col].mean()

	# Eigenvectors for the covariance matrix
	vec = Matrix(data.cov()).eigenvects()
	eigenvectors_matrix = vec[0][2][0].col_insert(0, vec[1][2][0])

	# Rotating data points with respect to the eigenvectors
	rotated_points = Matrix(data_norm) * eigenvectors_matrix
	first_column = rotated_points.col(0).tolist()
	x_values = [min(first_column), max(first_column)]

	# Principal component axis line
	y_values = 0.99819 * np.array(x_values) + 0.10326
	plt.plot(x_values, y_values, zorder=1, c='b', linewidth=.5)

	# Original data points
	plt.scatter(data[first_col], data[second_col], zorder=2, c='b', s=10)

	# Horizontal guide-line
	plt.plot([min(first_column), max(first_column)], [0, 0], zorder=1, c='r', linewidth=.5)

	# Rotated points with principal component axis
	plt.scatter(rotated_points.col(0), rotated_points.col(1), zorder=2, c='g', s=10)

	# Projected points over principal component axis (1-dimensional)
	plt.scatter(rotated_points.col(0), zeros(1, len(rotated_points.col(0))), zorder=3, c='r', s=10)
	plt.xlabel(first_col)
	plt.ylabel(second_col)
	plt.show()


if __name__ == '__main__':
	main()
