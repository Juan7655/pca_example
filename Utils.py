import types

import matplotlib.pyplot as plt
from pandas import DataFrame


def plot_graph(fun: types.FunctionType):
    def plotter(*args, **kwargs):
            set1, set2 = fun(*args, **kwargs)
            plt.plot(set1[0], set1[1], zorder=2, c=kwargs['color'], linewidth=.5)
            plt.scatter(set2[0], set2[1], zorder=3, c=kwargs['color'], s=10)
            if kwargs['draw']:
                plt.xlabel('x')
                plt.ylabel('y')
                plt.show()
    return plotter


@plot_graph
def plot_rotated(df: DataFrame, lin_reg: types.FunctionType, *args, **kwargs):
    if df.shape[1] > 1:
        y_values = df[df.columns[1]]
        m, b = lin_reg(df)
    else:
        y_values = df - df
        m, b = lin_reg(df.assign(name=0))

    x_values = [min(df[df.columns[0]]), max(df[df.columns[0]])]
    y_trendline = list(map(lambda x: m * x + b, x_values))

    return (x_values, y_trendline), (df[df.columns[0]], y_values)
