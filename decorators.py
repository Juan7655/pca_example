import matplotlib.pyplot as plt


def plot_graph(color, draw=False):
    def wrapper(fun):
        def plotter(*args):
            set1, set2 = fun(*args)
            plt.plot(set1[0], set1[1], zorder=1, c=color, linewidth=.5)
            plt.scatter(set2[0], set2[1], zorder=3, c=color, s=10)
            if draw:
                plt.xlabel('x')
                plt.ylabel('y')
                plt.show()
        return plotter
    return wrapper
