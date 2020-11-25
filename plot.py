from matplotlib import pyplot
import numpy as np
import pickle


def plot_history(hist, title=''):
    hist = np.asarray(hist)
    hist = hist.transpose()
    order = [2, 1, 0, 3]
    color_order = ('yellow', 'red', 'green', 'blue')
    hist = hist[order]
    pyplot.stackplot(range(len(hist[0])), hist, colors=color_order)
    pyplot.title(title)
    pyplot.show()


if __name__ == '__main__':
    with open('results/test.p', 'rb') as f:
        history = pickle.load(f)
    plot_history(history)
