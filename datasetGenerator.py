from typing import Tuple, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Vector = np.array


def generate_linear_lambda(dim: int, noise: float = 0) -> Tuple[Vector, Callable[[Vector], float]]:
    """
    Generates a linear function of the form f(x) = w_0 + w_1 * x_1 + ... + w_n * x_n
    :param dim: the dimension of the weight vector
    :param noise: the variance of the normal noise to add to the function
    :return: the weights and the function (as a lambda expression)
    """
    weights = np.random.uniform(low=-1, high=1, size=(dim,))
    bias = np.random.uniform(low=-0.5, high=0.5)

    return np.insert(weights, 0, bias, axis=0), lambda x: (x.T.dot(weights) + bias + np.random.normal(0, noise))


def generate_linear_separable_dataframe(shape: Tuple[int, int], seed: int = None, *args, **kwargs) -> \
        Tuple[Vector, Callable[[Vector], float], pd.DataFrame]:
    """
    Generates a linearly separable dataset, i.e. a dataset where the labels are 1 if the point is above the
    linear line and -1 otherwise
    :param seed: the random seed to use for generating the dataset
    :param shape: the shape of the dataset (N x d)
    :param args: arguments to pass to the linear function generator
    :param kwargs: keyword arguments to pass to the linear function generator
    :return: the weights, the function and the dataset (as a pandas dataframe)
    """
    if seed is not None:
        np.random.seed(seed)

    data = np.random.uniform(low=-1, high=1, size=shape)
    col_names = [f'x{i + 1}' for i in range(shape[1])]
    _df = pd.DataFrame(data=data, columns=col_names)
    weights, f = generate_linear_lambda(dim=shape[1], *args, **kwargs)
    _df['class'] = pd.Categorical(_df.apply(axis=1, func=lambda r: 1 if f(r) > 0 else -1))
    return weights, f, _df


def plot_line(w: Vector, llim, hlim, *args, **kwargs) -> None:
    """
    Plots a line given the weight 3-vector. The line is plotted in the range [-1, 1].
    :param llim: lower limit
    :param hlim: higher limit
    :param w: the weight vector
    :param args: arguments to pass to the plot function
    :param kwargs: keyword arguments to pass to the plot
    """
    assert w.shape[0] == 3

    xs = np.linspace(llim, hlim, 10)
    ys = -(w[0] + w[1] * xs) / w[2]
    plt.xlim([llim, hlim])
    plt.ylim([llim, hlim])
    plt.plot(xs, ys, *args, **kwargs)
