# TODO: Optimize linear case
# TODO: Cache errors


from functools import partial, cache

import numpy as np

kernel_dict = dict()


def add_to_kernels(name):
    def decorator(func):
        kernel_dict[name] = func
        return func

    return decorator


@add_to_kernels('linear')
def linear_kernel(x, y):
    return np.dot(x, y)


@add_to_kernels('poly')
def polynomial_kernel(x, y, c=1, q=3):
    return (c + np.dot(x, y)) ** q


@add_to_kernels('rbf')
def rbf_kernel(x, y, gamma=0.01):
    return np.exp(-gamma * np.linalg.norm(x - y) ** 2)


@add_to_kernels('laplace')
def laplace_kernel(x, y, gamma=0.01):
    return np.exp(-gamma * np.linalg.norm(x - y))


class SVM:
    """
    This class implements the soft margin Support Vector Machine classifier.
    The class implements the following methods:
        - fit: fits the model to the training data
        - predict: predicts the class of the test data
    The fit method uses the SMO algorithm to solve the dual optimization problem, rather than
    using a QP solver.
    """

    def __init__(self, C=1.0, kernel='linear', tol=0.001, seed=None, *args, **kwargs):
        """
        Constructor
        :param C: the regularization parameter
        :param kernel: the kernel function
        :param tol: stopping criterion threshold
        :param seed: random seed
        :param args: additional arguments for the kernel function
        :param kwargs: additional keyword arguments for the kernel function
        """
        self._b = None
        self._N = None
        self._y = None
        self._X = None
        self._d = None
        self.C = C
        self.tol = tol
        self.n_iter = 0
        self._dual_coef = np.array([])
        self._kernel = partial(kernel_dict.get(kernel, 'linear'), *args, **kwargs)
        self._error_cache = None

        if seed is not None:
            np.random.seed(seed)

    def evaluate(self, x):
        """
        Evaluates the SVM decision function at a given point

        :param x: point
        :return: SVM decision function value
        """
        return np.sum(self._dual_coef * self._y * np.array([self._kernel(X_s, x) for X_s in self._X])) - self._b

    def _error(self, i):
        """
        Computes the error for the i-th sample

        :param i: index of the sample
        :return: error
        """
        # return self.error_cache.get(i, self.evaluate(self._X[i]) - self._y[i])
        return self.evaluate(self._X[i]) - self._y[i]

    def _examine_example(self, i2):
        """
        Examines the i2-th sample

        :param i2: index of the sample
        :return: took step or not
        """
        y2 = self._y[i2]
        alpha2 = self._dual_coef[i2]
        E2 = self._error_cache[i2]
        r2 = E2 * y2
        if (r2 < -self.tol and alpha2 < self.C) or (r2 > self.tol and alpha2 > 0):
            # Select the second sample
            if len(self._dual_coef[(self._dual_coef > 0) & (self._dual_coef < self.C)]) > 1:
                # Select the sample that maximizes the absolute difference between the errors
                i1 = np.argmax([np.abs(E2 - self._error_cache[i1]) for i1 in np.arange(self._N)])
                if self._take_step(i1, i2):
                    return True

            # Select the second sample randomly
            non_bound = np.arange(self._N)[(self._dual_coef > 0) & (self._dual_coef < self.C)]
            if len(non_bound) > 0:
                i1 = np.random.choice(non_bound)
                if self._take_step(i1, i2):
                    return True

            # Select the second sample randomly
            i1 = np.random.choice(np.arange(self._N))
            if self._take_step(i1, i2):
                return True
        return False

    def _take_step(self, i1, i2) -> bool:
        """
        Performs the SMO algorithm step

        :param i1: first index
        :param i2: second index
        :return: took step or not
        """
        if i1 == i2:
            return False

        y1, y2 = self._y[i1], self._y[i2]
        E1, E2 = self._error_cache[i1], self._error_cache[i2]
        alpha1, alpha2 = self._dual_coef[i1], self._dual_coef[i2]
        s = y1 * y2

        L, H = (max(0., alpha2 - alpha1), min(self.C, self.C + alpha2 - alpha1)) if s == -1 \
            else (max(0., alpha1 + alpha2 - self.C), min(self.C, alpha1 + alpha2))

        if L == H:
            return False

        # Calculate eta
        k11 = self._kernel(self._X[i1], self._X[i1])
        k12 = self._kernel(self._X[i1], self._X[i2])
        k22 = self._kernel(self._X[i2], self._X[i2])
        eta = k11 + k22 - 2 * k12

        if eta > 0:
            a2 = alpha2 + y2 * (E1 - E2) / eta  # alpha2, new
            a2 = max(L, min(H, a2))  # alpha2, new, clipped
        elif eta < 0:
            print('Eta was < 0; raising an exception.')
            raise NotImplemented
        else:
            return False

        # If change was miniscule enough, return unchanged
        if abs(a2 - alpha2) < self.tol * (a2 + alpha2 + self.tol):
            return False

        # Update alpha1
        a1 = alpha1 + s * (alpha2 - a2)

        # Calculate threshold to reflect change in Lagrange multipliers
        b1 = self._b + E1 + y1 * k11 * (a1 - alpha1) + y2 * k12 * (a2 - alpha2)
        b2 = self._b + E2 + y1 * k12 * (a1 - alpha1) + y2 * k22 * (a2 - alpha2)
        b_new = b1 if 0 < a1 < self.C else b2 if 0 < a2 < self.C else (b1 + b2) / 2.

        # Update error cache for optimized alphas is set to 0 if they're unbound
        for index, alpha in zip([i1, i2], [a1, a2]):
            if 0.0 < alpha < self.C:
                self._error_cache[index] = 0.0

        # Set non-optimized errors based on equation 12.11 in Platt's book
        for n in range(self._N):
            if n != i1 and n != i2:
                self._error_cache[n] = self._error_cache[n] + \
                                        y1 * (a1 - alpha1) * self._kernel(self._X[i1], self._X[n]) + \
                                        y2 * (a2 - alpha2) * self._kernel(self._X[i2], self._X[n]) + self._b - b_new

        # Update threshold
        self._b = b_new

        # Store new Lagrange multipliers
        self._dual_coef[i1], self._dual_coef[i2] = a1, a2

        # Return 1 (True) and ++iters
        self.n_iter += 1
        return True

    def fit(self, X, y):
        """
        Fits the model to the training data

        :param X: training data
        :param y: training labels
        :return: None
        """
        # Initialize the number of training samples and the dimensionality of the data
        self._X = X
        self._y = y
        self._N, self._d = X.shape
        self._b = 0

        # Initialize the dual coefficients
        self._dual_coef = np.zeros(self._N)

        # Initialize the error cache
        self._error_cache = np.array([self._error(i) for i in range(self._N)])

        num_changed = 0
        examine_all = True
        while num_changed > 0 or examine_all:
            num_changed = 0
            if examine_all:
                for i in range(self._N):
                    num_changed += int(self._examine_example(i))
            else:
                for i in np.arange(self._N)[(self._dual_coef > 0) & (self._dual_coef < self.C)]:
                    num_changed += int(self._examine_example(i))
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

    def predict(self, x):
        """
        Predicts the class of the test data

        :param x: test data
        :return: predicted labels
        """
        return np.sign(self.evaluate(x))

    @property
    def support_vectors(self):
        return self._X[self._dual_coef != 0.]

    @property
    def dual_coef(self):
        return self._dual_coef

    @property
    def weights(self):
        return np.sum([self._dual_coef[i] * self._y[i] * self._X[i] for i in np.arange(self._N)], axis=0)

    @property
    def b(self):
        return -self._b
