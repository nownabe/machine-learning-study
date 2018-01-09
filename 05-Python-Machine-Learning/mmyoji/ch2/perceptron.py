import numpy as np

class Perceptron(object):
    """ Perceptron classifier

    Parameters
    ------

    eta : float
      learning rate (0.0..1.0)

    n_iter : int
      # of training for training data

    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """ Fit to training data

        Params
        ------

        X : array_like, shape = [n_samples, n_features]
            training data
            n_samples: # of samples
            n_features: # of features

        y : array_like, shape = [n_samples]


        Return value
        ------

        self : object

        """

        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0

            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi

                self.w_[0] += update

                errors += int(update != 0.0)

            self.errors_.append(errors)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]


    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
