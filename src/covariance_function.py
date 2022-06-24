import numpy as np
import matplotlib.pyplot as plt


class CovarianceFunction():
    """
    Covariance function class

    Attributes
    ----------

    Methods
    -------

    Notes
    -----
    """

    mm_to_m = 1e-3  # class constants

    def __init__(self):
        pass

    def visualise(self, C):
        plt.imshow(C)


class JCSS(CovarianceFunction):
    """
    JCSS probabilistic model code covariance function class
    (Joint Commission on Structural Safety)

    Attributes
    ----------

    Methods
    -------

    Notes
    -----
    """

    def __init__(self, lc, rho):
        """
        Initialise an instance of the JCSS covariance function class

        Parameters
        ----------
        lc : float
            Correlation length (or length scale)

        rho : float
            Threshold value for the correlation between two points in space

        Returns
        -------

        """
        self.lc = lc * self.mm_to_m
        self.rho = rho

    def build_correlation_matrix(self, x):

        C = np.zeros([len(x), len(x)])

        for i in range(len(x)):
            for j in range(i + 1):
                d = np.linalg.norm(x[i, :] - x[j, :])
                C[i, j] = self.rho + (1 - self.rho) * np.exp(-d / self.lc)
                C[j, i] = C[i, j]

        return C


class Exponential(CovarianceFunction):
    """
    Exponential covariance function class

    Attributes
    ----------

    Methods
    -------

    Notes
    -----
    """

    def __init__(self, lc, sigma=1):
        """
        Initialise an instance of the exponential covariance function class

        Parameters
        ----------
        lc : float
            Correlation length (or length scale)

        sigma : float
            Marginal standard deviation (optional)

        Returns
        -------

        """
        self.lc = lc * self.mm_to_m
        self.sigma = sigma

    def build_correlation_matrix(self, x):

        C = np.zeros([len(x), len(x)])

        for i in range(len(x)):
            for j in range(i + 1):
                d = np.linalg.norm(x[i, :] - x[j, :])
                C[i, j] = self.sigma * np.exp(-d / self.lc)
                C[j, i] = C[i, j]

        return C


class Gaussian(CovarianceFunction):
    """
    Gaussian covariance function class

    Attributes
    ----------

    Methods
    -------

    Notes
    -----
    """

    def __init__(self):
        pass
