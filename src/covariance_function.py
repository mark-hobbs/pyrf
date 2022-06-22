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

    def __init__(self, x, lc, rho):
        """
        Initialise an instance of the JCSS covariance function class

        Parameters
        ----------
        x : ndarray
            Mesh
        
        lc : float
            Correlation length (or length scale)

        rho : float
            Threshold value for the correlation between two points in space

        Returns
        -------

        """
        self.x = x
        self.lc = lc * self.mm_to_m
        self.rho = rho

    def build_correlation_matrix(self):

        C = np.zeros([len(self.x), len(self.x)])

        for i in range(len(self.x)):
            for j in range(i + 1):
                C[i, j] = (self.rho + (1 - self.rho)
                           * np.exp(-np.linalg.norm(self.x[i, :] - self.x[j, :])
                                    / self.lc))
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

    def __init__(self):
        pass


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
