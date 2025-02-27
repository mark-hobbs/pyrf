import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, kv
from scipy.spatial import distance_matrix


class CovarianceFunction:
    """
    Base class for covariance functions.

    Attributes
    ----------
    mm_to_m : float
        Conversion factor from millimetres to metres.

    Methods
    -------
    visualise(C)
        Displays the covariance matrix as a heatmap.
    """

    def __init__(self):
        self.mm_to_m = 1e-3

    def visualise(self, C):
        """
        Visualise the covariance matrix as an image
        """
        plt.imshow(C, cmap="viridis")

    def build_correlation_matrix(self, x):
        """
        Construct the correlation matrix
        """
        raise NotImplementedError("Subclasses must implement this method")


class JCSS(CovarianceFunction):
    """
    JCSS (Joint Committee on Structural Safety) covariance function.

    Parameters
    ----------
    lc : float
        Correlation length (or length scale) in millimetres.
    rho : float
        Threshold value for correlation between two points.

    Methods
    -------
    build_correlation_matrix(x)
        Constructs the correlation matrix for input coordinates.
    """

    def __init__(self, lc, rho):
        super().__init__()
        self.lc = lc * self.mm_to_m
        self.rho = rho

    def build_correlation_matrix(self, x):
        """
        Construct the correlation matrix using the JCSS model
        """
        d = distance_matrix(x, x)
        C = self.rho + (1 - self.rho) * np.exp(-d / self.lc)
        return C


class Exponential(CovarianceFunction):
    """
    Exponential covariance function.

    Parameters
    ----------
    lc : float
        Correlation length (or length scale) in millimetres.
    sigma : float, optional (default=1)
        Marginal standard deviation.

    Methods
    -------
    build_correlation_matrix(x)
        Constructs the correlation matrix for input coordinates.
    """

    def __init__(self, lc, sigma=1):
        super().__init__()
        self.lc = lc * self.mm_to_m
        self.sigma = sigma

    def build_correlation_matrix(self, x):
        """
        Construct the correlation matrix using the exponential model
        """
        d = distance_matrix(x, x)
        C = self.sigma * np.exp(-d / self.lc)
        return C


class Gaussian(CovarianceFunction):
    """
    Gaussian covariance function.

    Parameters
    ----------
    lc : float
        Correlation length (or length scale) in millimetres.
    sigma : float, optional (default=1)
        Marginal standard deviation.

    Methods
    -------
    build_correlation_matrix(x)
        Constructs the correlation matrix for input coordinates.
    """

    def __init__(self, lc, sigma=1):
        super().__init__()
        self.lc = lc * self.mm_to_m
        self.sigma = sigma

    def build_correlation_matrix(self, x):
        """
        Construct the correlation matrix using the Gaussian model
        """
        d = distance_matrix(x, x)
        C = self.sigma**2 * np.exp(-((d / self.lc) ** 2))
        return C


class Matern(CovarianceFunction):
    """
    Matérn covariance function.

    Parameters
    ----------
    lc : float
        Correlation length (or length scale) in millimetres.
    sigma : float, optional (default=1)
        Marginal standard deviation.
    nu : float, optional (default=1/2)
        Shape parameter (often 1/2, 3/2, or 5/2).

    Methods
    -------
    build_correlation_matrix(x)
        Constructs the correlation matrix for input coordinates.
    """

    def __init__(self, lc, sigma=1, nu=1 / 2):
        super().__init__()
        self.lc = lc * self.mm_to_m
        self.sigma = sigma
        self.nu = nu

    def build_correlation_matrix(self, x):
        """
        Construct the correlation matrix using the Matérn model
        """
        d = distance_matrix(x, x)
        C = np.zeros_like(d)

        np.fill_diagonal(C, self.sigma**2)

        mask = d > 0
        alpha = np.sqrt(2 * self.nu) * (d[mask] / self.lc)
        C[mask] = (
            self.sigma**2
            * (2 ** (1 - self.nu) / gamma(self.nu))
            * alpha**self.nu
            * kv(self.nu, alpha)
        )

        return C
