import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class RandomField():
    """
    Random field class

    Attributes
    ----------

    Methods
    -------

    Notes
    -----
    """

    def __init__(self, covariance_function, probability_distribution):
        pass

    def generate_normally_distributed_variables(self, array):
        """
        Generate a standard Gaussian random vector

        Parameters
        ----------

        Returns
        -------
        xi : ndarray
            Standard Gaussian random vector
        """
        return np.random.uniform(low=0.0, high=1.0, size=[len(array)])

    def generate_correlated_random_variables(self):
        pass

    def visualise(self, x, K):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.scatter(x[:, 0], x[:, 1], s=30, c=K, marker='o', cmap=cm.jet)
        plt.axis('scaled')


class KLexpansion(RandomField):
    """
    Karhunen-Loève expansion class

    Attributes
    ----------

    Methods
    -------

    Notes
    -----
    """
    pass


class MatrixDecomposition(RandomField):
    """
    Matrix-decomposition class

    Attributes
    ----------

    Methods
    -------

    Notes
    -----
    """

    def __init__(self, C, probability_distribution):
        self.C = C
        self.distribution = probability_distribution
        (self.eigenvalues,
         self.eigenvectors) = self.decompose_covariance_matrix()
        self.L = self.compute_lower_triangular_matrix()
        self.xi = self.generate_normally_distributed_variables(self.C)

    def decompose_covariance_matrix(self):
        """
        Decompose the covariance matrix into its eigenvalues and eigenvectors
        """
        eigenvalues, eigenvectors = np.linalg.eig(self.C)
        return eigenvalues, eigenvectors

    def compute_lower_triangular_matrix(self):
        """
        Compute the lower triangular matrix

        Parameters
        ----------

        Returns
        -------
        L : ndarray
            Lower triangular matrix
        """
        return np.sqrt(np.absolute(self.eigenvalues)) * self.eigenvectors

    def generate_sample(self):
        """
        Generate a single sample of the random field

        Parameters
        ----------

        Returns
        -------
        K : ndarray
            Sample of the random field
        """
        return np.matmul(self.L, self.xi)

    def build_distribution(self):
        """
        TODO: write description

        Parameters
        ----------

        Returns
        -------

        TODO: rename

        """
        return self.distribution.build(self.generate_sample())
