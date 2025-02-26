import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class RandomField:
    """
    Base class for generating random fields.

    Parameters
    ----------
    covariance_function : callable
        Function defining the covariance structure.

    probability_distribution : callable
        Function defining the probability distribution of the random field.

    Methods
    -------
    generate_normally_distributed_variables(size)
        Generates a standard Gaussian random vector.

    visualise(x, K, sz=10)
        Visualises the generated random field.
    """

    def __init__(self, covariance_function, probability_distribution):
        self.covariance_function = covariance_function
        self.probability_distribution = probability_distribution

    @staticmethod
    def generate_normally_distributed_variables(size):
        """
        Generate a standard Gaussian random vector
        """
        return np.random.randn(size)

    def visualise(self, x, K, sz=10):
        """
        Visualise the generated random field
        """
        _, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(x[:, 0], x[:, 1], s=sz, c=K, cmap=cm.jet)
        ax.axis("off")
        ax.set_frame_on(False)
        plt.tight_layout()
        plt.axis("scaled")


class KLExpansion(RandomField):
    """
    Karhunen-Loève expansion class for generating random fields
    """

    pass


class MatrixDecomposition(RandomField):
    """
    Matrix decomposition method for generating random fields.

    Parameters
    ----------
    C : ndarray
        Covariance matrix.

    probability_distribution : callable
        Function defining the probability distribution of the random field.

    Methods
    -------
    decompose_covariance_matrix()
        Computes eigenvalues and eigenvectors of the covariance matrix.

    compute_lower_triangular_matrix()
        Computes the lower triangular matrix for sampling.

    generate_sample_normal()
        Generates a standard normal random field sample.

    generate_sample()
        Generates a sample with the user-defined probability distribution.

    generate_samples(n_samples)
        Generates multiple samples of the random field.
    """

    def __init__(self, C, probability_distribution):
        self.C = C
        self.probability_distribution = probability_distribution
        self.eigenvalues, self.eigenvectors = self.decompose_covariance_matrix()
        self.L = self.compute_lower_triangular_matrix()

    def decompose_covariance_matrix(self):
        """
        Compute eigenvalues and eigenvectors of the covariance matrix,
        sorted in descending order
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self.C)
        idx = np.argsort(eigenvalues)[::-1]
        return eigenvalues[idx], eigenvectors[:, idx]

    def compute_lower_triangular_matrix(self):
        """
        Compute the lower triangular matrix for sampling
        """
        return self.eigenvectors @ np.diag(np.sqrt(np.abs(self.eigenvalues)))

    def generate_sample_normal(self):
        """
        Generate a sample of the random field with a standard normal distribution
        """
        xi = self.generate_normally_distributed_variables(len(self.eigenvalues))
        return self.L @ xi

    def generate_sample(self):
        """
        Generate a sample of the random field with the user-defined probability distribution
        """
        return self.probability_distribution.build(self.generate_sample_normal())

    def generate_samples(self, n_samples):
        """
        Generate multiple samples of the random field
        """
        return np.array([self.generate_sample() for _ in range(n_samples)]).T
