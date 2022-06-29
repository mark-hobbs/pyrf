
import numpy as np


class ProbabilityDistribution():
    """
    Probability distribution class

    Attributes
    ----------

    Methods
    -------

    Notes
    -----
    """

    def __init__(self):
        pass


class Gaussian(ProbabilityDistribution):

    def __init__(self, mu, sigma):
        """
        Initialise an instance of the Gaussian probability distribution class

        Parameters
        ----------
        mu : float
            Mean or expectation of the distribution

        sigma : float
            Standard deviation of the distribution

        Returns
        -------

        """
        self.mu = mu
        self.sigma = sigma

    def build(self, K):
        return self.mu + (K * self.sigma)


class LogNormal(ProbabilityDistribution):

    def __init__(self, m, v):
        """
        Initialise an instance of the log-normal probability distribution
        class

        Parameters
        ----------
        m : float

        v : float

        Returns
        -------

        """
        self.mu = np.log(m**2 / np.sqrt(v + m**2))
        self.sigma = np.sqrt(np.log((v/m**2) + 1))

    def build(self, K):
        return np.exp(self.mu + (K * self.sigma))


class Weibull(ProbabilityDistribution):
    pass
