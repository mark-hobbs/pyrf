
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
        self.mu = mu
        self.sigma = sigma

    def build(self, K):
        return self.mu + (K * self.sigma)


class LogNormal(ProbabilityDistribution):

    def __init__(self, m, v):
        self.mu = np.log(m**2 / np.sqrt(v + m**2))
        self.sigma = np.sqrt(np.log((v/m**2) + 1))

    def build(self, K):
        return np.exp(self.mu + (K * self.sigma))


class Weibull(ProbabilityDistribution):
    pass
