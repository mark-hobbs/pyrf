import numpy as np
from scipy import stats


class ProbabilityDistribution:
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

    @staticmethod
    def print_distribution_properties(K):
        # TODO: why does np.mean(K) return a complex number?
        print("{} : {:.2f}".format("Mean", np.mean(np.real(K))))
        print("{} : {:.2f}".format("Standard deviation", np.std(K)))
        print(
            "{} : {:.2f}".format(
                "Coefficient of variation (%)", (np.std(K) / np.mean(np.real(K))) * 100
            )
        )


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
            Mean of a lognormally distributed variable Y

        v : float
            Variance of a lognormally distributed variable Y

        Returns
        -------
        mu : float
            Mean of log(Y)

        sigma : float
            Standard deviation of log(Y)

        """
        self.mu = np.log(m**2 / np.sqrt(v + m**2))
        self.sigma = np.sqrt(np.log((v / m**2) + 1))

    def build(self, K):
        return np.exp(self.mu + (K * self.sigma))


class Weibull(ProbabilityDistribution):

    def __init__(self, mu, sigma, shape):
        """
        Initialise an instance of the Weibull probability distribution class

        Parameters
        ----------
        mu : float
            Mean or expectation of the distribution

        sigma : float
            Standard deviation of the distribution

        shape : float
            The Weibull shape parameter. In the field of materials science,
            the shape parameter is more commonly known as the Weibull modulus.
            If shape = 1, the Weibull distribution reduces to an exponential
            distribution.

        Returns
        -------

        """
        self.mu = mu
        self.sigma = sigma
        self.shape = shape

    def build(self, K):
        """
        Using Copula theory
        """
        rv_norm = stats.norm.cdf(self.mu + (K * self.sigma), self.mu, self.sigma)
        return stats.weibull_min.ppf(rv_norm, self.shape, loc=0, scale=self.mu)
