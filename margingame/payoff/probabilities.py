import numpy as np
from scipy.stats import norm


def gaussian_probabilities(delta, prob_b_without_margin, variance=1):
    """
    :param delta: Radius of uncertainty margin.
    :param prob_b_without_margin: Expected probability of attack being successful (breaching) given that there is no
                                    uncertainty margin. In other words, the mean target of the attack.
    :param variance: Variance in the result of the attack.
    :returns: Numpy array containing the probability of attack failing, being detained and succeeding, respectively.
    """

    midpoint = norm.ppf(1 - prob_b_without_margin, scale=variance)  # Finds the midpoint of the margin.

    prob_q = norm.cdf(midpoint - delta, scale=variance)  # Gives area to the left of the left side of the margin.

    prob_b = 1 - norm.cdf(midpoint + delta, scale=variance)  # Gives area to the right of the right side
    # of the margin.

    prob_d = 1 - prob_b - prob_q  # Gives area in the margin

    return np.array([prob_q, prob_d, prob_b])
