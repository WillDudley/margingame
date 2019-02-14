import numpy as np
from scipy.stats import norm


def gaussian_probabilities(delta, prob_b, variance=1):
    """
    :param delta: Radius of uncertainty margin.
    :param prob_b: Expected probability of attack being successful (breaching) given that there is no uncertainty
                        margin. In other words, the mean target of the attack.
    :param variance: Variance in the result of the attack.
    :returns: Numpy array containing the probability of attack failing, being detained and succeeding, respectively.
    """
    d = norm.ppf(1 - prob_b, scale=variance)  # Finds where the margin is around

    prob_q = norm.cdf(-delta, loc=-d, scale=variance)  # Gives area to left of left side of margin
    prob_less_than_delta = norm.cdf(delta, loc=-d, scale=variance)
    prob_b = 1 - prob_less_than_delta  # Gives area to right side of margin
    prob_d = prob_less_than_delta - prob_q  # Gives area in margin

    return np.array([prob_q, prob_d, prob_b])
