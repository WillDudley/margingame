import numpy as np


def cost_of_increasing_margin(delta, scaling_parameter=1):
    """
    :param delta: Radius of uncertainty margin.
    :param scaling_parameter: Adjusts the cost.
    :return: Cost of setting 'delta'.
    """
    return scaling_parameter * (np.exp(delta))


def cost_of_increasing_success_rate(prob_b, scaling_parameter=1):
    """
    :param prob_b: Expected probability of attack being successful (breaching) given that there is no uncertainty
                        margin. In other words, the mean target of the attack.
    :param scaling_parameter: Adjusts the cost.
    :return: Cost of setting 'prob_b'.
    """
    return scaling_parameter * np.tan((np.pi / 2) * prob_b)


def cost_of_reducing_variance(variance=1, scaling_parameter=0.2):
    """
    :param variance: Variance in the result of the attack.
    :param scaling_parameter: Adjusts the cost.
    :return: Cost of setting 'variance'.
    """
    return scaling_parameter / variance
