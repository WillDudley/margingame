import numpy as np
import pandas as pd
import functions.costs as costs
from functions.probabilities import gaussian_probabilities


def expected_payoff_with_costs(delta,
                               prob_b,
                               variance,
                               defender_naive_payoffs=(0, 25, -50),
                               adversary_naive_payoffs=(0, -10, 50),
                               delta_parameter=4,
                               prob_b_parameter=3,
                               variance_parameter=3,
                               defender=True):
    """
    :param delta: Radius of uncertainty margin.
    :param prob_b: Expected probability of attack being successful (breaching) given that there is no uncertainty
                        margin. In other words, the mean target of the attack.
    :param variance: Variance in the result of the attack.
    :param defender_naive_payoffs: The defender's respective payoff of an attack failing, being detained and succeeding.
    :param adversary_naive_payoffs: The attacker's respective payoff of an attack failing, being detained and
                        succeeding.
    :param delta_parameter: Adjusts the cost of adjusting delta.
    :param prob_b_parameter: Adjusts the cost of adjusting prob_b.
    :param variance_parameter: Adjusts the cost of adjusting variance.
    :param defender: If True, calculates defender's expected payoff. If False, calculates attacker's expected payoff.
    :return: Calculates a chosen player's expected payoff given the input parameters.
    """

    probabilities = gaussian_probabilities(delta, prob_b, variance)

    if defender:
        return np.dot(np.array(defender_naive_payoffs), probabilities) \
               - costs.cost_of_increasing_margin(delta, delta_parameter)
    else:
        return np.dot(np.array(adversary_naive_payoffs), probabilities) \
               - costs.cost_of_increasing_success_rate(prob_b, prob_b_parameter) \
               - costs.cost_of_reducing_variance(variance, variance_parameter)


def open_linspace(start, stop, step):
    """
    :param start: Start point.
    :param stop: End point.
    :param step: Number of elements desired.
    :return: List of 'step' evenly-spaced numbers between 'start' and 'end', excluding 'start' and 'end'.
    """
    return np.linspace(start, stop, step+2)[start+1:step+1]


def payoff_matrix_with_costs(no_of_deltas=4,
                             delta_limit=2,
                             no_of_probabilities=4,
                             no_of_variances=3,
                             variance_limit=2,
                             defender_naive_payoffs=(0, 25, -50),
                             adversary_naive_payoffs=(0, -10, 50),
                             delta_parameter=4,
                             prob_b_parameter=3,
                             variance_parameter=3,
                             defender=True):
    """
    :param no_of_deltas: Number of evenly spaced deltas (size of uncertainty margin) to be investigated in the
                         range (0, 'delta_limit').
    :param delta_limit: The exclusive endpoint that the deltas go up to.
    :param no_of_probabilities: Number of evenly spaced probabilities (attacker targets) to be investigated in the
                                range (0, 1).
    :param no_of_variances: Number of evenly spaced attacker variances to be investigated in the
                            range (0, 'variance_limit').
    :param variance_limit: The exclusive endpoint that the variances go up to.
    :param defender_naive_payoffs: The defender's respective payoff of an attack failing, being detained and succeeding.
    :param adversary_naive_payoffs: The attacker's respective payoff of an attack failing, being detained and
                        succeeding.
    :param delta_parameter: Adjusts the cost of adjusting delta.
    :param prob_b_parameter: Adjusts the cost of adjusting prob_b.
    :param variance_parameter: Adjusts the cost of adjusting variance.
    :param defender: If True, calculates defender's expected payoff matrix. If False, calculates attacker's expected
                     payoff matrix.
    :return: Calculates a chosen player's expected payoff matrix given the input parameters.
    """
    no_of_columns = no_of_probabilities * no_of_variances
    no_of_rows = no_of_deltas

    list_of_probabilities = [x.round(1) for x in open_linspace(0, 1, no_of_probabilities)]
    list_of_variances = [x.round(1) for x in open_linspace(0, variance_limit, no_of_variances)]
    columns = pd.MultiIndex.from_product([list_of_probabilities, list_of_variances], names=['Target', 'Variance'])

    list_of_deltas = [x.round(1) for x in open_linspace(0, delta_limit, no_of_deltas)]

    payoff_matrix = pd.DataFrame(np.zeros((no_of_rows, no_of_columns)), index=list_of_deltas, columns=columns)

    for row in range(0, no_of_rows):
        for column in range(0, no_of_columns):
            payoff_matrix.iloc[row, column] = expected_payoff_with_costs(list_of_deltas[row],
                                                                         columns[column][0],
                                                                         variance=columns[column][1],
                                                                         adversary_naive_payoffs=adversary_naive_payoffs,
                                                                         defender_naive_payoffs=defender_naive_payoffs,
                                                                         delta_parameter=delta_parameter,
                                                                         prob_b_parameter=prob_b_parameter,
                                                                         variance_parameter=variance_parameter,
                                                                         defender=defender)

    return payoff_matrix

