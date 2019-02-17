import numpy as np
import pandas as pd
import margingame.payoff.costs as costs
from margingame.payoff.probabilities import gaussian_probabilities


def expected_payoff_with_costs(delta,
                               prob_b,
                               std_dev,
                               defender_naive_payoffs=(0, 25, -50),
                               adversary_naive_payoffs=(0, -10, 50),
                               delta_parameter=4,
                               prob_b_parameter=3,
                               std_dev_parameter=3,
                               defender=True):
    """
    :param delta: Radius of uncertainty margin.
    :param prob_b: Expected probability of attack being successful (breaching) given that there is no uncertainty
                        margin. In other words, the mean target of the attack.
    :param std_dev: Standard deviation in the result of the attack.
    :param defender_naive_payoffs: The defender's respective payoff of an attack failing, being detained and succeeding.
    :param adversary_naive_payoffs: The attacker's respective payoff of an attack failing, being detained and
                        succeeding.
    :param delta_parameter: Adjusts the cost of adjusting delta.
    :param prob_b_parameter: Adjusts the cost of adjusting prob_b.
    :param std_dev_parameter: Adjusts the cost of adjusting standard deviation.
    :param defender: If True, calculates defender's expected payoff. If False, calculates attacker's expected payoff.
    :return: Calculates a chosen player's expected payoff given the input parameters.
    """

    probabilities = gaussian_probabilities(delta, prob_b, std_dev)

    if defender:
        return np.dot(np.array(defender_naive_payoffs), probabilities) \
               - costs.cost_of_increasing_margin(delta, delta_parameter)
    else:
        return np.dot(np.array(adversary_naive_payoffs), probabilities) \
               - costs.cost_of_increasing_success_rate(prob_b, prob_b_parameter) \
               - costs.cost_of_reducing_std_dev(std_dev, std_dev_parameter)


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
                             no_of_std_devs=3,
                             std_dev_limit=2,
                             defender_naive_payoffs=(0, 25, -50),
                             adversary_naive_payoffs=(0, -10, 50),
                             delta_parameter=4,
                             prob_b_parameter=3,
                             std_dev_parameter=3,
                             defender=True):
    """
    :param no_of_deltas: Number of evenly spaced deltas (size of uncertainty margin) to be investigated in the
                         range (0, 'delta_limit').
    :param delta_limit: The exclusive endpoint that the deltas go up to.
    :param no_of_probabilities: Number of evenly spaced probabilities (attacker targets) to be investigated in the
                                range (0, 1).
    :param no_of_std_devs: Number of evenly spaced attacker standard deviations to be investigated in the
                            range (0, 'std_dev_limit').
    :param std_dev_limit: The exclusive endpoint that the standard deviations go up to.
    :param defender_naive_payoffs: The defender's respective payoff of an attack failing, being detained and succeeding.
    :param adversary_naive_payoffs: The attacker's respective payoff of an attack failing, being detained and
                        succeeding.
    :param delta_parameter: Adjusts the cost of adjusting delta.
    :param prob_b_parameter: Adjusts the cost of adjusting prob_b.
    :param std_dev_parameter: Adjusts the cost of adjusting standard deviation.
    :param defender: If True, calculates defender's expected payoff matrix. If False, calculates attacker's expected
                     payoff matrix.
    :return: Calculates a chosen player's expected payoff matrix given the input parameters.
    """
    no_of_columns = no_of_probabilities * no_of_std_devs
    no_of_rows = no_of_deltas

    list_of_probabilities = [x.round(1) for x in open_linspace(0, 1, no_of_probabilities)]
    list_of_std_devs = [x.round(1) for x in open_linspace(0, std_dev_limit, no_of_std_devs)]
    columns = pd.MultiIndex.from_product([list_of_probabilities, list_of_std_devs], names=['Target', 'Standard Deviation'])

    list_of_deltas = [x.round(1) for x in open_linspace(0, delta_limit, no_of_deltas)]

    payoff_matrix = pd.DataFrame(np.zeros((no_of_rows, no_of_columns)), index=list_of_deltas, columns=columns)

    for row in range(0, no_of_rows):
        for column in range(0, no_of_columns):
            payoff_matrix.iloc[row, column] = expected_payoff_with_costs(list_of_deltas[row],
                                                                         columns[column][0],
                                                                         std_dev=columns[column][1],
                                                                         adversary_naive_payoffs=adversary_naive_payoffs,
                                                                         defender_naive_payoffs=defender_naive_payoffs,
                                                                         delta_parameter=delta_parameter,
                                                                         prob_b_parameter=prob_b_parameter,
                                                                         std_dev_parameter=std_dev_parameter,
                                                                         defender=defender)

    return payoff_matrix

