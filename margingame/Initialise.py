import nashpy as nash
import dask.array as da
from margingame.payoff.payoffs import payoff_matrix_with_costs

#  https://nashpy.readthedocs.io/en/stable/tutorial/index.html#creating-a-game


class Initialise:
    def __init__(self,
                 no_of_deltas=4,
                 delta_limit=2,
                 no_of_probabilities=4,
                 no_of_variances=3,
                 variance_limit=2,
                 defender_naive_payoffs=(0, 25, -50),
                 adversary_naive_payoffs=(0, -10, 50),
                 delta_parameter=4,
                 prob_b_parameter=3,
                 variance_parameter=3):

        self.attacker_payoff_matrix = payoff_matrix_with_costs(no_of_deltas=no_of_deltas,
                                                               delta_limit=delta_limit,
                                                               no_of_probabilities=no_of_probabilities,
                                                               no_of_variances=no_of_variances,
                                                               variance_limit=variance_limit,
                                                               adversary_naive_payoffs=adversary_naive_payoffs,
                                                               defender_naive_payoffs=defender_naive_payoffs,
                                                               delta_parameter=delta_parameter,
                                                               prob_b_parameter=prob_b_parameter,
                                                               variance_parameter=variance_parameter,
                                                               defender=False)

        self.defender_payoff_matrix = payoff_matrix_with_costs(no_of_deltas=no_of_deltas,
                                                               delta_limit=delta_limit,
                                                               no_of_probabilities=no_of_probabilities,
                                                               no_of_variances=no_of_variances,
                                                               variance_limit=variance_limit,
                                                               adversary_naive_payoffs=adversary_naive_payoffs,
                                                               defender_naive_payoffs=defender_naive_payoffs,
                                                               delta_parameter=delta_parameter,
                                                               prob_b_parameter=prob_b_parameter,
                                                               variance_parameter=variance_parameter,
                                                               defender=True)

        self.A = self.attacker_payoff_matrix.to_numpy()
        self.B = self.defender_payoff_matrix.to_numpy()

        no_of_columns = no_of_probabilities * no_of_variances
        no_of_rows = no_of_deltas
        self.dask_A = da.from_array(self.A, (no_of_rows, no_of_columns))
        self.dask_B = da.from_array(self.B, (no_of_rows, no_of_columns))

    def calculate_equilibria_support_enum(self, A=None, B=None):
        """
        :return: List of equilibria for the game found via support enumeration
        """
        if A is None and B is None:
            A, B = self.dask_A, self.dask_B

        game = nash.Game(A, B)
        equilibria = game.support_enumeration()
        return list(equilibria)
