import nashpy as nash
import dask.array as da
from margingame.payoff.payoffs import payoff_matrix_with_costs

#  https://nashpy.readthedocs.io/en/stable/tutorial/index.html#creating-a-game


class Initialise:
    def __init__(self,
                 no_of_deltas=10,
                 delta_limit=2,
                 no_of_probabilities=4,
                 no_of_std_devs=4,
                 std_dev_limit=2,
                 defender_naive_payoffs=(0, 50, -150),
                 adversary_naive_payoffs=(0, -150, 50),
                 delta_parameter=5,
                 prob_b_parameter=5,
                 std_dev_parameter=1):

        self.attacker_payoff_matrix = payoff_matrix_with_costs(no_of_deltas=no_of_deltas,
                                                               delta_limit=delta_limit,
                                                               no_of_probabilities=no_of_probabilities,
                                                               no_of_std_devs=no_of_std_devs,
                                                               std_dev_limit=std_dev_limit,
                                                               adversary_naive_payoffs=adversary_naive_payoffs,
                                                               defender_naive_payoffs=defender_naive_payoffs,
                                                               delta_parameter=delta_parameter,
                                                               prob_b_parameter=prob_b_parameter,
                                                               std_dev_parameter=std_dev_parameter,
                                                               defender=False)

        self.defender_payoff_matrix = payoff_matrix_with_costs(no_of_deltas=no_of_deltas,
                                                               delta_limit=delta_limit,
                                                               no_of_probabilities=no_of_probabilities,
                                                               no_of_std_devs=no_of_std_devs,
                                                               std_dev_limit=std_dev_limit,
                                                               adversary_naive_payoffs=adversary_naive_payoffs,
                                                               defender_naive_payoffs=defender_naive_payoffs,
                                                               delta_parameter=delta_parameter,
                                                               prob_b_parameter=prob_b_parameter,
                                                               std_dev_parameter=std_dev_parameter,
                                                               defender=True)

        self.A = self.defender_payoff_matrix.to_numpy()
        self.B = self.attacker_payoff_matrix.to_numpy()

        no_of_columns = no_of_probabilities * no_of_std_devs
        no_of_rows = no_of_deltas
        self.dask_A = da.from_array(self.A, (no_of_rows, no_of_columns))
        self.dask_B = da.from_array(self.B, (no_of_rows, no_of_columns))
        self.game = nash.Game(self.dask_A, self.dask_B)

    def calculate_equilibria_support_enum(self, A=None, B=None):
        """
        :return: List of equilibria for the game found via support enumeration
        """
        if A is None and B is None:
            A, B = self.dask_A, self.dask_B

        game = nash.Game(A, B)
        equilibria = game.support_enumeration()
        return list(equilibria)

    def calculate_equilibria_vertex_enum(self, A=None, B=None):
        """
        :return: List of equilibria for the game found via support enumeration
        """
        if A is None and B is None:
            A, B = self.dask_A, self.dask_B

        game = nash.Game(A, B)
        equilibria = game.vertex_enumeration()
        return list(equilibria)
