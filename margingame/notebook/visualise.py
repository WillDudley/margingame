import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets as widgets
from margingame.Initialise import Initialise
from ipywidgets import fixed


def plot_heatmaps(d1, d2, d3,
                  a1, a2, a3,
                  delta_parameter, prob_b_parameter, variance_parameter,
                  no_of_deltas=4,
                  delta_limit=2,
                  no_of_probabilities=4,
                  no_of_variances=3,
                  variance_limit=2):
    Game = Initialise(no_of_deltas=no_of_deltas,
                      delta_limit=delta_limit,
                      no_of_probabilities=no_of_probabilities,
                      no_of_variances=no_of_variances,
                      variance_limit=variance_limit,
                      defender_naive_payoffs=(d1, d2, d3),
                      adversary_naive_payoffs=(a1, a2, a3),
                      delta_parameter=delta_parameter,
                      prob_b_parameter=prob_b_parameter,
                      variance_parameter=variance_parameter)

    f, (ax1, ax2) = plt.subplots(2, figsize=(32, 12))
    f.figsize = (6, 32)

    sns.heatmap(Game.attacker_payoff_matrix, ax=ax1, annot=True, linewidths=0.3)
    sns.heatmap(Game.defender_payoff_matrix, ax=ax2, annot=True, linewidths=0.3)

    ax1.set_title("Attacker Payoff Matrix")
    ax2.set_title("Defender Payoff Matrix")

    ax1.set_ylabel("Uncertainty margin width, delta")
    ax2.set_ylabel("Uncertainty margin width, delta")

    ax1.set_xlabel("Attacker's confidence, prob_b-variance")
    ax2.set_xlabel("Attacker's confidence, prob_b-variance")


def visualise(no_of_deltas=4,
              delta_limit=2,
              no_of_probabilities=4,
              no_of_variances=3,
              variance_limit=2):
    style = {'description_width': 'initial'}

    # Attacker naive payoffs
    a1 = widgets.IntSlider(value=0,
                           min=-50,
                           max=50,
                           step=5,
                           description='Quench:',
                           continuous_update=False,
                           style=style)
    a2 = widgets.IntSlider(value=-5,
                           min=-50,
                           max=50,
                           step=5,
                           description='Detain:',
                           continuous_update=False,
                           style=style)
    a3 = widgets.IntSlider(value=25,
                           min=-50,
                           max=50,
                           step=5,
                           description='Breach:',
                           continuous_update=False,
                           style=style)

    # Defender naive payoffs
    d1 = widgets.IntSlider(value=5,
                           min=-50,
                           max=50,
                           step=5,
                           description='Quench:',
                           continuous_update=False,
                           style=style)
    d2 = widgets.IntSlider(value=10,
                           min=-50,
                           max=50,
                           step=5,
                           description='Detain:',
                           continuous_update=False,
                           style=style)
    d3 = widgets.IntSlider(value=-50,
                           min=-50,
                           max=50,
                           step=5,
                           description='Breach:',
                           continuous_update=False,
                           style=style)

    # Cost parameters
    delta_parameter = widgets.IntSlider(value=7,
                                        min=4,
                                        max=10,
                                        step=1,
                                        description='Delta parameter:',
                                        continuous_update=False,
                                        style=style)
    prob_b_parameter = widgets.IntSlider(value=7,
                                         min=4,
                                         max=10,
                                         step=1,
                                         description='Prob_b parameter:',
                                         continuous_update=False,
                                         style=style)
    variance_parameter = widgets.IntSlider(value=7,
                                           min=4,
                                           max=10,
                                           step=1,
                                           description='Variance parameter:',
                                           continuous_update=False,
                                           style=style)

    tab = widgets.Tab(children=[widgets.HBox([delta_parameter, prob_b_parameter, variance_parameter]),
                                widgets.HBox([a1, a2, a3]),
                                widgets.HBox([d1, d2, d3])])

    tab.set_title(0, 'Cost Parameters')
    tab.set_title(1, 'Attacker Payoffs')
    tab.set_title(2, 'Defender Payoffs')

    out = widgets.interactive_output(plot_heatmaps,
                                     {'d1': d1,
                                      'd2': d2,
                                      'd3': d3,
                                      'a1': a1,
                                      'a2': a2,
                                      'a3': a3,
                                      'delta_parameter': delta_parameter,
                                      'prob_b_parameter': prob_b_parameter,
                                      'variance_parameter': variance_parameter,
                                      'no_of_deltas': fixed(no_of_deltas),
                                      'delta_limit': fixed(delta_limit),
                                      'no_of_probabilities': fixed(no_of_probabilities),
                                      'no_of_variances': fixed(no_of_variances),
                                      'variance_limit': fixed(variance_limit)})

    return widgets.VBox([tab, out])