""" Determine the uncertainties for the cut-in scenario category.

Creation date: 2022 07 01
Author(s): Erwin de Gelder
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from domain_model import DocumentManagement
from simulation import SimulationCutIn, ACC, acc_cutin_pars, kde_from_file
from uncertainty import FOLDER, Uncertainty


def cutin_valid(parameters):
    """ Determine whether the cut-in parameters are valid.

    All 3 parameters must be positive.

    :param parameters: numpy array of the parameters.
    :return: vector if logical values (True if valid, False if invalid).
    """
    return np.all(parameters > 0, axis=1)


CUTINS = DocumentManagement(os.path.join("data", "scenarios", "cut-in.json"))
KDE_CUTIN = kde_from_file(os.path.join("data", "kde", "cut-in.p"))
SIMULATOR_CUTIN = SimulationCutIn(ACC(), acc_cutin_pars)
UNCERTAINTY_CUTIN = Uncertainty("cutin", CUTINS, KDE_CUTIN, SIMULATOR_CUTIN,
                                ["dinit", "vlead", "vego"], cutin_valid)


if __name__ == "__main__":
    if not os.path.exists(FOLDER):
        os.mkdir(FOLDER)

    UNCERTAINTY_CUTIN.exposure_category()
    plt.title("Cut-in")

    UNCERTAINTY_CUTIN.bootstrap_is_result2()
    plt.title("Cut-in")

    UNCERTAINTY_CUTIN.vary_simulations()
    plt.title("Cut-in")

    UNCERTAINTY_CUTIN.total_variance_hours()
    plt.title("Cut-in")

    plt.show()
