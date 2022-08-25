""" Determine the uncertainties for the ASV scenario category.

Creation date: 2022 07 01
Author(s): Erwin de Gelder
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from domain_model import DocumentManagement
from simulation import SimulationApproaching, ACC, acc_approaching_pars, kde_from_file
from uncertainty import FOLDER, Uncertainty


def asv_valid(parameters):
    """ Determine whether the cut-in parameters are valid.

    All 2 parameters must be positive.
    Second parameter must be smaller than 1.

    :param parameters: numpy array of the parameters.
    :return: vector if logical values (True if valid, False if invalid).
    """
    return np.logical_and(np.all(parameters > 0, axis=1), parameters[:, 1] < 1)


APPROACHING = DocumentManagement(os.path.join("data", "scenarios", "approaching_vehicle.json"))
KDE_ASV = kde_from_file(os.path.join("data", "kde", "approaching_slower_vehicle.p"))
SIMULATOR_ASV = SimulationApproaching([ACC()], [acc_approaching_pars])
UNCERTAINTY_ASV = Uncertainty("asv", APPROACHING, KDE_ASV, SIMULATOR_ASV,
                              ["vego", "ratio_vtar_vego"], asv_valid)


if __name__ == "__main__":
    if not os.path.exists(FOLDER):
        os.mkdir(FOLDER)

    UNCERTAINTY_ASV.exposure_category()
    plt.title("Approaching slower vehicle")

    UNCERTAINTY_ASV.bootstrap_is_result2()
    plt.title("Approaching slower vehicle")

    UNCERTAINTY_ASV.vary_simulations()
    plt.title("Approaching slower vehicle")

    UNCERTAINTY_ASV.total_variance_hours()
    plt.title("Approaching slower vehicle")

    plt.show()
