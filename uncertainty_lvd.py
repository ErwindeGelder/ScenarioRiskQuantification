""" Determine the uncertainties for the LVD scenario category.

Creation date: 2022 07 01
Author(s): Erwin de Gelder
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from domain_model import DocumentManagement
from simulation import SimulationLeadBraking, acc_lead_braking_pars, ACC, kde_from_file
from uncertainty import Uncertainty


def lvd_valid(parameters: np.ndarray) -> np.ndarray:
    """ Determine whether the LVD parameters are valid.

    All 3 parameters must be positive.
    The 3rd parameter must not be larger than the first parameters.

    :param parameters: numpy array of the parameters.
    :return: vector if logical values (True if valid, False if invalid).
    """
    return np.logical_and(np.all(parameters > 0, axis=1), parameters[:, 2] <= parameters[:, 0])


LEADDEC = DocumentManagement(os.path.join("data", "scenarios", "leading_vehicle_braking.json"))
KDE_LVD = kde_from_file(os.path.join("data", "kde", "leading_vehicle_decelerating.p"))
SIMULATOR_LVD = SimulationLeadBraking(ACC(), acc_lead_braking_pars)
UNCERTAINTY_LVD = Uncertainty("lvd", LEADDEC, KDE_LVD, SIMULATOR_LVD, ["v0", "amean", "dv"],
                              lvd_valid)


if __name__ == "__main__":
    UNCERTAINTY_LVD.exposure_category()
    plt.title("Leading vehicle decelerating")

    UNCERTAINTY_LVD.bootstrap_is_result2()
    plt.title("Leading vehicle decelerating")

    UNCERTAINTY_LVD.vary_simulations()
    plt.title("Leading vehicle decelerating")

    UNCERTAINTY_LVD.total_variance_hours()
    plt.title("Leading vehicle decelerating")

    plt.show()
