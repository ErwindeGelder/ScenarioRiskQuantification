""" Simulation of the scenario "approaching slower vehicle".

Creation date: 2020 08 13
Author(s): Erwin de Gelder

Modifications:
2020 12 09: Adding idm_approaching_pars.
2020 12 10: Make SimulationApproaching a subclass of SimulationString.
2020 12 11: Adding acc_idm_approaching_pars.
"""

from typing import List
import numpy as np
from .acc import ACCParameters
from .acc_idmplus import ACCIDMPlusParameters
from .idm import IDMParameters
from .idmplus import IDMPlus
from .leader_braking import LeaderBraking, LeaderBrakingParameters
from .simulation_string import SimulationString


INIT_POSITION_FOLLOWER = 0


def acc_approaching_pars(**kwargs):
    """ Define the ACC parameters in an approaching scenario.

    :return: Parameter object that can be passed via init_simulation.
    """
    parms = dict()
    for parm in ["amin", "sensor_range", "k1_acc", "k2_acc", "k_cruise"]:
        if parm in kwargs:
            parms[parm] = kwargs[parm]
    return ACCParameters(speed=kwargs["vego"],
                         init_speed=kwargs["vego"],
                         init_position=INIT_POSITION_FOLLOWER,
                         n_reaction=0,
                         cruise_after_collision=True,
                         **parms)


def acc_idm_approaching_pars(**kwargs):
    """ Define the parameters for the ACCIDMPlus model.

    :return: Parameter object that can be passed via init_simulation.
    """
    if "amin" in kwargs:
        amin = kwargs["amin"]
    else:
        amin = -10
        kwargs["amin"] = amin
    parms = dict()
    for parm in ["k1_acc", "k2_acc"]:
        if parm in kwargs:
            parms[parm] = kwargs[parm]
    parms["driver_takeover_view"] = kwargs["max_view"] if "max_view" in kwargs else 150
    if "reactiontime" not in kwargs:
        kwargs["reactiontime"] = np.random.lognormal(np.log(.92**2/np.sqrt(.92**2+.28**2)),
                                                     np.sqrt(np.log(1+.28**2/.92**2)))
    fcw_delay = kwargs["reactiontime"]
    return ACCIDMPlusParameters(speed=kwargs["vego"],
                                init_speed=kwargs["vego"],
                                init_position=INIT_POSITION_FOLLOWER,
                                n_reaction=0,
                                amin=amin,
                                driver_parms=idm_approaching_pars(**kwargs),
                                driver_model=IDMPlus(),
                                fcw_delay=fcw_delay,
                                **parms)


def idm_approaching_pars(i=1, **kwargs):
    """ Define the parameters for the IDM model.

    The reaction time is sampled from the lognormal distribution mentioned in
    Wang & Stamatiadis (2014) if it not provided through kwargs.

    :param i: Optional parameter telling which vehicle it is in the string
        (start counting from 0).
    :param kwargs: Parameter object that can be passed via init_simulation.
    """
    if "reactiontime" in kwargs:
        reactiontime = kwargs["reactiontime"]
    else:
        reactiontime = np.random.lognormal(np.log(.92**2/np.sqrt(.92**2+.28**2)),
                                           np.sqrt(np.log(1+.28**2/.92**2)))
    parms = dict()
    for parm in ["amin", "max_view"]:
        if parm in kwargs:
            parms[parm] = kwargs[parm]
    steptime = 0.01
    thw = kwargs["thw"] if "thw" in kwargs else 1.1
    safety_distance = 2
    init_position = INIT_POSITION_FOLLOWER - (i-1)*(thw*kwargs["vego"]+safety_distance)
    return IDMParameters(speed=kwargs["vego"],
                         init_speed=kwargs["vego"],
                         init_position=init_position,
                         timestep=steptime,
                         n_reaction=int(reactiontime/steptime),
                         thw=thw,
                         safety_distance=safety_distance,
                         **parms)


class SimulationApproaching(SimulationString):
    """ Class for simulation the scenario "approaching slower vehicle". """
    def __init__(self, followers: List, followers_parameters: List, **kwargs):
        SimulationString.__init__(self, [LeaderBraking()] + followers,
                                  [self._leader_parameters] + followers_parameters,
                                  **kwargs)

    @staticmethod
    def _leader_parameters(**kwargs):
        """ Return the paramters for the leading vehicle. """
        if "init_position" not in kwargs:
            init_position = kwargs["vego"] * 4  # At least at THW of 4 s. Therefore, also TTC>4s.
        else:
            init_position = kwargs["init_position"]
        return LeaderBrakingParameters(init_position=init_position,
                                       init_speed=kwargs["vego"]*kwargs["ratio_vtar_vego"],
                                       average_deceleration=1,
                                       speed_difference=0,
                                       tconst=5)
