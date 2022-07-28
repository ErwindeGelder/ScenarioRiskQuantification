""" Simulation of the scenario "approaching cut-in vehicle".

Creation date: 2020 08 13
Author(s): Erwin de Gelder

Modifications:
"""

import numpy as np
from .acc import ACCParameters
from .acc_idmplus import ACCIDMPlusParameters
from .idm import IDMParameters
from .idmplus import IDMPlus
from .leader_braking import LeaderBraking, LeaderBrakingParameters
from .simulation_string import SimulationString


def idm_cutin_pars(**kwargs):
    """ Define the parameters for the IDM model in a cut-in scenario.

    The reaction time is sampled from the lognormal distribution mentioned in
    Wang & Stamatiadis (2014) if it not provided through kwargs.

    :param kwargs: Parameter object that can be passed via init_simulation.
    """
    if "reactiontime" in kwargs:
        reactiontime = kwargs["reactiontime"]
    else:
        reactiontime = np.random.lognormal(np.log(.92**2/np.sqrt(.92**2+.28**2)),
                                           np.sqrt(np.log(1+.28**2/.92**2)))
    steptime = 0.01
    parms = dict()
    for parm in ["amin", "max_view"]:
        if parm in kwargs:
            parms[parm] = kwargs[parm]
    thw = kwargs["thw"] if "thw" in kwargs else 1.1
    return IDMParameters(speed=kwargs["vego"],
                         init_speed=kwargs["vego"],
                         init_position=-kwargs["dinit"],
                         timestep=steptime,
                         n_reaction=int(reactiontime/steptime),
                         thw=thw,
                         **parms)


def acc_cutin_pars(**kwargs):
    """ Define the ACC parameters in a cut-in scenario.

    :return: Parameter object that can be passed via init_simulation.
    """
    parms = dict()
    for parm in ["amin", "sensor_range", "k1_acc", "k2_acc", "k_cruise"]:
        if parm in kwargs:
            parms[parm] = kwargs[parm]
    return ACCParameters(speed=kwargs["vego"],
                         init_speed=kwargs["vego"],
                         init_position=-kwargs["dinit"],
                         n_reaction=0,
                         **parms)


def acc_idm_cutin_pars(**kwargs):
    """ Define the parameters for the ACCIDM model.

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
    if "reactiontime" not in kwargs:
        kwargs["reactiontime"] = np.random.lognormal(np.log(.92**2/np.sqrt(.92**2+.28**2)),
                                                     np.sqrt(np.log(1+.28**2/.92**2)))
    fcw_delay = kwargs["reactiontime"]
    return ACCIDMPlusParameters(speed=kwargs["vego"],
                                init_speed=kwargs["vego"],
                                init_position=-kwargs["dinit"],
                                n_reaction=0,
                                amin=amin,
                                driver_parms=idm_cutin_pars(**kwargs),
                                driver_model=IDMPlus(),
                                fcw_delay=fcw_delay,
                                **parms)


class SimulationCutIn(SimulationString):
    """ Class for simulation the scenario "approaching slower vehicle". """
    def __init__(self, follower, follower_parameters, **kwargs):
        SimulationString.__init__(self, [LeaderBraking(), follower],
                                  [self._leader_parameters, follower_parameters], **kwargs)
        self.min_simulation_time = 1

    @staticmethod
    def _leader_parameters(**kwargs):
        """ Return the paramters for the leading vehicle. """
        return LeaderBrakingParameters(init_position=0,
                                       init_speed=kwargs["vlead"],
                                       average_deceleration=1,
                                       speed_difference=0,
                                       tconst=5)
