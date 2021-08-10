""" Model of ACC with FCW and IDM+ to take over from Xiao et al. (2017).

Creation date: 2020 08 12
Author(s): Erwin de Gelder

Modifications:
"""

import numpy as np
from .acc import ACC, ACCParameters, ACCState
from .idm import IDMParameters
from .idmplus import IDMPlus


class ACCIDMPlusParameters(ACCParameters):
    """ Parameters for the ACCIDMPlus. """
    fcw_threshold: float = 0.75  # [0-1] If probability is above, give warning.
    fcw_delay: float = 1.0  # [s] After this delay, driver takes over.
    driver_takeover_speed: float = 15.0  # [m/s] Speed difference at which driver takes control.
    driver_takeover_view: float = 150  # [m] Only take over if car is within this distance.
    driver_model: IDMPlus = None  # The model of the driver.
    driver_parms: IDMParameters = None  # Parameters of the driver model.

    def __init__(self, **kwargs):
        self.driver_model = IDMPlus()
        ACCParameters.__init__(self, **kwargs)


class ACCIDMPlusState(ACCState):
    """ State of the ACCIDMPlus. """
    fcw: bool = False
    samples_since_fcw: int = 0
    samples_in_view: int = 0
    driver_takeover: bool = False


class ACCIDMPlus(ACC):
    """ Class for simulation of the human (IDMplus) + ACC. """
    def __init__(self):
        ACC.__init__(self)
        self.parms = ACCIDMPlusParameters()
        self.state = ACCIDMPlusState()

        self.nstep = 0

    def init_simulation(self, parms: ACCIDMPlusParameters) -> None:
        """ Initialize the simulation.

        See the ACC for the default parameters.
        The following additional parameters can also be set:
        fcw_threshold: float = 0.75  # [0-1] If probability is above, give warning.
        fcw_delay: float = 1.0  # [s] After this delay, driver takes over.
        driver_takeover_speed: float = 15.0  # [m/s] Speed difference at which driver takes control.
        driver_takeover_view: float = 150  # [m] Only take over if car is within this distance.
        driver_model: StandardModel = None  # The model of the driver.
        driver_parms: StandardParameters = None  # Parameters of the driver model.

        :param parms: The parameters listed above.
        """
        # Set the parameters of the ACC model.
        ACC.init_simulation(self, parms)

        # Initialize the driver model.
        self.parms.driver_model = parms.driver_model
        self.parms.driver_parms = parms.driver_parms
        self.parms.driver_model.init_simulation(parms.driver_parms)

        # Initialize parameters of the Forward Collision Warning
        self.parms.fcw_delay, self.parms.fcw_threshold = parms.fcw_delay, parms.fcw_threshold

        # Reset the state regarding the takeover.
        self.state.fcw = False
        self.state.samples_since_fcw = 0
        self.state.samples_in_view = 0
        self.state.driver_takeover = False

        self.nstep = 0

    def step_simulation(self, leader) -> None:
        self.nstep += 1
        self.integration_step()

        # Update the driver model.
        self.parms.driver_model.step_simulation(leader)

        # If the FCW is active for longer than `fcw_delay`, the driver is active.
        # Note: additional requirement is that the car should be in the viewing range for at least
        # as long as the reactiontime of the driver.
        if leader.state.position-self.state.position < self.parms.driver_parms.max_view:
            self.state.samples_in_view += 1
        if self.state.fcw:
            self.state.samples_since_fcw += 1
        elif not self.state.driver_takeover:
            self.state.fcw = self.fcw_warning(leader)
        if not self.state.driver_takeover and self.state.fcw:
            if self.state.samples_since_fcw*self.parms.timestep >= self.parms.fcw_delay and \
                    self.state.samples_in_view > self.parms.driver_parms.n_reaction+1:
                self.state.driver_takeover = True

        # Following Xiao et al. (2017), the driver takes over if approaching speed > 15 m/s and
        # car is within 150 m. The speed (15 m/s) and view (150 m) are parameterized with
        # `driver_take_over_speed`, `driver_take_over_view`.
        # As a additional requirement, the driver should brake.
        if not self.state.driver_takeover:
            if self.state.speed-leader.state.speed > self.parms.driver_takeover_speed and \
                (leader.state.position-self.state.position) < self.parms.driver_takeover_view and \
                    self.parms.driver_model.state.acceleration < 0:
                self.state.driver_takeover = True

        if self.state.driver_takeover:
            # Update our own states with that from the driver model.
            self.state.position = self.parms.driver_model.state.position
            self.state.speed = self.parms.driver_model.state.speed
            self.state.acceleration = self.parms.driver_model.state.acceleration
        else:
            # Update the states of the driver model with that from the ACC.
            self.update(leader.state.position-self.state.position,
                        self.state.speed,
                        self.state.speed-leader.state.speed)

            self.parms.driver_model.state.position = self.state.position
            self.parms.driver_model.state.speed = self.state.speed
            self.parms.driver_model.state.acceleration = self.state.acceleration

    def fcw_warning(self, leader) -> bool:
        """ Issue a FCW based on the model of Kiefer et al.

        :param leader: The leading vehicle that contains position and speed.
        :return: Whether an FCW will be issued.
        """
        inv_ttc = ((self.state.speed - leader.state.speed) /
                   (leader.state.position - self.state.position))
        if leader.state.speed > 0 > leader.state.acceleration:
            tmp = -6.092 + 18.816*inv_ttc + 0.119*self.state.speed
        elif leader.state.speed > 0:
            tmp = -6.092 + 12.584*inv_ttc + 0.119*self.state.speed
        else:
            tmp = -9.073 + 24.225*inv_ttc + 0.119*self.state.speed

        probability = 1 / (1 + np.exp(-tmp))
        if probability > self.parms.fcw_threshold:
            return True
        return False
