""" A default class that can be used for the driver models.

Creation date: 2020 08 05
Author(s): Erwin de Gelder

Modifications:
2020 12 13: Add functions set_acceleration and get_acceleration.
"""

from abc import ABC, abstractmethod
import collections
from .options import Options


class StandardParameters(Options):
    """ Default parameters. """
    # pylint: disable=too-many-instance-attributes
    amin: float = -6  # [m/s^2] Minimum acceleration
    speed: float = 30  # [m/s] Desired speed
    thw: float = 1.1  # [s] Desired THW
    n_reaction: int = 0  # Custom parameter (samples of delay)
    timestep: float = 0.01  # Sample time (needed for delay)
    init_position: float = 0
    init_speed: float = 1
    only_positive_speed: bool = True
    cruise_after_collision: bool = False


class StandardState(Options):
    """ Default state """
    position: float = 0
    speed: float = 0
    acceleration: float = 0


class StandardModel(ABC):
    """ Class for modeling driver behaviour """
    def __init__(self):
        self.state = StandardState()
        self.parms = StandardParameters()
        self.accelerations = collections.deque(maxlen=1)

    def init_simulation(self, parms: StandardParameters) -> None:
        """ Initialize the simulation.

        The default parameters:
        amin: float = -6  # [m/s^2] Minimum acceleration
        speed: float = 30  # [m/s] Desired speed
        thw: float = 1  # [s] Desired THW
        n_reaction: int = 0  # Custom parameter (samples of delay)
        timestep: float = 0.01  # Sample time (needed for delay)
        init_position: float = 0
        init_speed: float = 1

        :param parms: The parameters listed above.
        """
        # Set parameters.
        self.parms.amin = parms.amin
        self.parms.speed, self.parms.thw = parms.speed, parms.thw
        self.parms.timestep, self.parms.n_reaction = parms.timestep, parms.n_reaction

        # Create the list with accelerations to account for the delay.
        self.accelerations = collections.deque(maxlen=self.parms.n_reaction+1)
        self.accelerations.append(0)

        # Set state.
        self.state.position = parms.init_position
        self.state.speed = parms.init_speed
        self.state.acceleration = 0

        self.parms.only_positive_speed = parms.only_positive_speed
        self.parms.cruise_after_collision = parms.cruise_after_collision

    def step_simulation(self, leader) -> None:
        """ Compute the state (position, speed, acceleration).

        :param leader: The leading vehicle that contains position and speed.
        """
        self.integration_step()
        self.update(leader.state.position - self.state.position,
                    self.state.speed,
                    self.state.speed - leader.state.speed)

    def update(self, gap: float, vhost: float, vdiff: float) -> None:
        """ Compute a step using the inputs gap, host speed, speed difference.

        :param gap: Gap with preceding vehicle.
        :param vhost: Speed of host vehicle.
        :param vdiff: Difference in speed (vhost - vlead).
        """
        # Calculate acceleration based on the model.
        self.set_acceleration(self.acceleration(gap, vhost, vdiff))

    def get_acceleration(self) -> float:
        """ Obtain the acceleration of the vehicle.

        :return: The current acceleration of the vehicle.
        """
        return self.accelerations[0]

    def set_acceleration(self, acceleration: float) -> None:
        """ Set the acceleration of the vehicle.

        :param acceleration: Acceleration to be set.
        """
        self.accelerations.append(acceleration)

    @abstractmethod
    def acceleration(self, gap: float, vhost: float, vdiff: float) -> float:
        """ Compute the acceleration based on the gap, vhost, vdiff.

        :param gap: Gap with preceding vehicle.
        :param vhost: Speed of host vehicle.
        :param vdiff: Speed of host vehicle minus speed leading vehicle.
        :return: The acceleration.
        """

    def integration_step(self) -> None:
        """ Integrate the acceleration to obtain speed and position.

        Because the state will be updated, there is nothing to return.
        """
        # Update speed
        self.state.acceleration = max(self.parms.amin, self.accelerations[0])
        self.state.speed += self.state.acceleration * self.parms.timestep

        # If vehicle is allowed to only have positive speed, check for this.
        if self.parms.only_positive_speed and self.state.speed < 0:
            self.state.acceleration = 0
            self.state.speed = 0

        # Update position
        self.state.position += self.state.speed * self.parms.timestep
