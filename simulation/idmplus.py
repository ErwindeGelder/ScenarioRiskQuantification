""" Intelligent Driver Model plus (IDM+) from Schakel et al. (2010).

Creation date: 2020 05 28
Author(s): Erwin de Gelder

Modifications:
2020 06 11 If "leading vehicle" is behind, decelerate as quick as possible.
2020 08 12 Add options of having a maximum view. Targets further away are not considered.
"""

from .idm import IDM


class IDMPlus(IDM):
    """ Class for simulation of the Intelligent Driver Model plus (IDM+). """
    def __init__(self):
        IDM.__init__(self)

    def acceleration(self, gap: float, vhost: float, vdiff: float) -> float:
        if gap < 0:
            return max(self.parms.amin, -10)
        if gap > self.parms.max_view:
            return self.parms.a_acc * (1 - self._freeflowpart(vhost))
        return self.parms.a_acc * min(1 - self._freeflowpart(vhost),
                                      1 - self._nonfreeflowpart(gap, vhost, vdiff))
