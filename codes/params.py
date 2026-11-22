"""
params.py -- physical parameters for the ball-and-beam (AE 695 project).

Everything is in SI units. The problem statement doesn't give us all the
numbers, so we picked values that look like a typical lab rig (roughly a
1 m beam with a motor at the middle):

  * Ball  : solid steel sphere, m = 0.11 kg, R = 15 mm
            -> J_b = (2/5) m R^2 about its centre
  * Beam  : uniform bar, M = 0.5 kg, L = 1.0 m, pivoted at the centre
            -> J = M L^2 / 12
  * g     : 9.81 m/s^2

All the other scripts pull PARAMS from here, so if you want to try a
different ball or beam, just change it here and rerun.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    m: float = 0.11       # ball mass                     [kg]
    R: float = 0.015      # ball radius                   [m]
    M: float = 0.5        # beam mass                     [kg]
    L: float = 1.0        # beam length                   [m]
    g: float = 9.81       # gravitational acceleration    [m/s^2]

    @property
    def Jb(self) -> float:
        """Ball's moment of inertia about its centre (solid sphere) [kg m^2]."""
        return 2.0 / 5.0 * self.m * self.R ** 2

    @property
    def J(self) -> float:
        """Beam's moment of inertia about the middle pivot (uniform rod) [kg m^2]."""
        return self.M * self.L ** 2 / 12.0

    @property
    def r_max(self) -> float:
        """Furthest the ball can go before it falls off the end [m]."""
        return self.L / 2.0


PARAMS = Params()


if __name__ == "__main__":
    p = PARAMS
    print(f"m = {p.m} kg, R = {p.R} m, J_b = {p.Jb:.3e} kg m^2")
    print(f"M = {p.M} kg, L = {p.L} m, J = {p.J:.4f} kg m^2, g = {p.g} m/s^2")
