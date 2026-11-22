"""
design.py -- all the controller and observer gains live here, so parts
(l), (m), (n), (o) and (p) are guaranteed to be using the same ones.

Why these numbers (the report goes into more detail)
----------------------------------------------------
Controller, designed at r_e = 0:
  * Dominant poles with zeta = 0.7 (~4.6 % overshoot) and wn = 2.5 rad/s,
    giving a 2 % settling time of about 4/(zeta wn) = 2.3 s. Felt reasonable
    for a 1 m beam -- not sluggish, but not asking for crazy torques either.
  * Two more real poles at -5 and -6, roughly 3x further out than the dominant pair.
  * Lyapunov (Bass) design with decay rate beta = 4. This method only works if
    -(A + beta I) is Hurwitz, so beta has to beat the stable open-loop pole (3.67).
  * LQR (Lyapunov 2nd method + Riccati), with weights from Bryson's rule.

Observer:
  * Poles at -10, -11, -12, -13, about 5x faster than the controller, so the
    estimate has settled before the controller really gets going.
  * Lyapunov observer: just the dual of the Bass design, with beta_o = 10.
  * Ackermann's formula only works for one output, so that observer only uses y = r.
  * Reduced-order observer (y = r only): poles at -10, -11, -12.
"""
from functools import lru_cache

import numpy as np
from numpy.linalg import matrix_power, inv
from scipy.linalg import solve_continuous_lyapunov, solve_continuous_are
from scipy.signal import place_poles

from model import linearize
from sim import FullOrderObserver, ReducedOrderObserver

R_E = 0.0

ZETA, WN = 0.7, 2.5
_sig, _wd = ZETA * WN, WN * np.sqrt(1 - ZETA**2)
CTRL_POLES = np.array([-_sig + 1j * _wd, -_sig - 1j * _wd, -5.0, -6.0])
LYAP_BETA_CTRL = 4.0
LQR_Q = np.diag(1.0 / np.array([0.05, 0.5, 0.2, 2.0]) ** 2)   # Bryson's rule: 1 / (max allowed)^2
LQR_R = np.array([[1.0 / 0.5**2]])                            # we're OK with |du| up to ~0.5 N m

OBS_POLES = np.array([-10.0, -11.0, -12.0, -13.0])
LYAP_BETA_OBS = 10.0
RED_OBS_POLES = np.array([-10.0, -11.0, -12.0])

NOISE_STD = np.array([2e-3, 1e-3])   # sensor noise std: 2 mm on r, 1 mrad (~0.06 deg) on theta


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------
def ctrb(A, B):
    return np.hstack([matrix_power(A, k) @ B for k in range(A.shape[0])])


def obsv(A, C):
    return np.vstack([C @ matrix_power(A, k) for k in range(A.shape[0])])


def place(A, B, poles):
    """Pole placement via scipy (Kautsky-Nichols / Tits-Yang), for u = -K x."""
    return place_poles(A, B, poles).gain_matrix


def acker(A, B, poles):
    """Ackermann's formula, written out by hand (single input only):
    K = [0 ... 0 1] Mc^{-1} phi_d(A)."""
    n = A.shape[0]
    coeffs = np.real(np.poly(poles))                 # desired char. poly: [1, a1, ..., an]
    phi = sum(c * matrix_power(A, n - i) for i, c in enumerate(coeffs))
    e_n = np.zeros((1, n)); e_n[0, -1] = 1.0
    return e_n @ np.linalg.solve(ctrb(A, B), phi)


def lyapunov_gain(A, B, beta):
    """Bass's Lyapunov-equation method: pick a decay rate beta and get K.

    Solve (A + beta I) P + P (A + beta I)^T = B B^T  (for P > 0 we need
    -(A + beta I) to be Hurwitz), then take K = B^T P^{-1}. Plugging back in,
        (A - BK + beta I) P + P (A - BK + beta I)^T = -B B^T <= 0,
    and because (A, B) is controllable that pushes every eigenvalue of A - BK
    to the left of -beta. The Lyapunov function is V(x) = x^T P^{-1} x."""
    n = A.shape[0]
    P = solve_continuous_lyapunov(A + beta * np.eye(n), B @ B.T)
    return B.T @ inv(P), P


def lqr(A, B, Q, R):
    """LQR (comes out of Lyapunov's 2nd method with V = x^T P x). P solves the CARE
    A^T P + P A - P B R^{-1} B^T P + Q = 0, and then K = R^{-1} B^T P."""
    P = solve_continuous_are(A, B, Q, R)
    return inv(R) @ B.T @ P, P


def observer_place(A, C, poles):
    return place(A.T, C.T, poles).T


def observer_acker(A, C, poles):
    return acker(A.T, C.T, poles).T


def observer_lyapunov(A, C, beta):
    """Bass's method again but on the dual system: L = P^{-1} C^T where
    (A^T + beta I) P + P (A^T + beta I)^T = C^T C, which gives Re eig(A - LC) < -beta."""
    K, P = lyapunov_gain(A.T, C.T, beta)
    return K.T, P


def reduced_observer_gain(A, poles, method="place"):
    """Gain Ke for the reduced-order observer (the measured state is x_a = r)."""
    Abb, Aab = A[1:, 1:], A[0:1, 1:]
    fn = place if method == "place" else acker
    return fn(Abb.T, Aab.T, poles).T


# ---------------------------------------------------------------------------
# All the designs, at the nominal operating point (computed once and cached)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def designs():
    A, B, C, D = linearize(R_E)
    d = dict(A=A, B=B, C=C, D=D)
    d["K_place"] = place(A, B, CTRL_POLES)
    d["K_acker"] = acker(A, B, CTRL_POLES)
    d["K_lyap"], d["P_lyap"] = lyapunov_gain(A, B, LYAP_BETA_CTRL)
    d["K_lqr"], d["P_lqr"] = lqr(A, B, LQR_Q, LQR_R)
    d["L_place"] = observer_place(A, C, OBS_POLES)
    d["L_acker"] = observer_acker(A, C[[0]], OBS_POLES)
    d["L_lyap"], d["P_obs_lyap"] = observer_lyapunov(A, C, LYAP_BETA_OBS)
    d["Ke_red"] = reduced_observer_gain(A, RED_OBS_POLES, "place")
    d["Ke_red_acker"] = reduced_observer_gain(A, RED_OBS_POLES, "acker")
    return d


def make_observers():
    """The four observers we compare in parts (m)-(p)."""
    d = designs()
    A, B, C = d["A"], d["B"], d["C"]
    return {
        "Full: pole placement (y=r,θ)": FullOrderObserver(A, B, C, d["L_place"], (0, 1)),
        "Full: Ackermann (y=r)":        FullOrderObserver(A, B, C[[0]], d["L_acker"], (0,)),
        "Full: Lyapunov (y=r,θ)":       FullOrderObserver(A, B, C, d["L_lyap"], (0, 1)),
        "Reduced-order (y=r)":          ReducedOrderObserver(A, B, d["Ke_red"]),
    }
