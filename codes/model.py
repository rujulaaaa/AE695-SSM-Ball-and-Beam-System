"""
model.py -- the nonlinear ball-and-beam model, its equilibria, and the linearization.

System
------
A ball rolls (no slipping) along a beam that pivots about its middle, and a
motor at the pivot puts a torque tau on the beam.

    State   x = [x1, x2, x3, x4] = [r, r_dot, theta, theta_dot]
              r      : ball position along the beam, measured from the pivot [m]
              theta  : beam angle from the horizontal (positive raises the end
                       where r > 0)                                       [rad]
    Input   u = tau  : torque applied to the beam at the pivot            [N m]
    Output  y = [r, theta]  (ball position sensor + beam angle encoder)

Assumptions we made
-------------------
  A1. No slipping, so the ball spins at r_dot / R.
  A2. Solid uniform sphere for the ball, uniform rigid rod for the beam.
  A3. No friction or damping anywhere, and the motor gives exactly the torque
      we ask for straight away (we ignore motor dynamics).
  A4. Ball treated as a point mass sitting on the beam's centre-line, and we
      ignore the beam's rotation adding to the ball's spin (the usual textbook
      simplification).
  A5. The ball never loses contact with the beam and |r| <= L/2.

Equations of motion (Lagrange)
------------------------------
  T = 1/2 m (r_dot^2 + r^2 theta_dot^2) + 1/2 J_b (r_dot/R)^2 + 1/2 J theta_dot^2
  V = m g r sin(theta)

  (m + J_b/R^2) r_ddot   = m r theta_dot^2 - m g sin(theta)
  (m r^2 + J)   theta_ddot = tau - 2 m r r_dot theta_dot - m g r cos(theta)

What's in here
  * symbolic_model()  : Lagrange derivation in SymPy (mostly for the report)
  * f(x, u), h(x, u)  : plain NumPy versions, these are what the sims call
  * equilibrium(r_e)  : equilibrium state + torque for the ball sitting at r_e
  * linearize(r_e)    : numeric A, B, C, D about that equilibrium
"""
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import sympy as sp

from params import PARAMS, Params

N_STATES, N_INPUTS, N_OUTPUTS = 4, 1, 2
STATE_NAMES = [r"$r$ [m]", r"$\dot r$ [m/s]", r"$\theta$ [rad]", r"$\dot\theta$ [rad/s]"]
OUTPUT_NAMES = [r"$r$ [m]", r"$\theta$ [rad]"]


# ---------------------------------------------------------------------------
# Symbolic derivation (Lagrangian mechanics)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def symbolic_model():
    """Do the whole derivation symbolically. Gives back a namespace holding the
    Lagrangian, f and h, the equilibrium substitutions and the symbolic
    Jacobians A, B, C, D. It's cached, since SymPy is slow and we call this a lot."""
    t = sp.symbols("t", real=True)
    m, R, Jb, J, g = sp.symbols("m R J_b J g", positive=True)
    tau = sp.symbols("tau", real=True)
    r = sp.Function("r", real=True)(t)
    th = sp.Function("theta", real=True)(t)
    rd, thd = r.diff(t), th.diff(t)

    # KE = ball sliding along/around + ball spinning (rolling) + beam turning
    T = (sp.Rational(1, 2) * m * (rd**2 + r**2 * thd**2)
         + sp.Rational(1, 2) * Jb * (rd / R) ** 2
         + sp.Rational(1, 2) * J * thd**2)
    V = m * g * r * sp.sin(th)          # only the ball's height changes (beam pivots at its CG)
    Lag = T - V

    # Euler-Lagrange: d/dt(dL/dq_dot) - dL/dq = Q. Torque only acts on theta, so Q_r = 0.
    eom_r = sp.simplify(sp.diff(Lag.diff(rd), t) - Lag.diff(r))
    eom_th = sp.simplify(sp.diff(Lag.diff(thd), t) - Lag.diff(th) - tau)
    sol = sp.solve([eom_r, eom_th], [r.diff(t, 2), th.diff(t, 2)], dict=True)[0]

    # Swap r, r_dot, theta, theta_dot for plain symbols x1..x4.
    # (xreplace matches the biggest sub-trees first, so the derivatives get
    # replaced before r(t) itself does -- otherwise we'd end up with d/dt(x1).)
    x1, x2, x3, x4 = sp.symbols("x1:5", real=True)
    to_state = {rd: x2, thd: x4, r: x1, th: x3}
    r_dd = sp.simplify(sol[r.diff(t, 2)].xreplace(to_state))
    th_dd = sp.simplify(sol[th.diff(t, 2)].xreplace(to_state))

    x = sp.Matrix([x1, x2, x3, x4])
    u = sp.Matrix([tau])
    f = sp.Matrix([x2, r_dd, x4, th_dd])
    h = sp.Matrix([x1, x3])

    # Equilibria (x_dot = 0): there's one for every ball position r_e
    r_e = sp.symbols("r_e", real=True)
    eq_subs = {x1: r_e, x2: 0, x3: 0, x4: 0, tau: m * g * r_e}

    A = sp.simplify(f.jacobian(x).subs(eq_subs))
    B = sp.simplify(f.jacobian(u).subs(eq_subs))
    C = h.jacobian(x)
    D = h.jacobian(u)

    return SimpleNamespace(
        t=t, m=m, R=R, Jb=Jb, J=J, g=g, tau=tau, r_e=r_e,
        T=T, V=V, L=Lag, eom_r=eom_r, eom_th=eom_th,
        x=x, u=u, f=f, h=h, eq_subs=eq_subs, A=A, B=B, C=C, D=D,
    )


def param_subs(p: Params = PARAMS) -> dict:
    """Substitution dict: SymPy parameter symbols -> our actual numbers."""
    s = symbolic_model()
    return {s.m: p.m, s.R: p.R, s.Jb: p.Jb, s.J: p.J, s.g: p.g}


# ---------------------------------------------------------------------------
# Numeric model (this is what all the simulations actually use)
# ---------------------------------------------------------------------------
def f(x, u, p: Params = PARAMS) -> np.ndarray:
    """Nonlinear x_dot = f(x, u). u can be a plain number or a 1-element array."""
    r, rd, th, thd = x
    tau = float(np.squeeze(u))
    m_eff = p.m + p.Jb / p.R**2                          # rolling makes the ball "feel" heavier
    r_dd = (p.m * r * thd**2 - p.m * p.g * np.sin(th)) / m_eff
    th_dd = (tau - 2 * p.m * r * rd * thd - p.m * p.g * r * np.cos(th)) / (p.m * r**2 + p.J)
    return np.array([rd, r_dd, thd, th_dd])


def h(x, u=None) -> np.ndarray:
    """Outputs y = [r, theta] -- we measure ball position and beam angle."""
    return np.array([x[0], x[2]])


def equilibrium(r_e: float = 0.0, p: Params = PARAMS):
    """Equilibrium state and torque with the ball parked at r_e.

    Setting f(x_e, u_e) = 0 gives r_dot = theta_dot = 0 and sin(theta) = 0,
    so theta = 0. (theta = pi also works on paper, but that's the beam flipped
    upside down, so we drop it.) The torque then just has to cancel the ball's
    weight: tau_e = m g r_e."""
    x_e = np.array([r_e, 0.0, 0.0, 0.0])
    u_e = np.array([p.m * p.g * r_e])
    return x_e, u_e


def linearize(r_e: float = 0.0, p: Params = PARAMS):
    """Numeric A, B, C, D, linearized about the equilibrium at r_e."""
    s = symbolic_model()
    subs = {**param_subs(p), s.r_e: r_e}
    to_np = lambda M: np.array(M.subs(subs).evalf(), dtype=float)
    return to_np(s.A), to_np(s.B), to_np(s.C), to_np(s.D)


def numerical_jacobians(x0, u0, p: Params = PARAMS, eps: float = 1e-6):
    """Finite-difference (central) Jacobians of f -- just a sanity check on the
    symbolic A and B."""
    x0 = np.asarray(x0, float)
    u0 = np.atleast_1d(np.asarray(u0, float))
    A = np.zeros((N_STATES, N_STATES))
    B = np.zeros((N_STATES, N_INPUTS))
    for j in range(N_STATES):
        dx = np.zeros(N_STATES); dx[j] = eps
        A[:, j] = (f(x0 + dx, u0, p) - f(x0 - dx, u0, p)) / (2 * eps)
    for j in range(N_INPUTS):
        du = np.zeros(N_INPUTS); du[j] = eps
        B[:, j] = (f(x0, u0 + du, p) - f(x0, u0 - du, p)) / (2 * eps)
    return A, B
