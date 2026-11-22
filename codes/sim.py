"""
sim.py -- all the simulation bits.

* simulate_nonlinear_ol : open-loop sim of the nonlinear plant (RK45, adaptive
                          step). Stops early if the ball rolls off the beam.
* simulate_linear_ol    : same thing for the linearized (deviation) model.
* FullOrderObserver     : Luenberger observer  xhat_dot = A xhat + B u + L (y - C xhat)
* ReducedOrderObserver  : Gopinath/Ogata reduced-order observer, only measures r
* simulate_closed_loop  : our own RK4 loop with dt = 1 ms. Steps the nonlinear plant,
                          the observer and the state feedback together. Can add
                          Gaussian sensor noise -- one new sample per step, held
                          for that step, so it acts like a 1 kHz sensor.
* response_metrics / estimation_metrics : the numbers that go in the tables.

Note: the observers and controllers all work with deviations from the
operating point:  dx = x - x_e,  du = u - u_e,  dy = y - y_e.
"""
import numpy as np
from scipy.integrate import solve_ivp

from model import PARAMS, f, h, equilibrium

STATE_SCALE = np.array([0.05, 0.5, 0.2, 2.0])   # rough "normal" sizes of r, r_dot, theta, theta_dot


# ---------------------------------------------------------------------------
# Open-loop simulations
# ---------------------------------------------------------------------------
def simulate_nonlinear_ol(x0, u_fun, t_end, n_pts=2001, p=PARAMS):
    """Integrate x_dot = f(x, u_fun(t)). Returns (t, X, t_off); t_off is when
    |r| hits L/2 (i.e. the ball falls off), or None if it stays on."""
    def rhs(t, x):
        return f(x, u_fun(t), p)

    def off_beam(t, x):
        return p.r_max - abs(x[0])
    off_beam.terminal = True
    off_beam.direction = -1

    sol = solve_ivp(rhs, (0.0, t_end), np.asarray(x0, float), rtol=1e-10, atol=1e-12,
                    max_step=1e-3, dense_output=True, events=off_beam)
    t_off = sol.t_events[0][0] if sol.t_events[0].size else None
    t = np.linspace(0.0, sol.t[-1], n_pts)
    return t, sol.sol(t).T, t_off


def simulate_linear_ol(A, B, dx0, du_fun, t):
    """Integrate the linear model dx_dot = A dx + B du(t), sampled on the grid t."""
    def rhs(tt, dx):
        return A @ dx + B[:, 0] * du_fun(tt)
    sol = solve_ivp(rhs, (t[0], t[-1]), np.asarray(dx0, float), t_eval=t,
                    rtol=1e-10, atol=1e-12, max_step=1e-3)
    return sol.y.T


# ---------------------------------------------------------------------------
# Observers (deviation coordinates)
# ---------------------------------------------------------------------------
class FullOrderObserver:
    """xhat_dot = A xhat + B du + L (dy_used - C_used xhat).
    out_idx picks which sensors this observer gets to use (0 = r, 1 = theta)."""

    def __init__(self, A, B, C_used, L, out_idx=(0, 1), name="full-order"):
        self.A, self.B, self.C, self.L = A, B, C_used, L
        self.out_idx = list(out_idx)
        self.name = name

    def init(self, dxhat0, dy0):
        return np.array(dxhat0, float)

    def zdot(self, z, du, dy):
        y_used = dy[self.out_idx]
        return self.A @ z + self.B[:, 0] * du + self.L @ (y_used - self.C @ z)

    def estimate(self, z, dy):
        return z


class ReducedOrderObserver:
    """Reduced-order observer for when we only measure x_a = r.

    Split the state as x = [x_a; x_b] with x_b = [r_dot, theta, theta_dot]:
        x_a_dot = Aaa x_a + Aab x_b + Ba u ,   x_b_dot = Aba x_a + Abb x_b + Bb u
    We integrate z = xhat_b - Ke y instead of xhat_b directly (avoids needing y_dot):
        z_dot   = (Abb - Ke Aab) z + [(Abb - Ke Aab) Ke + Aba - Ke Aaa] y + (Bb - Ke Ba) u
        xhat_b  = z + Ke y
    and the error then just obeys e_b_dot = (Abb - Ke Aab) e_b."""

    def __init__(self, A, B, Ke, name="reduced-order"):
        Ke = np.asarray(Ke, float).ravel()
        self.Aaa, self.Aab = A[0, 0], A[0, 1:]
        self.Aba, self.Abb = A[1:, 0], A[1:, 1:]
        self.Ba, self.Bb = B[0, 0], B[1:, 0]
        self.Ke = Ke
        self.F = self.Abb - np.outer(Ke, self.Aab)
        self.G = self.F @ Ke + self.Aba - Ke * self.Aaa
        self.H = self.Bb - Ke * self.Ba
        self.out_idx = [0]
        self.name = name

    def init(self, dxhat0, dy0):
        return np.asarray(dxhat0, float)[1:] - self.Ke * dy0[0]

    def zdot(self, z, du, dy):
        return self.F @ z + self.G * dy[0] + self.H * du

    def estimate(self, z, dy):
        return np.concatenate([[dy[0]], z + self.Ke * dy[0]])


# ---------------------------------------------------------------------------
# Closed-loop sim: nonlinear plant + state feedback (+ an observer if we want one)
# ---------------------------------------------------------------------------
def _rk4(fun, x, dt):
    k1 = fun(x)
    k2 = fun(x + 0.5 * dt * k1)
    k3 = fun(x + 0.5 * dt * k2)
    k4 = fun(x + dt * k3)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_closed_loop(x0, t_end, K=None, observer=None, dxhat0=None,
                         feedback="estimate", noise_std=(0.0, 0.0), seed=0,
                         r_e=0.0, dt=1e-3, p=PARAMS):
    """Run the nonlinear plant with u = u_e - K dx_fb.

    feedback = "estimate" : feed back the observer's estimate (proper observer-based control)
    feedback = "true"     : feed back the true state; the observer just tags along
                            so we can see how good its estimate is
    K = None              : no control at all (u = u_e)
    r_e                   : where we want the ball (operating point / set-point)."""
    x_e, u_e = equilibrium(r_e, p)
    y_e = h(x_e)
    n = int(round(t_end / dt)) + 1
    t = np.arange(n) * dt
    X = np.full((n, 4), np.nan)
    Xhat = np.full((n, 4), np.nan)
    U = np.full(n, np.nan)
    Y = np.full((n, 2), np.nan)
    rng = np.random.default_rng(seed)
    noise_std = np.asarray(noise_std, float)

    x = np.asarray(x0, float).copy()
    z = None
    left_beam = None
    k = 0
    for k in range(n):
        dy = h(x) - y_e + rng.standard_normal(2) * noise_std
        if observer is not None:
            if z is None:
                z = observer.init(np.zeros(4) if dxhat0 is None else dxhat0, dy)
            dxhat = observer.estimate(z, dy)
        else:
            dxhat = x - x_e
        dx_fb = (x - x_e) if (feedback == "true" or observer is None) else dxhat
        du = -(K @ dx_fb).item() if K is not None else 0.0
        u = u_e[0] + du

        X[k], Xhat[k], U[k], Y[k] = x, x_e + dxhat, u, y_e + dy
        if abs(x[0]) > p.r_max:
            left_beam = t[k]
            break
        if k == n - 1:
            break
        x = _rk4(lambda xx: f(xx, u, p), x, dt)
        if observer is not None:
            z = _rk4(lambda zz: observer.zdot(zz, du, dy), z, dt)

    s = slice(0, k + 1)
    return dict(t=t[s], x=X[s], xhat=Xhat[s], u=U[s], y=Y[s],
                left_beam=left_beam, u_e=u_e[0], x_e=x_e)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def response_metrics(res, r_ref=0.0, band=0.02):
    """Numbers for the ball-position response: 2 % settling time, overshoot,
    peak |theta|, peak |du|, and control "energy" (integral of du^2 dt)."""
    t, r = res["t"], res["x"][:, 0]
    span = r[0] - r_ref
    if abs(span) < 1e-12:
        span = np.max(np.abs(r - r_ref)) or 1.0
    e = (r - r_ref) / span
    outside = np.where(np.abs(e) > band)[0]
    if outside.size == 0:
        ts = 0.0
    elif outside[-1] + 1 < len(t) and res["left_beam"] is None:
        ts = t[outside[-1] + 1]
    else:
        ts = np.nan
    du = res["u"] - res["u_e"]
    return dict(
        settling_time=ts,
        overshoot_pct=100.0 * max(0.0, -np.min(e)),
        max_theta_deg=np.degrees(np.max(np.abs(res["x"][:, 2]))),
        peak_du=np.max(np.abs(du)),
        energy=np.trapezoid(du**2, t),
        left_beam=res["left_beam"],
    )


def estimation_metrics(res, t_from=1.0, conv_frac=0.02):
    """RMS estimation error for each state (only counting t >= t_from), and a
    convergence time = last time the scaled error norm is above conv_frac of
    its peak."""
    e = res["xhat"] - res["x"]
    t = res["t"]
    rms = np.sqrt(np.mean(e[t >= t_from] ** 2, axis=0))
    en = np.linalg.norm(e / STATE_SCALE, axis=1)
    idx = np.where(en > conv_frac * en.max())[0]
    t_conv = t[idx[-1]] if idx.size else 0.0
    return dict(rms=rms, t_conv=t_conv, e=e)
