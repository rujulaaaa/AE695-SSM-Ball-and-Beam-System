"""
part_a_model.py -- Part A (a), (b), (c): state-space model, equilibria, linearization.

Writes
  results/part_a_abc.txt        readable summary (equations + matrices)
  results/part_a_abc_latex.tex  LaTeX bits we copy straight into the report

Sanity checks along the way
  * NumPy f(x, u) agrees with the SymPy version at a bunch of random states
  * f(x_e, u_e) really is 0 at the equilibria we claim
  * symbolic Jacobians agree with finite-difference ones
"""
import os

import numpy as np
import sympy as sp

from model import (PARAMS, symbolic_model, param_subs, f, equilibrium,
                   linearize, numerical_jacobians)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def _fmt(M, prec=4):
    return np.array2string(np.asarray(M), precision=prec, suppress_small=True,
                           max_line_width=120)


def run(verbose: bool = True):
    os.makedirs(RESULTS, exist_ok=True)
    s = symbolic_model()
    p = PARAMS
    out, tex = [], []

    # ---------------- (a) state-space formulation ----------------
    out += ["=" * 70, "(a) STATE-SPACE FORMULATION", "=" * 70,
            "x = [r, r_dot, theta, theta_dot]^T,  u = tau,  y = [r, theta]^T",
            "", "Lagrangian  L = T - V:",
            f"  T = {s.T}", f"  V = {s.V}",
            "", "Euler-Lagrange equations (each = 0):",
            f"  r     : {s.eom_r}", f"  theta : {s.eom_th}",
            "", "x_dot = f(x, u):"]
    out += [f"  x{i+1}_dot = {sp.simplify(fi)}" for i, fi in enumerate(s.f)]
    out += ["y = h(x, u):"] + [f"  y{i+1} = {hi}" for i, hi in enumerate(s.h)]
    tex += [r"% (a) Nonlinear state-space model",
            r"\dot{\mathbf x} = " + sp.latex(s.f),
            r"\mathbf y = " + sp.latex(s.h)]

    # Make sure the hand-typed NumPy f matches what SymPy derived
    f_sym = sp.lambdify((s.x, s.tau), s.f.subs(param_subs(p)), "numpy")
    rng = np.random.default_rng(0)
    err = 0.0
    for _ in range(20):
        xr = rng.uniform([-0.5, -1, -0.5, -2], [0.5, 1, 0.5, 2])
        ur = rng.uniform(-1, 1)
        err = max(err, np.max(np.abs(np.ravel(f_sym(xr, ur)) - f(xr, ur))))
    out += ["", f"[check] max |f_sympy - f_numpy| over 20 random points = {err:.2e}"]

    # ---------------- (b) equilibrium points ----------------
    eqs = sp.solve(list(s.f), [s.x[1], s.x[3], s.x[2], s.tau], dict=True)
    out += ["", "=" * 70, "(b) EQUILIBRIUM POINTS", "=" * 70,
            "Solving f(x, u) = 0 for (r_dot, theta_dot, theta, tau) with r free:"]
    out += [f"  {e}" for e in eqs]
    out += ["",
            "theta = pi is an upside-down beam (the ball cannot stay on it), so it is excluded.",
            "Physical equilibria:  x_e = [r_e, 0, 0, 0],  tau_e = m g r_e,  |r_e| <= L/2",
            "=> there is a continuum of equilibria, one for each ball position on the beam.",
            f"   with our parameters: tau_e = {p.m*p.g:.4f} * r_e  [N m],  "
            f"|tau_e| <= {p.m*p.g*p.r_max:.4f} N m"]
    for r_e in (0.0, 0.25):
        xe, ue = equilibrium(r_e)
        out.append(f"[check] r_e = {r_e:+.2f}:  |f(x_e, u_e)| = {np.linalg.norm(f(xe, ue)):.2e}")

    # ---------------- (c) linearization ----------------
    out += ["", "=" * 70, "(c) LINEARIZATION  (first-order Taylor expansion)", "=" * 70,
            "delta_x_dot = A delta_x + B delta_u,   delta_y = C delta_x + D delta_u",
            "delta_x = x - x_e,  delta_u = u - u_e", "",
            "General equilibrium r_e (symbolic):",
            f"  A = {s.A.tolist()}", f"  B = {s.B.tolist()}",
            f"  C = {s.C.tolist()}", f"  D = {s.D.tolist()}"]
    tex += [r"% (c) Jacobians at x_e = [r_e,0,0,0], tau_e = m g r_e",
            "A = " + sp.latex(s.A), "B = " + sp.latex(s.B),
            "C = " + sp.latex(s.C), "D = " + sp.latex(s.D)]

    for r_e in (0.0, 0.25):
        A, B, C, D = linearize(r_e)
        xe, ue = equilibrium(r_e)
        An, Bn = numerical_jacobians(xe, ue)
        out += ["", f"Numeric model at r_e = {r_e} m:",
                "A =", _fmt(A), "B =", _fmt(B), "C =", _fmt(C), "D =", _fmt(D),
                f"[check] max|A - A_fd| = {np.max(np.abs(A-An)):.2e}, "
                f"max|B - B_fd| = {np.max(np.abs(B-Bn)):.2e}"]

    out += ["", "Parameters used: "
            f"m={p.m}, R={p.R}, J_b={p.Jb:.3e}, M={p.M}, L={p.L}, J={p.J:.4f}, g={p.g}"]

    text = "\n".join(out)
    with open(os.path.join(RESULTS, "part_a_abc.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(RESULTS, "part_a_abc_latex.tex"), "w") as fh:
        fh.write("\n\n".join(tex) + "\n")
    if verbose:
        print(text)
    return linearize(0.0)


if __name__ == "__main__":
    run()
