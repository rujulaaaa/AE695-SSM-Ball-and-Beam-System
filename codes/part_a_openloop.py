"""
part_a_openloop.py -- Part A (d) open-loop runs, nonlinear vs linear, and
(e) eigenvalues/eigenvectors and what they say about stability.

Figures
  fig_d1_openloop_ic.png      : responses to a few initial offsets at r_e = 0
  fig_d2_openloop_other.png   : a different equilibrium (r_e = 0.25 m), plus a torque pulse
  fig_e1_poles.png            : open-loop poles, and how the unstable one changes with r_e
Results
  results/part_a_de.txt
"""
import numpy as np
import matplotlib.pyplot as plt

from model import PARAMS, equilibrium, linearize
from sim import simulate_nonlinear_ol, simulate_linear_ol
from plot_utils import savefig, write_result, fmt

T_END = 2.0


def compare(r_e, dx0, du_fun, t_end=T_END):
    """Run the nonlinear plant and the linear model side by side from x_e + dx0.
    t_div marks when they start to disagree: the first time the nonlinear r is
    off from the linear r by more than 10 % of the biggest linear deviation so far."""
    x_e, u_e = equilibrium(r_e)
    A, B, _, _ = linearize(r_e)
    dx0 = np.asarray(dx0, float)
    t, Xn, t_off = simulate_nonlinear_ol(x_e + dx0, lambda tt: u_e[0] + du_fun(tt), t_end)
    Xl = x_e + simulate_linear_ol(A, B, dx0, du_fun, t)
    dn, dl = Xn[:, 0] - r_e, Xl[:, 0] - r_e
    scale = np.maximum.accumulate(np.abs(dl))
    ok = scale > 1e-4
    rel = np.zeros_like(t)
    rel[ok] = np.abs(dn[ok] - dl[ok]) / scale[ok]
    idx = np.where(rel > 0.10)[0]
    th_err = np.max(np.abs(Xn[:, 2] - Xl[:, 2])) / max(np.max(np.abs(Xl[:, 2] - x_e[2])), 1e-9)
    return dict(t=t, Xn=Xn, Xl=Xl, t_off=t_off, t_div=t[idx[0]] if idx.size else np.nan,
                r_e=r_e, max_rel_r=rel.max(), max_rel_th=th_err,
                th_end=np.degrees(Xn[-1, 2]))


def _plot_cases(cases, results, fname):
    fig, axes = plt.subplots(2, len(cases), figsize=(4.6 * len(cases), 6.2), squeeze=False)
    for j, ((label, *_), res) in enumerate(zip(cases, results)):
        t = res["t"]
        for i, (k, name) in enumerate([(0, r"$r$ [m]"), (2, r"$\theta$ [deg]")]):
            sc = 1.0 if k == 0 else 180 / np.pi
            ax = axes[i, j]
            ax.plot(t, res["Xn"][:, k] * sc, label="nonlinear")
            ax.plot(t, res["Xl"][:, k] * sc, "--", label="linearized")
            if res["t_off"] is not None:
                ax.axvline(res["t_off"], color="k", lw=0.8, ls=":")
            if np.isfinite(res["t_div"]):
                ax.axvline(res["t_div"], color="tab:red", lw=0.8, ls="-.",
                           label=f"10% divergence ({res['t_div']:.2f} s)")
            if k == 0:
                ax.axhline(PARAMS.r_max, color="gray", lw=0.8)
                ax.axhline(-PARAMS.r_max, color="gray", lw=0.8)
                ax.set_title(label)
            ax.set_ylabel(name)
            ax.set_xlabel("time [s]")
            ax.legend(loc="best")
    fig.tight_layout()
    return savefig(fig, fname)


def run(verbose=True):
    out = ["=" * 70, "(d) OPEN-LOOP RESPONSE: NONLINEAR vs LINEARIZED", "=" * 70,
           "Input held at the equilibrium torque (du = 0) unless stated otherwise.",
           "Simulation stops when the ball reaches the end of the beam (|r| = L/2 = "
           f"{PARAMS.r_max} m; dotted line).", ""]

    zero = lambda t: 0.0
    cases1 = [("r_e = 0, δr(0) = 1 cm", 0.0, [0.01, 0, 0, 0], zero),
              ("r_e = 0, δr(0) = 5 cm", 0.0, [0.05, 0, 0, 0], zero),
              ("r_e = 0, δθ(0) = 1°", 0.0, [0, 0, np.radians(1.0), 0], zero)]
    pulse = lambda t: 0.02 if t < 0.1 else 0.0
    cases2 = [("r_e = 0.25 m, δr(0) = 1 cm", 0.25, [0.01, 0, 0, 0], zero),
              ("r_e = 0, torque pulse 0.02 N·m × 0.1 s", 0.0, [0, 0, 0, 0], pulse)]

    for cases, fname in ((cases1, "fig_d1_openloop_ic.png"),
                         (cases2, "fig_d2_openloop_other.png")):
        results = [compare(r_e, dx0, du) for _, r_e, dx0, du in cases]
        _plot_cases(cases, results, fname)
        for (label, *_), res in zip(cases, results):
            t_off = f"{res['t_off']:.3f} s" if res["t_off"] is not None else f"> {T_END} s"
            div = (f"{res['t_div']:.3f} s" if np.isfinite(res["t_div"])
                   else "never (until ball leaves)")
            out.append(f"  {label:40s} leaves beam at {t_off:>8s} | max rel. error r: "
                       f"{100*res['max_rel_r']:5.1f}%, theta: {100*res['max_rel_th']:5.1f}% | "
                       f">10% at: {div} | theta_end = {res['th_end']:+.1f} deg")
    out += ["",
            "Observations:",
            "  * Both models diverge, because the linear model has an eigenvalue in the RHP (part e).",
            "  * The linear model predicts the onset of instability (direction, growth rate) well.",
            "    Relative r-errors stay small because the divergence is dominated by the same",
            "    unstable eigenvalue. Discrepancies grow with the perturbation size and with time,",
            "    because sin(theta) ~ theta and the r*theta_dot^2, r*r_dot*theta_dot and m r^2",
            "    inertia terms stop being negligible as r and theta grow.",
            "  * The linear model has no |r| <= L/2 limit and keeps growing exponentially, while",
            "    the real ball simply falls off the beam."]

    # ---------------- (e) eigen-analysis ----------------
    A, B, C, D = linearize(0.0)
    lam, V = np.linalg.eig(A)
    order = np.argsort(-lam.real)
    lam, V = lam[order], V[:, order]
    Vn = V / V[0, :]                              # normalise so r-component = 1 (easier to read)
    omega = abs(lam.real).max()
    out += ["", "=" * 70, "(e) EIGENVALUES, EIGENVECTORS, STABILITY (r_e = 0)", "=" * 70,
            "Characteristic polynomial: det(sI - A) = " + fmt(np.real(np.poly(A)), 4),
            f"  i.e. s^4 - a c = 0 with a c = {-A[1,2]*-A[3,0]:.4f}, so the |eigenvalues| are "
            f"(a c)^(1/4) = {omega:.4f} rad/s", "",
            "Eigenvalues: " + ", ".join(f"{l:.4f}" for l in lam), "",
            "Eigenvectors (columns, scaled so that the r-component = 1):", fmt(Vn, 4), "",
            "Mode interpretation:",
            f"  lambda = +{omega:.3f} : divergent exponential mode (time constant {1/omega:.3f} s)",
            f"  lambda = -{omega:.3f} : decaying exponential mode",
            f"  lambda = ±j{omega:.3f}: undamped oscillation, period {2*np.pi/omega:.3f} s",
            "  Eigenvector of the unstable mode: theta = -(omega^2/a) r, i.e. the ball rolls",
            "  downhill while the beam keeps tilting further (the ball's weight makes the torque).",
            "=> One eigenvalue has Re > 0, so the equilibrium is UNSTABLE (in Lyapunov's sense and",
            "   also not BIBO stable). Feedback control is required.", ""]

    r_grid = np.linspace(-PARAMS.r_max, PARAMS.r_max, 201)
    w_grid = np.array([max(np.linalg.eigvals(linearize(r)[0]).real) for r in r_grid])
    out.append(f"Unstable eigenvalue vs r_e: {w_grid.min():.4f} (at |r_e| = L/2) ... "
               f"{w_grid.max():.4f} (at r_e = 0). Every equilibrium on the beam is unstable.")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(lam.real, lam.imag, "x", ms=10, mew=2)
    ax[0].axvline(0, color="k", lw=0.8)
    ax[0].axhline(0, color="k", lw=0.8)
    for l in lam:
        ax[0].annotate(f"{l.real:+.2f}{l.imag:+.2f}j", (l.real, l.imag),
                       textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax[0].set_xlabel("Re(λ)")
    ax[0].set_ylabel("Im(λ)")
    ax[0].set_title("Open-loop eigenvalues (r_e = 0)")
    ax[0].set_aspect("equal", "datalim")
    ax[1].plot(r_grid, w_grid)
    ax[1].set_xlabel(r"equilibrium ball position $r_e$ [m]")
    ax[1].set_ylabel("unstable eigenvalue [1/s]")
    ax[1].set_title("Unstable eigenvalue across all equilibria")
    fig.tight_layout()
    savefig(fig, "fig_e1_poles.png")

    text = "\n".join(out)
    write_result("part_a_de.txt", text)
    if verbose:
        print(text)


if __name__ == "__main__":
    run()
