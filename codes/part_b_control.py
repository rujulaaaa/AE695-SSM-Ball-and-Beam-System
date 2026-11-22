"""
part_b_control.py -- Part B (l): full-state feedback u = u_e - K (x - x_e), with K from
  1. pole placement        (scipy's place_poles)
  2. Ackermann's formula   (our own version, design.acker)
  3. Lyapunov              (Bass's method, with a chosen decay rate)
  3b. LQR                  (Lyapunov 2nd method + Riccati) -- extra, just to compare

All of them are then tested on the actual NONLINEAR plant (RK4, 1 ms steps),
not just the linear model.
Figures: fig_l1_controllers.png, fig_l2_lin_vs_nonlin.png
Results: results/part_b_l.txt
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

from model import PARAMS, equilibrium
from design import (designs, CTRL_POLES, LYAP_BETA_CTRL, LQR_Q, LQR_R, ZETA, WN)
from sim import simulate_closed_loop, response_metrics
from plot_utils import savefig, write_result, fmt

DX0 = np.array([0.2, 0.0, 0.0, 0.0])   # let go of the ball 20 cm from the centre, beam level
T_END = 6.0


def controller_set():
    d = designs()
    return {
        "Pole placement": d["K_place"],
        "Ackermann": d["K_acker"],
        f"Lyapunov (Bass, β={LYAP_BETA_CTRL:g})": d["K_lyap"],
        "LQR (Lyapunov/Riccati)": d["K_lqr"],
    }


def metrics_table(results, r_ref=0.0):
    hdr = (f"{'controller':34s} {'t_s(2%) [s]':>11s} {'OS [%]':>7s} {'max|θ| [deg]':>12s} "
           f"{'peak|δτ| [Nm]':>13s} {'∫δτ² dt':>9s}")
    rows = [hdr, "-" * len(hdr)]
    for nm, res in results.items():
        m = response_metrics(res, r_ref)
        rows.append(f"{nm:34s} {m['settling_time']:11.3f} {m['overshoot_pct']:7.2f} "
                    f"{m['max_theta_deg']:12.2f} {m['peak_du']:13.4f} {m['energy']:9.4f}"
                    + ("  (BALL FELL OFF)" if m["left_beam"] is not None else ""))
    return rows


def run(verbose=True):
    d = designs()
    A, B = d["A"], d["B"]
    n = A.shape[0]
    Ks = controller_set()
    out = ["=" * 70, "(l) FULL-STATE FEEDBACK  u = tau_e - K (x - x_e)", "=" * 70,
           f"Design point r_e = 0.  Specifications: zeta = {ZETA}, wn = {WN} rad/s",
           f"  -> dominant poles {CTRL_POLES[0]:.3f}, {CTRL_POLES[1]:.3f}; extra poles "
           f"{CTRL_POLES[2].real:g}, {CTRL_POLES[3].real:g}",
           f"  -> expected overshoot ~ {100*np.exp(-np.pi*ZETA/np.sqrt(1-ZETA**2)):.1f} %, "
           f"t_s(2%) ~ {4/(ZETA*WN):.2f} s", ""]

    out += ["Gains and closed-loop eigenvalues:"]
    for nm, K in Ks.items():
        ev = np.sort_complex(np.linalg.eigvals(A - B @ K))
        out += [f"  {nm:30s} K = {fmt(K, 4)}",
                f"  {'':30s} eig(A-BK) = " + ", ".join(f"{e:.3f}" for e in ev)]
    out += ["", "Analytical justification:",
            f"  * Pole placement and Ackermann: for a single-input controllable system the gain",
            f"    that places a given set of poles is unique, so the two must agree.",
            f"    max|K_place - K_acker| = {np.max(np.abs(d['K_place'] - d['K_acker'])):.2e}",
            "    Ackermann: K = [0 0 0 1] Mc^{-1} phi_d(A), phi_d(s) = prod(s - p_i)."]

    P = d["P_lyap"]
    Acl = A - B @ d["K_lyap"]
    res_lyap = (Acl + LYAP_BETA_CTRL * np.eye(n)) @ P + P @ (Acl + LYAP_BETA_CTRL * np.eye(n)).T \
        + B @ B.T
    out += ["  * Lyapunov (Bass): solve (A + bI)P + P(A + bI)^T = B B^T with b = "
            f"{LYAP_BETA_CTRL:g}",
            f"    (valid because -(A + bI) is Hurwitz: b > {max(-np.linalg.eigvals(A).real):.3f}).",
            f"    eig(P) = {fmt(np.linalg.eigvalsh(P), 6)} > 0 ;  K = B^T P^-1",
            "    Closed loop: (A_cl + bI)P + P(A_cl + bI)^T = -B B^T  "
            f"(residual {np.max(np.abs(res_lyap)):.1e})",
            "    so V(x) = x^T P^-1 x satisfies V_dot <= -2b V, i.e. Re eig(A - BK) < -b.",
            f"    max Re eig(A-BK) = {max(np.linalg.eigvals(Acl).real):.3f} < -{LYAP_BETA_CTRL:g}  ✔"]

    Pq = d["P_lqr"]
    Aq = A - B @ d["K_lqr"]
    res_lqr = Aq.T @ Pq + Pq @ Aq + LQR_Q + d["K_lqr"].T @ LQR_R @ d["K_lqr"]
    out += ["  * LQR: P from the CARE; V = x^T P x satisfies the Lyapunov equation",
            "    (A-BK)^T P + P(A-BK) = -(Q + K^T R K) < 0  "
            f"(residual {np.max(np.abs(res_lqr)):.1e}), eig(P) = {fmt(np.linalg.eigvalsh(Pq), 4)}",
            "    Q = diag(" + ", ".join(f"{q:g}" for q in np.diag(LQR_Q)) + f"), R = {LQR_R[0,0]:g} (Bryson's rule)", ""]

    # ----- now try them on the real (nonlinear) plant -----
    x_e, _ = equilibrium(0.0)
    results = {nm: simulate_closed_loop(x_e + DX0, T_END, K=K) for nm, K in Ks.items()}
    out += [f"Nonlinear validation, ball released at r(0) = {DX0[0]} m (at rest, beam level):"]
    out += metrics_table(results)

    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    styles = ["-", "--", "-", "-"]
    for (nm, res), ls in zip(results.items(), styles):
        t = res["t"]
        ax[0, 0].plot(t, res["x"][:, 0], ls, label=nm)
        ax[0, 1].plot(t, np.degrees(res["x"][:, 2]), ls, label=nm)
        ax[1, 0].plot(t, res["u"], ls, label=nm)
    for (nm, K), mk in zip(Ks.items(), ["x", "+", "x", "x"]):
        ev = np.linalg.eigvals(A - B @ K)
        ax[1, 1].plot(ev.real, ev.imag, mk, ms=11 if mk == "+" else 8, mew=2, label=nm)
    ev0 = np.linalg.eigvals(A)
    ax[1, 1].plot(ev0.real, ev0.imag, "ko", mfc="none", label="open loop")
    ax[1, 1].axvline(-LYAP_BETA_CTRL, color="gray", ls=":", lw=0.9)
    ax[1, 1].axvline(0, color="k", lw=0.8)
    ax[0, 0].set_ylabel("r [m]")
    ax[0, 1].set_ylabel("θ [deg]")
    ax[1, 0].set_ylabel("τ [N·m]")
    ax[1, 1].set_xlabel("Re(λ)")
    ax[1, 1].set_ylabel("Im(λ)")
    ax[1, 1].set_title("closed-loop eigenvalues (dotted: Re = -β)")
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.set_xlabel("time [s]")
    ax[0, 0].legend()
    ax[1, 1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle(f"State feedback on the nonlinear ball and beam, r(0) = {DX0[0]} m")
    fig.tight_layout()
    savefig(fig, "fig_l1_controllers.png")

    # ----- how well does the linear prediction hold up? (+ a set-point change) -----
    K = d["K_place"]
    Acl = A - B @ K
    cases = [("r(0) = 0.2 m → 0", 0.0, np.array([0.2, 0, 0, 0])),
             ("r(0) = 0.45 m → 0 (near beam end)", 0.0, np.array([0.45, 0, 0, 0])),
             ("set-point 0 → 0.25 m (K designed at r_e = 0)", 0.25, np.array([-0.25, 0, 0, 0]))]
    fig, ax = plt.subplots(2, 3, figsize=(14, 6.5))
    out += ["", "Pole-placement controller: nonlinear vs linear closed loop"]
    for j, (lab, r_ref, dx0) in enumerate(cases):
        xr, ur = equilibrium(r_ref)
        res = simulate_closed_loop(xr + dx0, T_END, K=K, r_e=r_ref)
        t = res["t"]
        lin = xr + np.array([expm(Acl * tt) @ dx0 for tt in t])
        ax[0, j].plot(t, res["x"][:, 0], label="nonlinear")
        ax[0, j].plot(t, lin[:, 0], "--", label="linear")
        ax[1, j].plot(t, np.degrees(res["x"][:, 2]), label="nonlinear")
        ax[1, j].plot(t, np.degrees(lin[:, 2]), "--", label="linear")
        ax[0, j].set_title(lab)
        ax[0, j].set_ylabel("r [m]")
        ax[1, j].set_ylabel("θ [deg]")
        ax[1, j].set_xlabel("time [s]")
        ax[0, j].legend()
        m = response_metrics(res, r_ref)
        out.append(f"  {lab:46s}: max|r_nl - r_lin| = {np.max(np.abs(res['x'][:,0]-lin[:,0])):.4f} m, "
                   f"t_s = {m['settling_time']:.2f} s, OS = {m['overshoot_pct']:.1f} %, "
                   f"max|θ| = {m['max_theta_deg']:.1f} deg, peak|δτ| = {m['peak_du']:.3f} N m")
    fig.tight_layout()
    savefig(fig, "fig_l2_lin_vs_nonlin.png")

    out += ["",
            "Observations:",
            " * All four controllers stabilise the nonlinear plant from r(0) = 0.2 m.",
            " * Pole placement and Ackermann give identical gains and identical responses.",
            " * The Lyapunov (Bass) design guarantees decay faster than e^{-bt}, but it gives no",
            "   direct control over damping or effort. Here it is the fastest and most aggressive",
            "   (largest torque).",
            " * LQR settles as fast as the Bass design but with less than half the peak torque and",
            "   about a third of the control energy: optimising a cost beats merely enforcing a",
            "   decay rate.",
            " * Pole placement is the gentlest (smallest torque, 3 deg of tilt). Its settling time",
            "   (2.75 s) is a little longer than the 2nd-order estimate (2.29 s) because the",
            "   extra poles at -5 and -6 are not far enough away to be negligible.",
            " * Linear and nonlinear closed loops agree closely (within 1 mm from 0.2 m, about 1 cm",
            "   from 0.45 m). The gap grows with the offset as sin θ, r θ_dot^2 and the r-dependent",
            "   inertia become noticeable. The same gain also works at r_e = 0.25 m, where A41 and",
            "   B4 differ by only about 14 % from their values at r_e = 0 (part c)."]

    text = "\n".join(out)
    write_result("part_b_l.txt", text)
    if verbose:
        print(text)


if __name__ == "__main__":
    run()
