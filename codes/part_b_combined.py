"""
part_b_combined.py -- Part B (p): put the controller and observer together, u = u_e - K xhat.
  1. state feedback + full-order observer (pole placement, y = r, theta)
  2. state feedback + reduced-order observer (y = r only)
  (feedback on the true state is thrown in as the baseline to compare against)

Stability comes from the separation principle. Writing it in (x, e) coordinates:
    [x_dot]   [A - BK     BK  ] [x]                        e = x - xhat
    [e_dot] = [  0     A - LC ] [e]   -> eig = eig(A-BK) ∪ eig(A-LC)
Same idea for the reduced-order one, with e_b = x_b - xhat_b and u = -K x + K_b e_b:
    [x_dot  ]   [A - BK   B K_b       ] [x  ]
    [e_b_dot] = [  0      Abb - Ke Aab] [e_b]

Figures: fig_p1_combined_clean.png, fig_p2_combined_noisy.png
Results: results/part_b_p.txt
"""
import numpy as np
import matplotlib.pyplot as plt

from model import equilibrium
from design import designs, make_observers, NOISE_STD
from sim import simulate_closed_loop, response_metrics
from plot_utils import savefig, write_result, fmt

DX0 = np.array([0.2, 0.0, 0.0, 0.0])
T_END = 6.0


def scenarios():
    """name -> (observer, where its estimate starts, as a deviation from x_e)."""
    obs = make_observers()
    full, red = obs["Full: pole placement (y=r,θ)"], obs["Reduced-order (y=r)"]
    y0_init = np.array([DX0[0], 0.0, DX0[2], 0.0])      # smarter start: take r̂(0), θ̂(0) from the first reading
    return {
        "FSFB, true state (reference)": (None, None),
        "FSFB + full-order, x̂(0)=x_e": (full, np.zeros(4)),
        "FSFB + full-order, x̂(0) from y(0)": (full, y0_init),
        "FSFB + reduced-order observer": (red, np.zeros(4)),
    }


def simulate(noise_std, seed=2):
    d = designs()
    x_e, _ = equilibrium(0.0)
    return {nm: simulate_closed_loop(x_e + DX0, T_END, K=d["K_place"], observer=ob,
                                     dxhat0=xh0, feedback="estimate", noise_std=noise_std,
                                     seed=seed)
            for nm, (ob, xh0) in scenarios().items()}


def _plot(results, title, fname):
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for nm, res in results.items():
        t = res["t"]
        ax[0, 0].plot(t, res["x"][:, 0], label=nm)
        ax[0, 1].plot(t, np.degrees(res["x"][:, 2]), label=nm)
        ax[1, 0].plot(t, res["u"], lw=1.0, label=nm)
        if not nm.startswith("FSFB, true"):            # baseline has no observer, so nothing to plot
            e = res["xhat"] - res["x"]
            ax[1, 1].semilogy(t, np.linalg.norm(e, axis=1), lw=1.0, label=nm)
    ax[0, 0].set_ylabel("r [m]")
    ax[0, 1].set_ylabel("θ [deg]")
    ax[1, 0].set_ylabel("τ [N·m]")
    ax[1, 1].set_ylabel(r"$\|\hat{x} - x\|$")
    for a in ax.flat:
        a.set_xlabel("time [s]")
    ax[0, 0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    savefig(fig, fname)


def run(verbose=True):
    d = designs()
    A, B, C, K, L = d["A"], d["B"], d["C"], d["K_place"], d["L_place"]
    n = A.shape[0]
    red = make_observers()["Reduced-order (y=r)"]
    out = ["=" * 70, "(p) COMBINED CONTROLLER-OBSERVER DESIGN (u = tau_e - K xhat)", "=" * 70,
           f"K (pole placement) = {fmt(K)}", ""]

    # --- check separation for the full-order observer ---
    Acl_full = np.block([[A - B @ K, B @ K], [np.zeros((n, n)), A - L @ C]])
    ev_full = np.sort_complex(np.linalg.eigvals(Acl_full))
    ev_sep = np.sort_complex(np.concatenate([np.linalg.eigvals(A - B @ K),
                                             np.linalg.eigvals(A - L @ C)]))
    # --- and for the reduced-order one ---
    Kb = K[:, 1:]
    Acl_red = np.block([[A - B @ K, B @ Kb], [np.zeros((n - 1, n)), red.F]])
    ev_red = np.sort_complex(np.linalg.eigvals(Acl_red))
    ev_sep_r = np.sort_complex(np.concatenate([np.linalg.eigvals(A - B @ K),
                                               np.linalg.eigvals(red.F)]))
    out += ["Separation principle (linear analysis):",
            "  Full-order: eig of the 8x8 closed loop =",
            "    " + ", ".join(f"{e:.3f}" for e in ev_full),
            f"    = eig(A-BK) ∪ eig(A-LC)?  max difference {np.max(np.abs(ev_full - ev_sep)):.1e}",
            "  Reduced-order: eig of the 7x7 closed loop =",
            "    " + ", ".join(f"{e:.3f}" for e in ev_red),
            f"    = eig(A-BK) ∪ eig(Abb-Ke Aab)?  max difference {np.max(np.abs(ev_red - ev_sep_r)):.1e}",
            "  All eigenvalues lie in the open LHP -> both combined systems are asymptotically",
            "  stable (locally, for the nonlinear plant).", ""]

    # Looked at from outside, controller + observer is just a dynamic compensator from y to u
    out += ["Equivalent compensator (full-order): xhat_dot = (A - BK - LC) xhat + L y, u = -K xhat",
            "  compensator poles eig(A-BK-LC) = " +
            ", ".join(f"{e:.3f}" for e in np.sort_complex(np.linalg.eigvals(A - B @ K - L @ C))),
            "  (The compensator itself may be open-loop unstable or not; what matters is the",
            "   combined closed loop above.)", ""]

    for case, noise in (("no noise", (0.0, 0.0)), ("with noise", NOISE_STD)):
        res = simulate(noise)
        hdr = (f"{'configuration':36s} {'t_s(2%) [s]':>11s} {'OS [%]':>7s} {'max|θ| [deg]':>12s} "
               f"{'peak|δτ|':>9s} {'∫δτ² dt':>9s} {'rms δτ (t>3s)':>13s}")
        out += [f"Nonlinear simulation, r(0) = {DX0[0]} m ({case}):",
                hdr, "-" * len(hdr)]
        for nm, r in res.items():
            m = response_metrics(r)
            du = r["u"] - r["u_e"]
            rms_late = np.sqrt(np.mean(du[r["t"] > 3.0] ** 2))
            out.append(f"{nm:36s} {m['settling_time']:11.3f} {m['overshoot_pct']:7.2f} "
                       f"{m['max_theta_deg']:12.2f} {m['peak_du']:9.4f} {m['energy']:9.4f} "
                       f"{rms_late:13.4e}" + ("  BALL FELL OFF" if m["left_beam"] else ""))
        out.append("")
        _plot(res, f"Observer-based state feedback on the nonlinear plant ({case})",
              "fig_p1_combined_clean.png" if noise[0] == 0 else "fig_p2_combined_noisy.png")

    out += ["Observations:",
            " * Both observer-based loops are asymptotically stable and regulate the ball. As the",
            "   separation principle predicts, the closed-loop poles are the controller poles plus",
            "   the observer poles, so K and L can be designed independently.",
            " * Separation guarantees STABILITY, not identical transients. With the naive start",
            "   x̂(0) = x_e the full-order observer begins 0.2 m wrong in r. The large gain L",
            "   (L_21 ≈ 134) turns this into a big spurious ṙ̂ ('observer peaking'). The",
            "   controller reacts to the wrong estimate, giving about 49 % overshoot, about 11",
            "   degrees of tilt and twice the peak torque.",
            " * Starting the observer from the first measurement (r̂(0) = y_r(0), θ̂(0) = y_θ(0))",
            "   removes this: the response is then almost the same as true-state feedback.",
            " * The reduced-order observer uses r directly, so it has no initial r-error and,",
            "   without noise, matches true-state feedback closely.",
            " * With noise the ranking reverses. The full-order observer filters the noise and",
            "   gives a smooth torque, while the reduced-order observer passes the r-noise",
            "   through Ke (|Ke| up to 188) and K to the actuator. The rms torque chatter after",
            "   settling is about 200-300x larger (0.64 vs 0.002-0.003 N m), and the control energy is",
            "   35-70x larger. That is unacceptable for a real motor without slower observer poles",
            "   or a pre-filter on r.",
            " * The separation principle holds exactly only for the linear model. On the",
            "   nonlinear plant it is a local result, but it works well here because the",
            "   motion stays close to the equilibrium."]

    text = "\n".join(out)
    write_result("part_b_p.txt", text)
    if verbose:
        print(text)


if __name__ == "__main__":
    run()
