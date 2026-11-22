"""
part_b_observers.py -- Part B (m) full-order observers (pole placement, Ackermann,
Lyapunov), (n) reduced-order observer (y = r only), (o) how they cope with and
without sensor noise.

How we test them (same for every observer): the pole-placement controller keeps
the nonlinear plant stable using the TRUE state, and the observer just runs
alongside. That way every observer sees exactly the same trajectory and we're
only comparing how good the estimates are. The plant starts at
x_e + [0.2, 0, 0.05, 0] and the observer starts at x_e, i.e. with no clue.

Figures: fig_m1_full_observers.png, fig_n1_reduced_observer.png,
         fig_o1_noise_traces.png, fig_o2_noise_rms.png
Results: results/part_b_mno.txt
"""
import numpy as np
import matplotlib.pyplot as plt

from model import equilibrium
from design import (designs, make_observers, OBS_POLES, RED_OBS_POLES, LYAP_BETA_OBS,
                    NOISE_STD, ctrb, obsv)
from sim import simulate_closed_loop, estimation_metrics
from plot_utils import savefig, write_result, fmt

DX0 = np.array([0.2, 0.0, 0.05, 0.0])
T_END = 5.0
STATE_LABELS = [r"$r$ [m]", r"$\dot r$ [m/s]", r"$\theta$ [rad]", r"$\dot\theta$ [rad/s]"]


def run_all(noise_std, seed=1):
    d = designs()
    x_e, _ = equilibrium(0.0)
    return {nm: simulate_closed_loop(x_e + DX0, T_END, K=d["K_place"], observer=obs,
                                     feedback="true", noise_std=noise_std, seed=seed)
            for nm, obs in make_observers().items()}


def run(verbose=True):
    d = designs()
    A, B, C = d["A"], d["B"], d["C"]
    n = A.shape[0]
    obs = make_observers()
    out = []

    # ---------------- (m) full-order observers ----------------
    out += ["=" * 70, "(m) FULL-ORDER (LUENBERGER) OBSERVERS", "=" * 70,
            "xhat_dot = A xhat + B u + L (y - C xhat);  error e = x - xhat obeys e_dot = (A - LC) e",
            f"Desired observer poles {OBS_POLES.tolist()}: about 5x faster than the dominant",
            "controller poles (-1.75 ± 1.79j), so the estimate settles before the controller acts.",
            "Duality: an observer for (A, C) is a controller for (A^T, C^T), so L = K_dual^T.", ""]
    P = d["P_obs_lyap"]
    Ao = A.T - C.T @ d["L_lyap"].T
    res_lyap = (Ao + LYAP_BETA_OBS * np.eye(n)) @ P + P @ (Ao + LYAP_BETA_OBS * np.eye(n)).T \
        + C.T @ C
    rows = [("pole placement (y = r, θ)", d["L_place"], C),
            ("Ackermann (y = r only)", d["L_acker"], C[[0]]),
            (f"Lyapunov/Bass β={LYAP_BETA_OBS:g} (y = r, θ)", d["L_lyap"], C)]
    for nm, L, Cu in rows:
        ev = np.sort_complex(np.linalg.eigvals(A - L @ Cu))
        out += [f"  {nm}:", "  L^T =", "  " + fmt(L.T, 3).replace("\n", "\n  "),
                "  eig(A - LC) = " + ", ".join(f"{e:.3f}" for e in ev), ""]
    out += ["Justification:",
            "  * Pole placement uses both outputs. With 2 outputs L is not unique; scipy's",
            "    robust (Kautsky-Nichols/Tits-Yang) algorithm picks the best-conditioned one.",
            "  * Ackermann's formula holds only for a single output, so it uses y = r (observable",
            "    from r alone, part k):  L = phi_d(A) Mo^{-1} [0 0 0 1]^T. The same 4 poles from",
            "    one sensor need much larger gains.",
            f"  * Lyapunov (dual Bass): (A^T + bI)P + P(A^T + bI)^T = C^T C, L = P^-1 C^T.",
            f"    eig(P) = {fmt(np.linalg.eigvalsh(P), 5)} > 0, error-Lyapunov residual = "
            f"{np.max(np.abs(res_lyap)):.1e}",
            f"    -> Re eig(A - LC) < -{LYAP_BETA_OBS:g}: guaranteed decay rate of the error.", ""]

    clean = run_all((0.0, 0.0))
    noisy = run_all(NOISE_STD)

    full = [k for k in obs if k.startswith("Full")]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for nm in full:
        e = clean[nm]["xhat"] - clean[nm]["x"]
        for i in range(4):
            ax.flat[i].plot(clean[nm]["t"], e[:, i], label=nm)
    for i in range(4):
        ax.flat[i].set_xlim(0, 1.5)
        ax.flat[i].set_ylabel("error in " + STATE_LABELS[i])
        ax.flat[i].set_xlabel("time [s]")
    ax.flat[0].legend()
    fig.suptitle(r"Full-order observers: estimation error $\hat{x} - x$ (no noise), nonlinear plant")
    fig.tight_layout()
    savefig(fig, "fig_m1_full_observers.png")

    # ---------------- (n) reduced-order observer ----------------
    red = obs["Reduced-order (y=r)"]
    out += ["=" * 70, "(n) REDUCED-ORDER OBSERVER (only y = r measured)", "=" * 70,
            "Why r? It is the variable being controlled, a ball-position sensor is the usual",
            "measurement on a real rig, and the system is observable from r alone (part k).",
            "Partition x = [x_a | x_b] = [r | r_dot, theta, theta_dot]:",
            f"  Aaa = {A[0,0]:g},  Aab = {fmt(A[0:1,1:])},  Aba^T = {fmt(A[1:,0])},",
            "  Abb =", "  " + fmt(A[1:, 1:]).replace("\n", "\n  "),
            "  (Abb, Aab) observable? rank of its observability matrix = "
            f"{np.linalg.matrix_rank(obsv(A[1:,1:], A[0:1,1:]))} (needs 3)",
            f"Desired error poles {RED_OBS_POLES.tolist()} -> Ke (pole placement) = "
            f"{fmt(d['Ke_red'].T)}^T",
            f"  check with Ackermann: max|Ke_place - Ke_acker| = "
            f"{np.max(np.abs(d['Ke_red'] - d['Ke_red_acker'])):.1e}",
            "  eig(Abb - Ke Aab) = " + ", ".join(f"{e:.3f}" for e in np.linalg.eigvals(red.F)),
            "Observer:  z_dot = (Abb - Ke Aab) z + [(Abb - Ke Aab)Ke + Aba - Ke Aaa] y + (Bb - Ke Ba) u,",
            "           xhat_b = z + Ke y,   r_hat = y  (the measured state is not re-estimated).",
            "Only n - p = 3 states are estimated: fewer integrators, and r is exact (no lag), but",
            "the measured noise goes straight into the estimate through Ke y (see part o).", ""]

    fig, ax = plt.subplots(2, 3, figsize=(14, 6.5))
    rr, fo = clean["Reduced-order (y=r)"], clean["Full: pole placement (y=r,θ)"]
    for j, i in enumerate([1, 2, 3]):
        ax[0, j].plot(rr["t"], rr["x"][:, i], "k", label="true")
        ax[0, j].plot(rr["t"], rr["xhat"][:, i], "--", label="reduced-order")
        ax[0, j].plot(fo["t"], fo["xhat"][:, i], ":", label="full-order (place)")
        ax[1, j].plot(rr["t"], rr["xhat"][:, i] - rr["x"][:, i], label="reduced-order")
        ax[1, j].plot(fo["t"], fo["xhat"][:, i] - fo["x"][:, i], label="full-order (place)")
        ax[0, j].set_ylabel(STATE_LABELS[i])
        ax[1, j].set_ylabel("error in " + STATE_LABELS[i])
        for a in ax[:, j]:
            a.set_xlim(0, 2.0)
            a.set_xlabel("time [s]")
    ax[0, 0].legend()
    ax[1, 0].legend()
    fig.suptitle("Reduced-order observer (y = r only) vs full-order observer, no noise")
    fig.tight_layout()
    savefig(fig, "fig_n1_reduced_observer.png")

    # ---------------- (o) noise study ----------------
    out += ["=" * 70, "(o) OBSERVER PERFORMANCE WITH / WITHOUT MEASUREMENT NOISE", "=" * 70,
            f"Noise: zero-mean Gaussian, sigma_r = {NOISE_STD[0]*1e3:g} mm, sigma_theta = "
            f"{NOISE_STD[1]*1e3:g} mrad, sampled at 1 kHz.",
            "RMS estimation error over t in [1, 5] s (after convergence); t_conv = time at which",
            "the scaled error norm last exceeds 2 % of its peak (no-noise case).", ""]
    hdr = (f"{'observer':32s} {'case':8s} {'t_conv[s]':>9s} " +
           " ".join(f"{s:>12s}" for s in ["rms r", "rms ṙ", "rms θ", "rms θ̇"]))
    out += [hdr, "-" * len(hdr)]
    rms_tab = {}
    for nm in obs:
        for case, res in (("clean", clean[nm]), ("noisy", noisy[nm])):
            m = estimation_metrics(res)
            rms_tab[(nm, case)] = m["rms"]
            tc = f"{m['t_conv']:9.3f}" if case == "clean" else f"{'':9s}"
            out.append(f"{nm:32s} {case:8s} {tc} " + " ".join(f"{v:12.3e}" for v in m["rms"]))
    out += ["",
            "Observations:",
            " * Without noise every observer converges (residual RMS error comes only from the",
            "   plant nonlinearity, since the observers use the linear model). The Lyapunov",
            "   observer is fastest because its poles lie beyond -β = -10.",
            " * With noise there is a speed vs noise trade-off: high-gain observers (Lyapunov,",
            "   Ackermann from a single sensor, reduced-order) pass more noise into the velocity",
            "   estimates (ṙ, θ̇), because L multiplies the measurement noise.",
            " * The reduced-order observer copies the noisy r straight into r̂ (no filtering) and",
            "   adds Ke·y to the other estimates, so it suffers most from r-noise. The full-order",
            "   observers low-pass filter the measurements through their own dynamics.",
            " * Using both sensors (r and θ) gives the best estimates for the same pole locations,",
            "   because redundancy keeps the gains small.",
            " * Even without noise, the single-sensor observers (Ackermann y=r, reduced-order) leave",
            "   a residual θ error about 100x larger than the two-sensor ones (~1e-3 vs ~1e-5 rad),",
            "   and the reduced-order observer has the longest t_conv. With θ unmeasured, the θ",
            "   estimate rests entirely on the LINEAR model, so the plant's nonlinearity",
            "   (sin θ, r θ_dot^2) shows up as a persistent bias. Fast poles do not remove a",
            "   model mismatch.",
            " * Reduced-order noise amplification: the direct feed-through Ke·y with",
            "   |Ke| up to 188 turns 2 mm of r-noise into about 0.38 rad/s of θ̇-noise. Mitigations:",
            "   slower reduced-observer poles, low-pass filtering r, or using the θ encoder."]

    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    for nm in obs:
        e = noisy[nm]["xhat"] - noisy[nm]["x"]
        for i in range(4):
            ax.flat[i].plot(noisy[nm]["t"], e[:, i], lw=0.8, label=nm, alpha=0.85)
    for i in range(4):
        ax.flat[i].set_ylabel("error in " + STATE_LABELS[i])
        ax.flat[i].set_xlabel("time [s]")
        ax.flat[i].set_xlim(0, T_END)
    ax.flat[1].set_ylim(-1, 1)
    ax.flat[3].set_ylim(-3, 3)
    ax.flat[0].legend(fontsize=7)
    fig.suptitle(f"Estimation errors with measurement noise (σ_r = {NOISE_STD[0]*1e3:g} mm, "
                 f"σ_θ = {NOISE_STD[1]*1e3:g} mrad)")
    fig.tight_layout()
    savefig(fig, "fig_o1_noise_traces.png")

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.8))
    names = list(obs)
    xs = np.arange(len(names))
    for i in range(4):
        ax[i].bar(xs - 0.2, [rms_tab[(nm, "clean")][i] for nm in names], 0.4, label="no noise")
        ax[i].bar(xs + 0.2, [rms_tab[(nm, "noisy")][i] for nm in names], 0.4, label="with noise")
        ax[i].set_yscale("log")
        ax[i].set_xticks(xs)
        ax[i].set_xticklabels([nm.replace(" (", "\n(").replace(": ", ":\n") for nm in names],
                              fontsize=6.5)
        ax[i].set_title("RMS error in " + STATE_LABELS[i])
        if i == 0:
            ax[i].annotate("0 (r is measured\ndirectly)", (xs[-1] - 0.2, ax[i].get_ylim()[0]),
                           xytext=(0, 12), textcoords="offset points", ha="center",
                           fontsize=6.5)
    ax[0].legend()
    fig.tight_layout()
    savefig(fig, "fig_o2_noise_rms.png")

    text = "\n".join(out)
    write_result("part_b_mno.txt", text)
    if verbose:
        print(text)


if __name__ == "__main__":
    run()
