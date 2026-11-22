"""
part_a_solution.py -- Part A (f) transfer functions, (g) state transition
matrix and the analytical solution, (h) zero-input and zero-state responses.

The linear model has a nice simple structure at any equilibrium r_e:
    A = [[0, 1, 0, 0], [0, 0, -a, 0], [0, 0, 0, 1], [-c, 0, 0, 0]],  B = [0, 0, 0, beta]^T
    a = m g R^2/(J_b + m R^2) = 5g/7,  c = m g/(J + m r_e^2),  beta = 1/(J + m r_e^2)
    det(sI - A) = s^4 - a c = (s^2 - w^2)(s^2 + w^2),   w = (a c)^(1/4)

Every entry of (sI-A)^{-1} is (polynomial in s, degree <= 3) / (s^4 - w^4),
so we only ever need these four inverse Laplace transforms:
    1/(s^4-w^4)   -> (sinh wt - sin wt)/(2 w^3)
    s/(s^4-w^4)   -> (cosh wt - cos wt)/(2 w^2)
    s^2/(s^4-w^4) -> (sinh wt + sin wt)/(2 w)
    s^3/(s^4-w^4) -> (cosh wt + cos wt)/2
and that gets us Phi(t) = e^{At} in closed form without any heavy lifting.

Figures:  fig_f1_bode.png, fig_g1_stm_check.png, fig_h1_zir_zsr.png
Results:  results/part_a_fgh.txt, results/part_a_fgh_latex.tex
"""
import numpy as np
import sympy as sp
import control as ct
import matplotlib.pyplot as plt
from scipy.linalg import expm

from model import linearize
from sim import simulate_linear_ol
from plot_utils import savefig, write_result, fmt

X0 = np.array([0.02, 0.0, 0.0, 0.0])   # for (h): ball starts 2 cm off centre
U0 = 0.01                              # for (h): size of the torque step [N m]
T_END = 1.5


def symbolic_linear():
    s = sp.symbols("s")
    t, a, c, b, w = sp.symbols("t a c beta omega", positive=True)
    A = sp.Matrix([[0, 1, 0, 0], [0, 0, -a, 0], [0, 0, 0, 1], [-c, 0, 0, 0]])
    B = sp.Matrix([0, 0, 0, b])
    C = sp.Matrix([[1, 0, 0, 0], [0, 0, 1, 0]])
    return s, t, a, c, b, w, A, B, C


def run(verbose=True):
    s, t, a, c, b, w, A_s, B_s, C_s = symbolic_linear()
    A, B, C, D = linearize(0.0)
    vals = {a: -A[1, 2], c: -A[3, 0], b: B[3, 0]}
    vals[w] = (vals[a] * vals[c]) ** 0.25
    out, tex = [], []

    # ---------------- (f) transfer functions ----------------
    sI_A = s * sp.eye(4) - A_s
    den = sp.expand(sI_A.det())
    adj = sI_A.adjugate()
    G = (C_s * adj * B_s).applyfunc(lambda e: sp.factor(e / den))
    out += ["=" * 70, "(f) TRANSFER FUNCTIONS  G(s) = C (sI - A)^{-1} B + D", "=" * 70,
            f"det(sI - A) = {den}",
            f"G_r(s)     = R(s)/T(s)     = {G[0]}",
            f"G_theta(s) = Theta(s)/T(s) = {G[1]}",
            "Matrix form: G(s) = 1/(s^4 - a c) * [ -a*beta ; beta*s^2 ]", ""]
    tex += [r"% (f) transfer function matrix", r"\mathbf G(s) = " + sp.latex(G)]

    sys = ct.ss(A, B, C, D)
    for i, name in enumerate(["r/tau", "theta/tau"]):
        gi = ct.ss2tf(ct.ss(A, B, C[[i]], D[[i]]))
        num = np.trim_zeros(np.round(gi.num[0][0], 10), "f")
        out.append(f"Numeric {name:9s}: num = {fmt(num, 4)}  den = {fmt(gi.den[0][0], 4)}")
        z = np.roots(num) if num.size > 1 else np.array([])
        out.append(f"    zeros = {fmt(z, 4)},  poles = {fmt(np.roots(gi.den[0][0]), 4)}")
    out += ["",
            "Interpretation:",
            " * Both channels share the 4 poles ±w, ±jw (w = %.4f): no cancellation, so the"
            % vals[w],
            "   realization is minimal (also confirmed in part j).",
            " * G_r has no finite zeros and a negative gain: positive torque tilts the beam so",
            "   the ball rolls towards -r. The relative degree is 4: torque -> angle -> ball",
            "   acceleration is a chain of 4 integrators.",
            " * G_theta has a double zero at s = 0: a constant torque cannot hold a steady",
            "   beam angle without the ball moving, so the DC path is blocked.", ""]

    wgrid = np.logspace(-1, 2, 2000)
    resp = np.array([C @ np.linalg.solve(1j * wi * np.eye(4) - A, B) for wi in wgrid])[:, :, 0]
    fig, ax = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
    for i, lab in enumerate([r"$G_r = R/T$", r"$G_\theta = \Theta/T$"]):
        ax[0].semilogx(wgrid, 20 * np.log10(np.abs(resp[:, i])), label=lab)
        ax[1].semilogx(wgrid, np.degrees(np.angle(resp[:, i])), label=lab)
    for axi in ax:
        axi.axvline(vals[w], color="gray", ls=":", lw=0.9)
    ax[0].set_ylabel("magnitude [dB]")
    ax[1].set_ylabel("phase [deg]")
    ax[1].set_xlabel("frequency [rad/s]")
    ax[0].set_title(f"Bode plot of the linearized model (dotted: ω = {vals[w]:.2f} rad/s)")
    ax[0].legend()
    fig.tight_layout()
    savefig(fig, "fig_f1_bode.png")

    # ---------------- (g) state transition matrix ----------------
    basis = [(sp.sinh(w * t) - sp.sin(w * t)) / (2 * w**3),
             (sp.cosh(w * t) - sp.cos(w * t)) / (2 * w**2),
             (sp.sinh(w * t) + sp.sin(w * t)) / (2 * w),
             (sp.cosh(w * t) + sp.cos(w * t)) / 2]

    def inv_laplace(entry):
        poly = sp.Poly(sp.expand(entry), s)
        return sp.expand(sum(poly.coeff_monomial(s**k) * basis[k] for k in range(4)))

    Phi = adj.applyfunc(inv_laplace)
    # in the denominator a and c always come in together as w^4 = a c
    Phi_num = sp.lambdify((t, a, c, b, w), Phi, "numpy")
    ev = lambda tt: np.array(Phi_num(tt, vals[a], vals[c], vals[b], vals[w]), dtype=float)

    t_chk = np.linspace(0, T_END, 31)
    err_stm = max(np.max(np.abs(ev(tt) - expm(A * tt)) / np.maximum(1, np.abs(expm(A * tt))))
                  for tt in t_chk)
    dPhi = Phi.diff(t) - A_s * Phi
    err_ode = max(np.max(np.abs(np.array(
        sp.lambdify((t, a, c, b, w), dPhi, "numpy")(tt, vals[a], vals[c], vals[b], vals[w]),
        dtype=float))) for tt in t_chk)
    out += ["=" * 70, "(g) STATE TRANSITION MATRIX  Phi(t) = e^{At} = L^{-1}{(sI - A)^{-1}}",
            "=" * 70, "with w = (a c)^(1/4):"]
    for i in range(4):
        for j in range(4):
            out.append(f"  Phi[{i+1}{j+1}] = {Phi[i, j]}")
    out += ["",
            f"[check] Phi(0) = I : {np.allclose(ev(0.0), np.eye(4))}",
            f"[check] max relative |Phi(t) - expm(At)| on [0, {T_END}] s = {err_stm:.2e}",
            f"[check] max |dPhi/dt - A Phi| on [0, {T_END}] s = {err_ode:.2e}", "",
            "Analytical solution of the state equation:",
            "  dx(t) = Phi(t) dx(0) + integral_0^t Phi(t - s) B du(s) ds",
            "For a step du(t) = u0:  dx(t) = Phi(t) dx(0) + Gamma(t) B u0,",
            "  Gamma(t) B = integral_0^t Phi(s) B ds  (given below)."]
    tex += [r"% (g) state transition matrix, \omega = (ac)^{1/4}",
            r"\Phi(t) = " + sp.latex(Phi)]

    sig = sp.symbols("sigma", positive=True)
    GamB = (Phi[:, 3] * b).applyfunc(
        lambda e: sp.simplify(sp.integrate(e.subs(t, sig), (sig, 0, t))))
    for i in range(4):
        out.append(f"  (Gamma B)[{i+1}] = {GamB[i]}")
    tex += [r"\Gamma(t)B = \int_0^t \Phi(\sigma)B\,d\sigma = " + sp.latex(GamB)]

    fig, axs = plt.subplots(4, 4, figsize=(11, 8), sharex=True)
    tt = np.linspace(0, T_END, 151)
    Ph = np.array([ev(x) for x in tt])
    Ex = np.array([expm(A * x) for x in tt])
    for i in range(4):
        for j in range(4):
            axs[i, j].plot(tt, Ph[:, i, j], lw=1.8, label="analytic Φ(t)")
            axs[i, j].plot(tt[::10], Ex[::10, i, j], "o", ms=3, label="expm(At)")
            axs[i, j].set_title(fr"$\Phi_{{{i+1}{j+1}}}(t)$", fontsize=9)
            axs[i, j].tick_params(labelsize=7)
    axs[0, 0].legend(fontsize=7)
    for j in range(4):
        axs[3, j].set_xlabel("time [s]")
    fig.suptitle("State transition matrix: closed form vs numerical matrix exponential")
    fig.tight_layout()
    savefig(fig, "fig_g1_stm_check.png")

    # ---------------- (h) zero-input and zero-state responses ----------------
    r0, v0, th0, q0, u0 = sp.symbols("r_0 v_0 theta_0 q_0 u_0", real=True)
    y_zi = (C_s * Phi * sp.Matrix([r0, v0, th0, q0])).applyfunc(sp.simplify)
    y_zs = (C_s * GamB * u0).applyfunc(sp.simplify)
    out += ["", "=" * 70, "(h) ZERO-INPUT AND ZERO-STATE RESPONSES", "=" * 70,
            "Zero-input response  y_zi(t) = C Phi(t) dx(0):",
            f"  r_zi(t)     = {y_zi[0]}", f"  theta_zi(t) = {y_zi[1]}",
            "Zero-state response (step du = u0)  y_zs(t) = C Gamma(t) B u0:",
            f"  r_zs(t)     = {y_zs[0]}", f"  theta_zs(t) = {y_zs[1]}"]
    tex += [r"% (h) zero-input response",
            r"\mathbf y_{zi}(t) = " + sp.latex(y_zi),
            r"% (h) zero-state response to a step \delta\tau = u_0",
            r"\mathbf y_{zs}(t) = " + sp.latex(y_zs)]

    GamB_num = sp.lambdify((t, a, c, b, w), GamB, "numpy")
    gb = lambda x: np.array(GamB_num(x, vals[a], vals[c], vals[b], vals[w]), float).ravel()
    t = np.linspace(0, T_END, 301)
    zir = np.array([C @ ev(x) @ X0 for x in t])
    zsr = np.array([C @ gb(x) * U0 for x in t])
    num = simulate_linear_ol(A, B, X0, lambda _: U0, t) @ C.T
    err = np.max(np.abs(zir + zsr - num))
    out += ["",
            f"Example: dx(0) = {X0.tolist()}, du = {U0} N m step on [0, {T_END}] s",
            f"[check] max |(ZIR + ZSR) - numerical simulation| = {err:.2e}  (superposition holds)",
            "",
            "Interpretation:",
            " * ZIR: the free motion produced by the initial state alone. It contains cosh/sinh",
            "   (from the ±w real poles) and cos/sin (from the ±jw poles). The sinh/cosh part",
            "   grows like e^{wt}, so any small offset of the ball grows without bound.",
            " * ZSR: the forced response from rest. A positive torque step tilts the beam",
            "   (theta > 0), so the ball accelerates towards -r and also diverges.",
            " * Because the system is linear the total response is ZIR + ZSR, which is verified",
            "   numerically above.",
            " * In this example the two parts OPPOSE each other on r: the ZIR pushes the ball",
            "   towards +r (r0 > 0) and the positive torque step pushes it towards -r. The step",
            "   is too small to win (cosh coefficients 0.01 vs -0.0046), so r still diverges.",
            "   It shows that well-timed torque can counteract the drift, which is what",
            "   feedback (part l) exploits."]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, (lab, sc) in enumerate([(r"$\delta r$ [m]", 1), (r"$\delta\theta$ [deg]", 180 / np.pi)]):
        ax[i].plot(t, zir[:, i] * sc, label="zero-input")
        ax[i].plot(t, zsr[:, i] * sc, label="zero-state")
        ax[i].plot(t, (zir + zsr)[:, i] * sc, "k", label="total (analytic)")
        ax[i].plot(t[::15], num[::15, i] * sc, "o", mfc="none", label="numerical sim.")
        ax[i].set_xlabel("time [s]")
        ax[i].set_ylabel(lab)
        ax[i].legend()
    fig.suptitle(fr"Linearized model: $\delta x(0)$ = [{X0[0]} m, 0, 0, 0], "
                 fr"$\delta\tau$ = {U0} N·m step")
    fig.tight_layout()
    savefig(fig, "fig_h1_zir_zsr.png")

    text = "\n".join(out)
    write_result("part_a_fgh.txt", text)
    write_result("part_a_fgh_latex.tex", "\n\n".join(tex))
    if verbose:
        print(text)


if __name__ == "__main__":
    run()
