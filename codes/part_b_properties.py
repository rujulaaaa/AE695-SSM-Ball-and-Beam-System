"""
part_b_properties.py -- Part B (i) canonical forms, (j) controllability and
observability (rank tests, indices, PBH, Gramians), (k) what happens when we
lose inputs/outputs.

Results: results/part_b_ijk.txt
"""
import numpy as np
from numpy.linalg import inv, matrix_rank, svd
from scipy.linalg import expm
from scipy.integrate import quad_vec

from model import linearize
from design import ctrb, obsv
from plot_utils import write_result, fmt


def W_matrix(char_coeffs):
    """The W matrix from Ogata, built from det(sI-A) = s^n + a1 s^{n-1} + ... + an.
    (Needed for the controllable/observable canonical transforms.)"""
    a = np.asarray(char_coeffs[1:], float)
    n = a.size
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n - i):
            W[i, j] = 1.0 if i + j == n - 1 else a[n - 2 - i - j]
    return W


def tf_eval(A, B, C, s):
    return C @ np.linalg.solve(s * np.eye(A.shape[0]) - A, B)


def same_tf(A1, B1, C1, A2, B2, C2):
    pts = [0.5j, 1 + 2j, 10.0, -2 + 0.3j]
    return max(np.max(np.abs(tf_eval(A1, B1, C1, p) - tf_eval(A2, B2, C2, p))) for p in pts)


def index_sequence(A, M, kind="ctrb"):
    """Walk through B, AB, A^2B, ... (or C, CA, ...) and keep a column (row) only
    if it adds something new, i.e. it's independent of what we've kept so far.
    Gives back the overall index and the Luenberger index for each channel."""
    n = A.shape[0]
    chans = M.shape[1] if kind == "ctrb" else M.shape[0]
    kept, per = [], [0] * chans
    blocked = [False] * chans
    for k in range(n):
        Ak = np.linalg.matrix_power(A, k)
        for ch in range(chans):
            if blocked[ch]:
                continue
            v = (Ak @ M[:, ch]) if kind == "ctrb" else (M[ch, :] @ Ak)
            if matrix_rank(np.array(kept + [v])) > len(kept):
                kept.append(v)
                per[ch] += 1
            else:
                blocked[ch] = True
    overall = next((k for k in range(1, n + 1)
                    if matrix_rank(ctrb(A, M)[:, :k * M.shape[1]] if kind == "ctrb"
                                   else obsv(A, M)[:k * M.shape[0], :]) == n), None)
    return overall, per, len(kept)


def gramian(A, M, T, kind="ctrb"):
    """Finite-horizon Gramian using Van Loan's trick (one big block matrix
    exponential instead of integrating numerically)."""
    n = A.shape[0]
    if kind == "ctrb":
        Q, Ablk = M @ M.T, A
    else:
        Q, Ablk = M.T @ M, A.T
    Z = np.block([[-Ablk, Q], [np.zeros((n, n)), Ablk.T]])
    E = expm(Z * T)
    return E[n:, n:].T @ E[:n, n:]


def run(verbose=True):
    A, B, C, D = linearize(0.0)
    n = A.shape[0]
    out = []
    char = np.real(np.poly(A))
    W = W_matrix(char)

    # ---------------- (i) canonical forms ----------------
    out += ["=" * 70, "(i) CANONICAL REALIZATIONS  (all equivalent to (A, B, C, D))", "=" * 70,
            "Characteristic polynomial coefficients [1 a1 a2 a3 a4] = " + fmt(char, 4), ""]

    Tc = ctrb(A, B) @ W                                   # x = Tc z
    Ac, Bc, Cc = inv(Tc) @ A @ Tc, inv(Tc) @ B, C @ Tc
    out += ["Controllable canonical form (x = Tc z, Tc = Mc W):",
            "Tc =", fmt(Tc), "Ac =", fmt(Ac), "Bc =", fmt(Bc.T) + "^T", "Cc =", fmt(Cc),
            "  (row 1 of Cc = numerator of G_r = -a beta; row 2 = numerator of G_theta = beta s^2)",
            f"[check] TF difference = {same_tf(A, B, C, Ac, Bc, Cc):.2e}", ""]

    for i, nm in enumerate(["r", "theta"]):
        Q = inv(W @ obsv(A, C[[i]]))                      # x = Q z (one output at a time)
        Ao, Bo, Co = inv(Q) @ A @ Q, inv(Q) @ B, C @ Q
        out += [f"Observable canonical form w.r.t. output y{i+1} = {nm}  (x = Q z, Q = (W O)^-1):",
                "Ao =", fmt(Ao), "Bo =", fmt(Bo.T) + "^T", "Co =", fmt(Co),
                f"[check] TF difference = {same_tf(A, B, C, Ao, Bo, Co):.2e}", ""]
    out += ["(The system has 2 outputs, so a single observable companion form can be built",
            " for one output at a time. Both are given; the system is observable from either.)", ""]

    lam, V = np.linalg.eig(A)
    order = np.argsort(-lam.real)
    lam, V = lam[order], V[:, order]
    Aj, Bj, Cj = inv(V) @ A @ V, inv(V) @ B, C @ V
    out += ["Jordan (modal) canonical form: the eigenvalues are distinct, so every Jordan block",
            "is 1x1 and the Jordan form is diagonal (complex):",
            "J = diag(" + ", ".join(f"{l:.4f}" for l in np.diag(Aj)) + ")",
            "Bj =", fmt(Bj.T, 4) + "^T", "Cj =", fmt(Cj, 4),
            f"[check] off-diagonal max = {np.max(np.abs(Aj - np.diag(np.diag(Aj)))):.2e}, "
            f"TF difference = {same_tf(A, B, C, Aj, Bj, Cj):.2e}",
            "  All entries of Bj are nonzero -> every mode is controllable (Gilbert test);",
            "  no column of Cj is zero -> every mode is observable.", ""]

    Tm = np.column_stack([V[:, 0].real, V[:, 3].real, V[:, 1].real, V[:, 1].imag])
    Am, Bm, Cm = inv(Tm) @ A @ Tm, inv(Tm) @ B, C @ Tm
    out += ["Real modal form (real Jordan form), T = [v(+w), v(-w), Re v(jw), Im v(jw)]:",
            "Am =", fmt(Am), "Bm =", fmt(Bm.T) + "^T", "Cm =", fmt(Cm),
            f"[check] TF difference = {same_tf(A, B, C, Am, Bm, Cm):.2e}", ""]

    # ---------------- (j) controllability / observability ----------------
    Mc, Mo = ctrb(A, B), obsv(A, C)
    mu, mu_per, _ = index_sequence(A, B, "ctrb")
    nu, nu_per, _ = index_sequence(A, C, "obsv")
    out += ["=" * 70, "(j) CONTROLLABILITY AND OBSERVABILITY", "=" * 70,
            "Controllability matrix Mc = [B AB A^2B A^3B] =", fmt(Mc),
            f"  rank = {matrix_rank(Mc)} (n = {n}),  singular values = {fmt(svd(Mc)[1], 3)}",
            f"  => CONTROLLABLE. Controllability index mu = {mu} (single input, so mu = n).", "",
            "Observability matrix Mo = [C; CA; CA^2; CA^3] (8 x 4) =", fmt(Mo),
            f"  rank = {matrix_rank(Mo)},  singular values = {fmt(svd(Mo)[1], 3)}",
            f"  => OBSERVABLE. Observability index nu = {nu}; "
            f"Luenberger indices (r, theta) = {tuple(nu_per)}",
            "  (C and CA already give r, theta, r_dot, theta_dot, i.e. the whole state.)", "",
            "PBH (Popov-Belevitch-Hautus) test at each eigenvalue:"]
    for l in lam:
        rc = matrix_rank(np.hstack([l * np.eye(n) - A, B]))
        ro = matrix_rank(np.vstack([l * np.eye(n) - A, C]))
        out.append(f"  lambda = {l:+.4f}: rank[lI-A, B] = {rc}, rank[lI-A; C] = {ro}")
    T = 1.0
    Wc, Wo = gramian(A, B, T, "ctrb"), gramian(A, C, T, "obsv")
    Wc_q = quad_vec(lambda s: expm(A * s) @ B @ B.T @ expm(A.T * s), 0, T)[0]
    out += ["", f"Finite-horizon Gramians on [0, {T}] s (A is unstable, so the infinite-horizon",
            "Gramian does not exist):",
            f"  Wc eigenvalues = {fmt(np.linalg.eigvalsh(Wc), 4)} (all > 0 -> controllable)",
            f"  Wo eigenvalues = {fmt(np.linalg.eigvalsh(Wo), 4)} (all > 0 -> observable)",
            f"  [check] Van Loan vs quadrature: {np.max(np.abs(Wc - Wc_q)):.2e}",
            "  The spread of the Wc eigenvalues shows some directions need far more control energy.", ""]

    # ---------------- (k) partial controllability / observability ----------------
    out += ["=" * 70, "(k) PARTIAL CONTROLLABILITY AND OBSERVABILITY", "=" * 70,
            "Inputs: the system has ONE input (beam torque). Removing it leaves B = 0, so",
            "rank Mc = 0 and no state is controllable. With a single actuator the 'absence of",
            "selected inputs' study is trivial; the only nontrivial input set is {tau}, which is",
            "fully controllable (part j).", "",
            "Outputs: observability with subsets of outputs (and, for interest, with any",
            "single state as the only measurement):"]
    tests = {"y = r only (C row 1)": C[[0]], "y = theta only (C row 2)": C[[1]],
             "y = r_dot only": np.array([[0, 1, 0, 0.]]),
             "y = theta_dot only": np.array([[0, 0, 0, 1.]]),
             "y = [r, theta] (both)": C, "no output": np.zeros((1, 4))}
    for nm, Ci in tests.items():
        O = obsv(A, Ci)
        rk = matrix_rank(O)
        idx = index_sequence(A, Ci, "obsv")[0] if rk == n else "-"
        out.append(f"  {nm:26s}: rank = {rk}, observable = {rk == n}, index = {idx}, "
                   f"cond(O) = {np.linalg.cond(O):.2e}" if rk == n else
                   f"  {nm:26s}: rank = {rk}, observable = False")
    out += ["",
            "Interpretation: the states form a coupled chain r -> r_dot -> theta -> theta_dot -> r",
            "(r_ddot = -a theta, theta_ddot = -c r + beta tau). Any single state therefore reaches",
            "the whole chain through differentiation, so ANY single measurement keeps the system",
            "observable. Losing one sensor removes redundancy (the index rises from 2 to 4) and",
            "worsens conditioning, but not observability. This justifies the reduced-order",
            "observer of part (n), which uses only y = r.",
            "",
            "Note: c = m g/(J + m r_e^2) and a = 5g/7 are nonzero at every equilibrium, so these",
            "conclusions hold for all operating points (they would fail only if g = 0)."]

    text = "\n".join(out)
    write_result("part_b_ijk.txt", text)
    if verbose:
        print(text)


if __name__ == "__main__":
    run()
