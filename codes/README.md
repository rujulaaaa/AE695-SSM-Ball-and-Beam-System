# AE 695 Course Project: Ball and Beam (code)

Running `python main.py` regenerates every figure (`figures/`) and every numerical result
(`results/`) in about 45 s. To run individual parts: `python main.py a_model b_control`.

Setup (Python ≥ 3.10):
```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python main.py
```

| File | Content | Problem part |
|---|---|---|
| `params.py` | physical parameters (assumed values, with justification) | — |
| `model.py` | Lagrangian derivation (SymPy), nonlinear f(x,u), h(x), equilibria, Jacobians | a, b, c |
| `sim.py` | ODE simulation, RK4 closed-loop simulator with noise, observers, metrics | — |
| `design.py` | all controller/observer gains (place, Ackermann, Lyapunov/Bass, LQR, reduced) | l, m, n |
| `plot_utils.py` | figure style, saving helpers | — |
| `part_a_model.py` | state-space form, equilibria, linearization + checks | a–c |
| `part_a_openloop.py` | nonlinear vs linear open loop; eigen-analysis | d, e |
| `part_a_solution.py` | transfer functions, closed-form e^{At}, ZIR/ZSR | f, g, h |
| `part_b_properties.py` | canonical forms, controllability/observability, PBH, Gramians, partial | i, j, k |
| `part_b_control.py` | state feedback design and nonlinear validation | l |
| `part_b_observers.py` | full-order / reduced-order observers, noise study | m, n, o |
| `part_b_combined.py` | observer-based control, separation principle | p |

State x = [r, ṙ, θ, θ̇], input u = beam torque τ, output y = [r, θ]. Everything is
linearized at r_e = 0 unless stated otherwise.
