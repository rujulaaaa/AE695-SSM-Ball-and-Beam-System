"""
main.py -- runs the whole AE 695 ball-and-beam project in one go.

From this folder:
    python main.py            # run everything
    python main.py a_model    # or just the step(s) you name

Plots end up in ./figures and the text results/tables in ./results.
"""
import importlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)

# (step name, module, what it does) -- every module has a run() we can call
STEPS = [
    ("a_model",        "part_a_model",        "(a)-(c) state space, equilibria, linearization"),
    ("a_openloop",   "part_a_openloop",     "(d)-(e) open-loop sims, eigen-analysis"),
    ("a_solution",   "part_a_solution",     "(f)-(h) transfer functions, STM, ZIR/ZSR"),
    ("b_properties", "part_b_properties",   "(i)-(k) canonical forms, ctrb/obsv"),
    ("b_control",    "part_b_control",      "(l) state feedback: place / Ackermann / Lyapunov"),
    ("b_observers",  "part_b_observers",    "(m)-(o) full & reduced observers, noise study"),
    ("b_combined",   "part_b_combined",     "(p) controller + observer"),
]


def main(selected=None):
    for name, module, desc in STEPS:
        if selected and name not in selected:
            continue
        print(f"\n{'#' * 72}\n# {name}: {desc}\n{'#' * 72}")
        t0 = time.time()
        importlib.import_module(module).run()
        print(f"--> {name} finished in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
