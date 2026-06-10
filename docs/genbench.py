"""Microbenchmark for the experiment-circuit deep-copy hotspot.

Compares the old pickle round-trip against ``QuantumCircuit.copy()`` (the landed
optimization) and reports the speed-up. Run with::

    uv run python docs/genbench.py

This is the Python baseline for the Qiskit C-API feasibility spike; see
``docs/qiskit_capi_feasibility.md``.
"""

import pickle
import time

from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate

import QCut as ck
from QCut import cut, cutGate

N = 20_000


def _base_subcircuit() -> QuantumCircuit:
    mult = 1.635
    cc = QuantumCircuit(4)
    cc.r(mult * 0.46262, mult * 0.1446, 0)
    cc.append(**cutGate(CXGate(), 0, 1))
    cc.append(cut(), [1])
    cc.cx(1, 2)
    cc.cx(2, 3)
    return ck.get_locations_and_subcircuits(cc).subcircuits[0]


def main() -> None:
    sub = _base_subcircuit()

    t = time.perf_counter()
    for _ in range(N):
        pickle.loads(pickle.dumps(sub))
    t_pickle = time.perf_counter() - t

    t = time.perf_counter()
    for _ in range(N):
        sub.copy()
    t_copy = time.perf_counter() - t

    print(f"deep-copies: {N}")
    print(f"pickle round-trip : {t_pickle:.3f} s")
    print(f"QuantumCircuit.copy: {t_copy:.3f} s")
    print(f"speed-up           : x{t_pickle / t_copy:.1f}")


if __name__ == "__main__":
    main()
