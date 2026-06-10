"""Example QCut MPI driver for LUMI-C.

Every MPI rank runs this script. ``QCut.mpi_run`` performs the rank-0/worker split:
rank 0 builds the experiment circuits and drives the pipeline; worker ranks only
help execute the scattered circuit shards. Rank 0 returns the expectation values,
workers return ``None``.

Launch via ``srun python -u run_qcut_job.py`` (see qcut_mpi.sbatch).
"""

from mpi4py import MPI
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate
from qiskit.quantum_info import SparsePauliOp

import QCut as ck
from QCut import cut, cutGate

comm = MPI.COMM_WORLD

if comm.Get_rank() == 0:
    print(f"QCut MPI job: {comm.Get_size()} ranks", flush=True)

# --- Build the circuit with cut markers (cheap; fine to do on every rank). ----
mult = 1.635
circuit = QuantumCircuit(4)
circuit.r(mult * 0.46262, mult * 0.1446, 0)
circuit.append(**cutGate(CXGate(), 0, 1))
circuit.append(cut(), [1])
circuit.cx(1, 2)
circuit.cx(2, 3)

observables = SparsePauliOp(["IIIZ", "IIZI", "IZII", "IIZZ"])

# --- Run the full pipeline across all ranks (default per-rank AerSimulator). ---
# For a noisy run, build an IQM fake backend on every rank and pass backend=...
expectation_values = ck.mpi_run(circuit, observables, shots=2**12)

if comm.Get_rank() == 0:
    print("Expectation values:", expectation_values, flush=True)
