# Running QCut on LUMI (HPC)

Circuit cutting turns one circuit into a large **ensemble of small, independent
subexperiment circuits** (the count grows combinatorially with the number of cuts).
Running that ensemble is the dominant cost and is embarrassingly parallel, so QCut
ships an **MPI task-farm**: the flat list of experiment circuits is scattered across
ranks, each rank simulates its shard locally, and counts are gathered on rank 0.

This is the right shape for LUMI-C (CPU). It is *not* a distributed-statevector
problem — cutting deliberately keeps each subcircuit small enough for one core.

## What runs where

`QCut.mpi_run(circuit, observables, ...)` is the entry point. Backends are routed by
capability:

| Backend | Strategy |
|---|---|
| `AerSimulator`, IQM fake/noisy backends (`IQMFakeAdonis`, `IQMFakeBackend`) | **farmed** across all ranks (each rank holds its own copy; Aer is re-seeded per rank) |
| Remote QPU (`IQMProvider(url).get_backend()`) | single submitter on rank 0 (the device serializes its own queue) |
| `fiqci.ems.FiQCISampler` | single submitter; the sampler owns batching + mitigation |

## Setup on LUMI

Use the **Cray-provided mpi4py**, not the PyPI wheel — a mismatched MPI silently runs
each rank as an independent size-1 job (works, but zero speedup):

```bash
module load LUMI/24.03 partition/C
module load cray-python          # python + mpi4py built against cray-mpich
# install QCut into a venv on top of cray-python (without the [mpi] extra):
python -m venv --system-site-packages venv && source venv/bin/activate
pip install /path/to/QCut        # mpi4py comes from the module
```

`mpi_run` logs `comm.Get_size()` on rank 0 — confirm it equals your total rank count.
If it prints `1` while you launched many ranks, your mpi4py is linked against the
wrong MPI.

## Run

```bash
sbatch qcut_mpi.sbatch           # edit --account / --nodes first
```

- **Regime A** (default in the sbatch): `--ntasks-per-node=128 --cpus-per-task=1`,
  `OMP_NUM_THREADS=1`. Best for many small circuits.
- **Regime B**: fewer ranks, multi-threaded Aer (`--ntasks-per-node=8
  --cpus-per-task=16`, `OMP_NUM_THREADS=16`). `mpi_run` reads `OMP_NUM_THREADS` and
  sets Aer's `max_parallel_threads` to match, with `max_parallel_experiments=1` to
  avoid OpenMP × MPI oversubscription.

## Reproducibility

Rank `r` uses `seed_simulator = base_seed + r`, so shot noise is decorrelated across
ranks. The shard assignment (LPT bin-packing) is deterministic for a given circuit
list and rank count, so results are reproducible for a fixed `(base_seed, ranks)`.
Different rank counts reshuffle shots and will not reproduce bit-identically.

## Single-node alternative (no MPI)

```python
ck.run(circuit, observables, AerSimulator(), executor="multiprocessing", n_workers=16)
```
