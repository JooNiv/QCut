"""Pluggable execution backends for running QCut experiment circuits.

Circuit cutting turns one circuit into a large *ensemble* of small, independent
subexperiment circuits. Running that ensemble is the dominant cost at scale and is
embarrassingly parallel. This module factors the execution step out of
:func:`QCut.circuit_knitting.run_experiments` behind a small ``CircuitExecutor``
interface so the same flat list of circuits can be run serially, across local
processes, or across MPI ranks on an HPC system such as LUMI.

Run targets fall into three categories, handled differently:

* **Local replicable simulators** -- :class:`qiskit_aer.AerSimulator` and IQM
  fake/noisy backends (``iqm.qiskit_iqm.IQMFakeAdonis`` / ``IQMFakeBackend``, which
  wrap Aer with a noise model). These are picklable and side-effect free, so each
  worker / MPI rank can hold its own copy and run a shard of the circuits in
  parallel -- true compute parallelism.
* **Remote QPU backends** -- e.g. ``IQMProvider(url).get_backend()``. The device
  serializes its own queue, so fanning out across ranks gives no speed-up and would
  open many sessions. These are run from a single submitter.
* **Samplers** -- :class:`fiqci.ems.FiQCISampler`, which performs its own batching
  and error mitigation. Run from a single submitter and let it own the batching.

The executor returns a flat list of ``(key, counts)`` pairs where ``key`` is the
``(group_idx, obs_idx, sub_idx)`` locator; results are reassembled by key on the
root, so gather order is irrelevant.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol, runtime_checkable

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

logger: logging.Logger = logging.getLogger(__name__)

KeyT = tuple[int, int, int]
RunnableT = list[tuple[KeyT, QuantumCircuit]]
ResultT = list[tuple[KeyT, dict[str, int]]]


# --------------------------------------------------------------------------- #
# Backend classification + a thin adapter over the two call shapes we support.
# --------------------------------------------------------------------------- #
def _is_sampler(obj: object) -> bool:
    """True for FiQCISampler-like objects that own their own batching/mitigation."""
    if obj is None:
        return False
    cls = type(obj)
    if cls.__name__ == "FiQCISampler":
        return True
    if (cls.__module__ or "").startswith("fiqci."):
        return True
    # FiQCISampler exposes a ``mitigation_level``; use it as a structural fallback.
    return hasattr(obj, "mitigation_level") and hasattr(obj, "run")


def _is_replicable_simulator(backend: object) -> bool:
    """True if ``backend`` can be safely replicated across processes / ranks.

    Local statevector simulators (Aer and IQM fake/noisy backends) qualify. Remote
    QPU backends and samplers do not.
    """
    if backend is None:
        return True  # default AerSimulator
    if _is_sampler(backend):
        return False
    if isinstance(backend, AerSimulator):
        return True
    cls = type(backend)
    module = (cls.__module__ or "").lower()
    # IQM fake backends live under iqm.qiskit_iqm.fake_backends and wrap Aer.
    if module.startswith("iqm.qiskit_iqm.fake_backends"):
        return True
    if "fake" in module or "Fake" in cls.__name__:
        return True
    return False


def _batched_backend_counts(
    backend, circuits: list[QuantumCircuit], shots: int, max_batch_size: int
) -> list[dict[str, int]]:
    """Run ``circuits`` on a Qiskit ``backend`` in batches; return ordered counts."""
    out: list[dict[str, int]] = []
    for start in range(0, len(circuits), max_batch_size):
        batch = circuits[start : start + max_batch_size]
        counts = backend.run(batch, shots=shots).result().get_counts()
        if isinstance(counts, dict):
            counts = [counts]
        out.extend(dict(c) for c in counts)
    return out


class BackendAdapter:
    """Normalize ``backend.run`` and ``sampler.run`` into a single counts call."""

    def __init__(self, backend=None):
        self.backend = backend if backend is not None else AerSimulator()
        self.is_sampler = _is_sampler(self.backend)
        self.replicable = _is_replicable_simulator(self.backend)

    def run_counts(
        self, circuits: list[QuantumCircuit], shots: int, max_batch_size: int
    ) -> list[dict[str, int]]:
        if not circuits:
            return []
        if self.is_sampler:
            # The sampler owns batching + mitigation; hand it the whole list.
            counts = (
                self.backend.run(
                    circuits, shots=shots, max_batch_size=max_batch_size
                )
                .result()
                .get_counts()
            )
            if isinstance(counts, dict):
                counts = [counts]
            return [dict(c) for c in counts]
        return _batched_backend_counts(self.backend, circuits, shots, max_batch_size)


# --------------------------------------------------------------------------- #
# Load balancing helpers (shared by the parallel executors).
# --------------------------------------------------------------------------- #
def _cost(circ: QuantumCircuit) -> int:
    """Cheap proxy for simulation effort: ~ statevector size * depth."""
    try:
        return (1 << circ.num_qubits) * max(1, circ.depth())
    except Exception:  # pragma: no cover - defensive; metadata should always exist
        return 1


def _partition_lpt(runnable: RunnableT, nbins: int) -> list[RunnableT]:
    """Longest-processing-time greedy bin packing for balanced makespan.

    Deterministic given ``runnable`` (so results are reproducible for a fixed
    rank count), O(n log n), and needs zero runtime communication.
    """
    bins: list[RunnableT] = [[] for _ in range(nbins)]
    load = [0] * nbins
    order = sorted(
        range(len(runnable)), key=lambda i: _cost(runnable[i][1]), reverse=True
    )
    for i in order:
        b = min(range(nbins), key=lambda b: load[b])
        bins[b].append(runnable[i])
        load[b] += _cost(runnable[i][1])
    return bins


# --------------------------------------------------------------------------- #
# Executor interface + implementations.
# --------------------------------------------------------------------------- #
@runtime_checkable
class CircuitExecutor(Protocol):
    """Runs a flat list of keyed circuits and returns keyed counts."""

    def run(
        self, runnable: RunnableT, shots: int, max_batch_size: int
    ) -> ResultT: ...


class SerialExecutor:
    """Single-process, single-submitter execution (the original behavior).

    Used for the default path, and for remote QPU backends / samplers that must
    not be replicated across workers.
    """

    is_worker = False

    def __init__(self, backend=None):
        self.adapter = BackendAdapter(backend)

    def run(self, runnable: RunnableT, shots: int, max_batch_size: int) -> ResultT:
        if not runnable:
            return []
        circuits = [c for _, c in runnable]
        counts = self.adapter.run_counts(circuits, shots, max_batch_size)
        return [(key, c) for (key, _), c in zip(runnable, counts)]


# Per-worker global so the (possibly heavy) backend is pickled once per process.
_WORKER_ADAPTER: BackendAdapter | None = None


def _mp_init(backend) -> None:
    global _WORKER_ADAPTER
    _WORKER_ADAPTER = BackendAdapter(backend)


def _mp_run_chunk(args):
    chunk, shots, max_batch_size = args
    assert _WORKER_ADAPTER is not None
    circuits = [c for _, c in chunk]
    counts = _WORKER_ADAPTER.run_counts(circuits, shots, max_batch_size)
    return [(key, c) for (key, _), c in zip(chunk, counts)]


class MultiprocessingExecutor:
    """Fan circuits out across local processes (single node).

    Falls back to serial execution when the backend cannot be replicated.
    """

    is_worker = False

    def __init__(self, backend=None, n_workers: int | None = None):
        self.backend = backend
        self.adapter = BackendAdapter(backend)
        self.n_workers = n_workers or os.cpu_count() or 1

    def run(self, runnable: RunnableT, shots: int, max_batch_size: int) -> ResultT:
        if not runnable:
            return []
        if not self.adapter.replicable:
            logger.warning(
                "Backend %s is not replicable; running serially instead of "
                "multiprocessing.",
                type(self.adapter.backend).__name__,
            )
            return SerialExecutor(self.backend).run(runnable, shots, max_batch_size)

        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        n = max(1, min(self.n_workers, len(runnable)))
        chunks = [c for c in _partition_lpt(runnable, n) if c]
        # Use 'spawn' rather than the Linux default 'fork': Aer/BLAS spawn threads,
        # and forking a multi-threaded process deadlocks the workers.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n,
            mp_context=ctx,
            initializer=_mp_init,
            initargs=(self.backend,),
        ) as pool:
            parts = pool.map(
                _mp_run_chunk, [(chunk, shots, max_batch_size) for chunk in chunks]
            )
        return [pair for part in parts for pair in part]


class MPIExecutor:
    """MPI task-farm: scatter the circuit list across ranks, gather counts on rank 0.

    A *collective* -- every rank must call :meth:`run` the same number of times.
    Rank 0 builds the balanced shards; workers receive theirs via ``scatter`` and
    contribute their counts via ``gather``. Use the :func:`QCut.mpi_run` entry
    point, which performs the rank-0/worker split for the full pipeline.

    Only use this with replicable simulators (Aer / IQM fake backends). For remote
    backends/samplers, :func:`QCut.mpi_run` routes to a single-submitter path.
    """

    def __init__(
        self,
        backend=None,
        comm=None,
        threads: int | None = None,
        base_seed: int = 1234,
        transpile_fn=None,
    ):
        from mpi4py import MPI

        self.comm = comm if comm is not None else MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.is_worker = self.rank != 0
        self.backend = backend
        self.threads = threads
        self.base_seed = base_seed
        self.transpile_fn = transpile_fn
        if self.rank == 0:
            logger.info("MPIExecutor initialized with %d ranks.", self.size)

    def _local_backend(self):
        """Per-rank backend. Aer is re-seeded per rank to decorrelate shot noise."""
        b = self.backend
        if b is None or isinstance(b, AerSimulator):
            return AerSimulator(
                max_parallel_threads=self.threads or 0,
                max_parallel_experiments=1,
                seed_simulator=self.base_seed + self.rank,
            )
        return b  # IQM fake backend (already replicated on each rank's process)

    def run(
        self, runnable: RunnableT | None, shots: int, max_batch_size: int
    ) -> ResultT:
        comm = self.comm
        shards = _partition_lpt(runnable or [], self.size) if self.rank == 0 else None
        my_shard: RunnableT = comm.scatter(shards, root=0)

        adapter = BackendAdapter(self._local_backend())
        circuits = [c for _, c in my_shard]
        if self.transpile_fn is not None:
            circuits = [self.transpile_fn(c) for c in circuits]
        counts = adapter.run_counts(circuits, shots, max_batch_size)
        local = [(key, c) for (key, _), c in zip(my_shard, counts)]

        gathered = comm.gather(local, root=0)
        if self.rank == 0 and gathered is not None:
            return [pair for part in gathered for pair in part]
        return []


# --------------------------------------------------------------------------- #
# Selection / detection.
# --------------------------------------------------------------------------- #
def _mpi_active() -> bool:
    """Best-effort detection of a multi-rank MPI launch without importing mpi4py."""
    import importlib.util

    if importlib.util.find_spec("mpi4py") is None:
        return False
    for var in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "PMIX_RANK"):
        val = os.environ.get(var)
        if val:
            try:
                if int(val.split(",")[0]) > 1:
                    return True
            except ValueError:
                continue
    return False


def get_default_executor(
    backend=None, executor: str | CircuitExecutor = "auto", n_workers: int | None = None
) -> CircuitExecutor:
    """Resolve the ``executor`` argument to a concrete :class:`CircuitExecutor`.

    Accepts an executor instance (returned as-is), or one of the strings
    ``"serial"``, ``"multiprocessing"``, ``"mpi"``, ``"auto"``. ``"auto"`` keeps
    things safe: it never silently fans a non-replicable backend out, and never
    auto-enables MPI for the plain ``run`` path (MPI requires the rank-aware
    :func:`QCut.mpi_run` entry point). Multiprocessing is opt-in.
    """
    if not isinstance(executor, str):
        if isinstance(executor, CircuitExecutor):
            return executor
        raise TypeError(
            "executor must be a CircuitExecutor or one of "
            "'auto'/'serial'/'multiprocessing'/'mpi'."
        )
    if executor == "serial":
        return SerialExecutor(backend)
    if executor == "multiprocessing":
        return MultiprocessingExecutor(backend, n_workers)
    if executor == "mpi":
        return MPIExecutor(backend)
    if executor != "auto":
        raise ValueError(f"Unknown executor {executor!r}.")

    # auto
    if not BackendAdapter(backend).replicable:
        return SerialExecutor(backend)
    return SerialExecutor(backend)
