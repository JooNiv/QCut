"""
A module for the main circuit knitting workflow.
"""

from __future__ import annotations

import logging
import os

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import (
    Qubit,
)
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

from QCut.backend_utility import transpile_subcircuits
from QCut.basis_transform import (
    _combine_pauli_ops,
    _get_obs_subcircuits,
)
from QCut.circuit_preparation import get_locations_and_subcircuits
from QCut.circuit_utils import _remove_obsm, _remove_obsm_2
from QCut.cutcircuit import CutCircuit, CutExperiment
from QCut.cutlocation import CutLocation
from QCut.postprocess import ERROR, estimate_expectation_values
from QCut.qcutresult import RawResult
from QCut.qpd_operations import (
    QPD_REGISTRY,
    _insert_2qubit_gate_cut_qpd,
    _insert_wire_cut_qpd,
    get_qpd_combinations,
)

logger: logging.Logger = logging.getLogger(__name__)


def _finalize_subcircuit(
    subcircuit: QuantumCircuit, qpd_qubits: list[int]
) -> QuantumCircuit:
    """Finalize the subcircuit by measuring remaining qubits and decomposing."""

    meas_qubits = [i for i in range(subcircuit.num_qubits) if i not in qpd_qubits]

    dag = circuit_to_dag(subcircuit)
    idle = list(dag.idle_wires())

    creg_to_use = (
        subcircuit.cregs[1] if len(subcircuit.cregs) >= 2 else subcircuit.cregs[0]
    )

    for wire in idle:
        if (
            isinstance(wire, Qubit)
            and wire._index in meas_qubits
            and len(meas_qubits) > len(creg_to_use)
        ):
            meas_qubits.remove(wire._index)

    if len(meas_qubits) == 0:
        return subcircuit

    subcircuit.measure(meas_qubits, creg_to_use)

    return subcircuit


def _has_measurements(circuit: QuantumCircuit) -> bool:
    return "measure" in circuit.count_ops()


def _get_placeholder_locations(subcircuits: list[QuantumCircuit]) -> list:
    """
    Identify the locations of placeholder operations in a list of quantum subcircuits.
    This function scans through each quantum subcircuit provided in the input list and
    identifies the indices and operations where either measurement ("Meas") or
    initialization ("Init") operations occur. It returns a list of lists, where each
    sublist corresponds to  a subcircuit and contains tuples of the
    form (index, operation).

    Args:
        subcircuits (list[QuantumCircuit]):
            A list of QuantumCircuit objects to be analyzed.
    Returns:
        list:
            A list of lists, where each sublist contains tuples (index, operation)
            indicating the positions of measurement or initialization operations in
            the corresponding subcircuit.

    """
    ops = []
    for circ in subcircuits:
        subops = []
        for ind, op in enumerate(circ.data):
            name = op.operation.name
            if (
                name.startswith("Meas")
                or name.startswith("Init")
                or (name.startswith("cut") and "_" in name)
            ):
                subops.append((ind, op))
        ops.append(subops)

    return ops


def get_experiment_circuits(  # noqa: C901
    cut_circuit: CutCircuit,
    observables: SparsePauliOp,
) -> CutExperiment:
    """Generate experiment circuits by inserting QPD operations on
    measure/initialize/cutCZ nodes.

    Args:
        subcircuits (list[QuantumCircuit]): subcircuits with measure/initialize nodes.
        cut_locations (np.ndarray[CutLocation]): cut locations.

    Returns:
        tuple: A tuple containing:
            - CutCircuit: A CutCircuit object containing the experiment circuits.
            - list[int]: A list of coefficients for each circuit.
            - list[tuple[int, int, int]]:
                A list of index pointers to results that need additional post-processing
                due to identity basis measurement.

    """

    num_qubits = 0
    for subcircuit in cut_circuit.subcircuits:
        crs = subcircuit.cregs
        for cr in crs:
            if cr.name == "meas":
                num_qubits += cr.size

    if all(len(obs) != num_qubits for obs in observables.paulis):
        raise ValueError(
            f"""ALL observable lengths must match 
            the number of qubits in the original uncut circuit 
            ({num_qubits})."""
        )

    qpd_combinations = get_qpd_combinations(cut_circuit.cut_locations)
    # generate the QPD
    # operation combinations

    check_circuit_type = cut_circuit.backend is not None

    measurement_settings = _combine_pauli_ops(observables)

    if len(measurement_settings) > 1:
        logger.info(
            f"Found {len(measurement_settings)} conflicting observables. Extra"
            f" circuits will be generated to evaluate all expectation values."
        )

    backend = None
    if check_circuit_type:
        backend = cut_circuit.backend
        try:
            basis = backend.configuration().basis_gates  # type: ignore[possibly-missing-attribute]
        except Exception:
            basis = list(backend.architecture.gates.keys())  # type: ignore[possibly-missing-attribute]
        basis = ["r" if gate == "prx" else gate for gate in basis]

    obs_subcircuits = None

    if check_circuit_type:
        x_meas_ops = QuantumCircuit(1)
        x_meas_ops.h(0)
        x_meas_ops.name = "X-meas"
        x_meas_ops = transpile(x_meas_ops, basis_gates=basis)
        x_meas_ops = x_meas_ops.to_instruction()

        y_meas_ops = QuantumCircuit(1)
        y_meas_ops.sdg(0)
        y_meas_ops.h(0)
        y_meas_ops.name = "Y-meas"
        y_meas_ops = transpile(y_meas_ops, basis_gates=basis)
        y_meas_ops = y_meas_ops.to_instruction()

        ops = {"X-meas": x_meas_ops, "Y-meas": y_meas_ops}

        obs_subcircuits = _get_obs_subcircuits(
            cut_circuit.subcircuits, measurement_settings, ops
        )
    else:
        obs_subcircuits = _get_obs_subcircuits(
            cut_circuit.subcircuits, measurement_settings
        )

    _remove_obsm(obs_subcircuits)

    _remove_obsm_2(cut_circuit.subcircuits)

    # initialize solution lists
    num_circs = 1
    for cut_loc in cut_circuit.cut_locations:
        if isinstance(cut_loc, CutLocation):
            num_circs *= len(QPD_REGISTRY[cut_loc.gate_name])
        else:
            num_circs *= 8
    experiment_circuits = []
    coefficients = np.empty(num_circs)
    placeholder_locations = _get_placeholder_locations(cut_circuit.subcircuits)
    for id_meas_experiment_index, qpd in enumerate(
        qpd_combinations
    ):  # loop through all
        # QPD combinations
        coefficients[id_meas_experiment_index] = np.prod([op["c"] for op in qpd])

        if check_circuit_type:
            for sub in qpd:
                sub["op_0"] = transpile(sub["op_0"], basis_gates=basis)
                sub["op_1"] = transpile(sub["op_1"], basis_gates=basis)

        # circuits
        inserted_operations = 0
        obs_set_circuits = []
        for obs_set in obs_subcircuits:
            cur_set_circuits = {}
            for id_meas_subcircuit_index, circ in obs_set.items():
                # QuantumCircuit.copy() deep-copies the instruction data and is
                # markedly faster than a pickle round-trip; this runs once per
                # (QPD combination x observable set x subcircuit), i.e. the
                # innermost hot loop of experiment generation.
                subcircuit = circ.copy()
                offset = 0
                classical_bit_index = 0
                qpd_qubits = []  # store the qubit indices of qubits used for qpd
                # measurements
                for op_ind in placeholder_locations[id_meas_subcircuit_index]:
                    ind, op = op_ind

                    actual_op = subcircuit.data[ind + offset]

                    if actual_op.operation.name != op.operation.name:
                        cur_ind = ind + offset
                        for i in range(len(subcircuit.data)):
                            cur_ind_minus = cur_ind - i
                            cur_ind_plus = cur_ind + i
                            if (
                                op.operation.name
                                == subcircuit.data[cur_ind_minus].operation.name
                            ):
                                ind = cur_ind_minus - offset
                                break
                            if (
                                op.operation.name
                                == subcircuit.data[cur_ind_plus].operation.name
                            ):
                                ind = cur_ind_plus - offset
                                break

                    if "cut" in op.operation.name:
                        (
                            offset,
                            classical_bit_index,
                            inserted_operations,
                        ) = _insert_2qubit_gate_cut_qpd(
                            ind,
                            op,
                            subcircuit,
                            offset,
                            qpd_qubits,
                            qpd,
                            classical_bit_index,
                            inserted_operations,
                        )

                    elif "Meas" in op.operation.name or "Init" in op.operation.name:
                        (
                            offset,
                            classical_bit_index,
                            inserted_operations,
                        ) = _insert_wire_cut_qpd(
                            ind,
                            op,
                            subcircuit,
                            offset,
                            qpd_qubits,
                            qpd,
                            classical_bit_index,
                            inserted_operations,
                        )
                    else:
                        raise ValueError(
                            f"""Unknown placeholder operation: {op.operation.name}.
                            Actual operation: {subcircuit.data[ind + offset]}"""
                        )

                subcircuit = _finalize_subcircuit(subcircuit, qpd_qubits)
                cur_set_circuits[id_meas_subcircuit_index] = subcircuit
            obs_set_circuits.append(cur_set_circuits)
        experiment_circuits.append(obs_set_circuits)

    cut_experiment = CutExperiment(
        experiment_circuits,
        cut_circuit.cut_locations,
        cut_circuit.map_qubit,
        coefficients,
        observables,
        backend=backend,
    )

    logger.info(f"Generated {cut_experiment.num_circuits} circuits for the experiment.")

    return cut_experiment


def run_experiments(  # noqa: C901
    cut_experiment: CutExperiment,
    shots: int = 2**12,
    backend=None,
    max_batch_size: int = 100,
    executor="auto",
    n_workers: int | None = None,
) -> RawResult:
    """Run experiment circuits.

    Loop through experiment circuits and then loop through circuit group and run each
    circuit. Store results as [group0, group1, ...] where group is [res0, res1, ...].
    where res is "xxx yy": count xxx are the measurements from the end of circuit
    measurements on the meas classical register and yy are the qpd basis measurement
    results from the qpd_meas class register.

    Args:
        experiment_circuits (CutCircuit): experiment circuits
        cut_locations (np.ndarray[CutLocation]): list of cut locations
        shots (int): number of shots per circuit run (optional)
        backend: backend used for running the circuits (optional)
        max_batch_size (int): maximum number of circuits submitted per backend.run
            call. Larger batches reduce per-job overhead on real hardware.
        executor: how to run the (independent) experiment circuits. Either a
            ``CircuitExecutor`` instance or one of ``"auto"`` (default, serial),
            ``"serial"``, ``"multiprocessing"``, ``"mpi"``. See ``QCut.executors``.
            For multi-node MPI runs prefer the ``QCut.mpi_run`` entry point.
        n_workers (int): worker count for the multiprocessing executor (optional).

    Returns:
        list[TotalResult]:
            list of transformed results

    """
    gamma = sum(abs(c) for c in cut_experiment.coefficients)
    samples = int(np.power(gamma, 2) / np.power(ERROR, 2))
    samples = int(samples / cut_experiment.num_groups)
    if backend is None:
        backend = AerSimulator()

    results: list[list[dict[int, dict[str, int]]]] = [
        [{} for _ in group] for group in cut_experiment.experiments
    ]

    runnable: list[tuple[tuple[int, int, int], QuantumCircuit]] = []
    empty_locations: list[tuple[tuple[int, int, int], int]] = []
    for group_idx, circuit_group in enumerate(cut_experiment.experiments):
        for obs_idx, obs_group in enumerate(circuit_group):
            for sub_idx, subcircuit in obs_group.items():
                key = (group_idx, obs_idx, sub_idx)
                if _has_measurements(subcircuit):
                    runnable.append((key, subcircuit))
                else:
                    empty_locations.append((key, subcircuit.num_clbits))

    from QCut.executors import get_default_executor

    ex = get_default_executor(backend, executor, n_workers)
    logger.info(
        f"Running {len(runnable)} circuits via {type(ex).__name__}"
        f" with {shots} shots each"
    )
    # Sort by key so the per-(group, obs) result dicts are populated in ascending
    # sub_idx order regardless of the executor's return order. Downstream
    # reconstruction combines subcircuit results by dict order, so this ordering
    # must be deterministic and match the serial path.
    for key, circ_counts in sorted(ex.run(runnable, shots, max_batch_size)):
        group_idx, obs_idx, sub_idx = key
        results[group_idx][obs_idx][sub_idx] = dict(circ_counts)

    for (group_idx, obs_idx, sub_idx), num_clbits in empty_locations:
        results[group_idx][obs_idx][sub_idx] = {" " + "0" * num_clbits: shots}

    all_keys = results[0][0].keys()

    for ind, sub_result in enumerate(results):
        for exp_ind, experiment_run in enumerate(sub_result):
            if experiment_run.keys() != all_keys:
                for key, val in results[0][0].items():
                    if key not in experiment_run:
                        experiment_run[key] = val

    return RawResult(results, samples, shots)


def run_cut_circuit(
    cut_circuit: CutCircuit,
    observables: SparsePauliOp,
    backend=AerSimulator(),
    max_batch_size: int = 100,
    executor="auto",
    n_workers: int | None = None,
) -> list[float]:
    """After splitting the circuit run the rest of the circuit knitting sequence.

    Args:
        subcircuits (list[QuantumCircuit]):
            subcircuits containing the placeholder operations
        cut_locations (np.ndarray[CutLocation]): list of cut locations
        observables (list[int | list[int]]):
            list of observables as qubit indices (Z observable)
        backend: backend to use for running experiment circuits (optional)
        max_batch_size (int): maximum number of circuits submitted per backend.run
            call (optional)
        executor: execution strategy passed to ``run_experiments`` (optional).
        n_workers (int): worker count for the multiprocessing executor (optional).

    Returns:
        list: a list of expectation values

    """
    from QCut.executors import _is_sampler

    # Transpile subcircuits to a hardware/fake backend (not for ideal Aer or for
    # samplers, which expect already-transpiled circuits).
    if not isinstance(backend, AerSimulator) and not _is_sampler(backend):
        transpiled_subcircuits = transpile_subcircuits(
            cut_circuit, backend, optimization_level=3
        )

        cut_experiment = get_experiment_circuits(transpiled_subcircuits, observables)
    else:
        cut_experiment = get_experiment_circuits(cut_circuit, observables)

    results = run_experiments(
        cut_experiment,
        backend=backend,
        max_batch_size=max_batch_size,
        executor=executor,
        n_workers=n_workers,
    )

    return estimate_expectation_values(results, cut_experiment.expv_data())


def run(
    circuit: QuantumCircuit,
    observables: SparsePauliOp,
    backend=AerSimulator(),
    max_batch_size: int = 100,
    executor="auto",
    n_workers: int | None = None,
) -> list[float]:
    """Run the whole circuit knitting sequence with one function call.

    Args:
        circuit (QuantumCircuit): circuit with cut experiments
        observables (list[int | list[int]]):
            list of observbles in the form of qubit indices (Z-obsevable).
        backend: backend to use for running experiment circuits (optional)
        max_batch_size (int): maximum number of circuits submitted per backend.run
            call (optional)
        executor: execution strategy passed to ``run_experiments`` (optional).
        n_workers (int): worker count for the multiprocessing executor (optional).

    Returns:
        list: a list of expectation values

    """
    # circuit = circuit.copy()
    cut_circuit = get_locations_and_subcircuits(circuit)

    return run_cut_circuit(
        cut_circuit,
        observables,
        backend,
        max_batch_size,
        executor=executor,
        n_workers=n_workers,
    )


def mpi_run(
    circuit: QuantumCircuit,
    observables: SparsePauliOp,
    backend=None,
    shots: int = 2**12,
    max_batch_size: int = 100,
    base_seed: int = 1234,
) -> list[float] | None:
    """Run the full circuit-knitting pipeline under MPI (LUMI task-farm).

    Launch with ``srun python job.py`` / ``mpirun -np N python job.py``. Every rank
    executes this function; rank 0 builds the experiment circuits and drives the
    pipeline, while worker ranks participate only in the execution scatter/gather.
    Rank 0 returns the expectation values; worker ranks return ``None``.

    Replicable simulators (Aer, IQM fake backends) are farmed across all ranks. For
    a remote QPU backend or a sampler the work cannot be parallelized this way, so
    only rank 0 runs (serially) and workers return immediately.

    Args:
        circuit (QuantumCircuit): circuit containing cut markers.
        observables (SparsePauliOp): observables to estimate.
        backend: execution backend; defaults to a per-rank ``AerSimulator``.
        shots (int): shots per circuit.
        max_batch_size (int): circuits per backend.run call within a rank.
        base_seed (int): rank ``r`` uses ``base_seed + r`` so shot noise is
            decorrelated across ranks (Aer backends).

    Returns:
        list of expectation values on rank 0, ``None`` on worker ranks.

    """
    from mpi4py import MPI

    from QCut.executors import BackendAdapter, MPIExecutor

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    replicable = BackendAdapter(backend).replicable and size > 1

    def _pipeline(ex) -> list[float]:
        cut_circuit = get_locations_and_subcircuits(circuit)
        cut_experiment = get_experiment_circuits(cut_circuit, observables)
        results = run_experiments(
            cut_experiment,
            shots=shots,
            backend=backend,
            max_batch_size=max_batch_size,
            executor=ex,
        )
        return estimate_expectation_values(results, cut_experiment.expv_data())

    if not replicable:
        # Single-submitter: only rank 0 does the work.
        return _pipeline("serial") if rank == 0 else None

    threads = int(os.environ.get("OMP_NUM_THREADS", "0")) or None
    mpi_ex = MPIExecutor(
        backend=backend, comm=comm, threads=threads, base_seed=base_seed
    )
    if rank == 0:
        return _pipeline(mpi_ex)
    # Workers contribute to the collective inside MPIExecutor.run, then exit.
    mpi_ex.run(None, shots, max_batch_size)
    return None
