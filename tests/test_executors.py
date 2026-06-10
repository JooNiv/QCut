"""Tests for the pluggable circuit executors.

The contract: every executor must reproduce the serial expectation values, and the
backend router must classify Aer / IQM-fake / remote / sampler backends correctly.
"""

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

import QCut as ck
from QCut import cut, cutGate
from QCut.executors import (
    BackendAdapter,
    SerialExecutor,
    _is_replicable_simulator,
    _is_sampler,
    _partition_lpt,
    get_default_executor,
)

# Two CX cuts + a wire cut -> a few hundred independent experiment circuits.
cut_circuit = QuantumCircuit(4)
mult = 1.635
cut_circuit.r(mult * 0.46262, mult * 0.1446, 0)
cut_circuit.append(**cutGate(CXGate(), 0, 1))
cut_circuit.append(cut(), [1])
cut_circuit.cx(1, 2)
cut_circuit.cx(2, 3)

observables = SparsePauliOp(["IIIZ", "IIZI", "IZII", "IIZZ"])
expected = [0.727323, 0.727323, 0.727323, 1.000000]


def _run(executor, n_workers=None):
    return ck.run(
        cut_circuit.copy(),
        observables,
        AerSimulator(),
        executor=executor,
        n_workers=n_workers,
    )


# --------------------------------------------------------------------------- #
# Backend classification
# --------------------------------------------------------------------------- #
class _FakeAdonis:  # mimics iqm.qiskit_iqm fake backend naming
    pass


_FakeAdonis.__module__ = "iqm.qiskit_iqm.fake_backends.adonis"


class FiQCISampler:  # structural stand-in for fiqci.ems.FiQCISampler
    mitigation_level = 1

    def run(self, *a, **k):  # noqa: D401
        ...


class _RemoteIQMBackend:
    pass


_RemoteIQMBackend.__module__ = "iqm.qiskit_iqm.iqm_provider"


def test_router_classifies_replicable_simulators():
    assert _is_replicable_simulator(AerSimulator())
    assert _is_replicable_simulator(_FakeAdonis())
    assert _is_replicable_simulator(None)  # default Aer


def test_router_classifies_non_replicable():
    assert _is_sampler(FiQCISampler())
    assert not _is_replicable_simulator(FiQCISampler())
    assert not _is_replicable_simulator(_RemoteIQMBackend())


def test_adapter_flags():
    assert BackendAdapter(AerSimulator()).replicable
    assert BackendAdapter(FiQCISampler()).is_sampler
    assert not BackendAdapter(FiQCISampler()).replicable


def test_get_default_executor_routes_remote_to_serial():
    # Even when asked for multiprocessing, a non-replicable backend stays serial-safe
    # at run time (MultiprocessingExecutor falls back internally); auto -> serial.
    assert isinstance(get_default_executor(FiQCISampler(), "auto"), SerialExecutor)
    assert isinstance(get_default_executor(AerSimulator(), "serial"), SerialExecutor)


def test_get_default_executor_rejects_unknown():
    with pytest.raises(ValueError):
        get_default_executor(AerSimulator(), "nonsense")


# --------------------------------------------------------------------------- #
# Load balancing
# --------------------------------------------------------------------------- #
def test_partition_lpt_covers_all_items():
    runnable = [((0, 0, i), QuantumCircuit(2)) for i in range(10)]
    bins = _partition_lpt(runnable, 3)
    flat = [item for b in bins for item in b]
    assert len(flat) == len(runnable)
    assert {k for k, _ in flat} == {k for k, _ in runnable}


# --------------------------------------------------------------------------- #
# Executors reproduce serial results
# --------------------------------------------------------------------------- #
def test_serial_executor_matches_expected():
    values = _run("serial")
    for got, want in zip(values, expected):
        assert abs(got - want) < 0.1


def test_multiprocessing_matches_serial():
    values = _run("multiprocessing", n_workers=2)
    for got, want in zip(values, expected):
        assert abs(got - want) < 0.1


def test_explicit_executor_instance_accepted():
    values = ck.run(
        cut_circuit.copy(), observables, AerSimulator(), executor=SerialExecutor()
    )
    for got, want in zip(values, expected):
        assert abs(got - want) < 0.1
