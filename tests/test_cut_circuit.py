from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import CXGate
from qiskit.quantum_info import SparsePauliOp

import QCut as ck
from QCut import cut, cutGate

cut_qc = QuantumCircuit(4)

mult = Parameter("mult")
cut_qc.r(mult * 0.46262, mult * 0.1446, 0)
cut_qc.append(**cutGate(CXGate(), 0, 1))
cut_qc.append(cut(), [1])
cut_qc.cx(1, 2)
cut_qc.cx(2, 3)


cut_circuit = ck.get_locations_and_subcircuits(cut_qc)
num_subcircuits = 3

num_qubits = [1, 1, 3]


def test_cut_circuit_properties():
    assert cut_circuit.num_qubits == num_qubits
    assert cut_circuit.num_subcircuits == num_subcircuits


def test_cut_circuit_assign_parameters():
    cut_circuit_param = cut_circuit.assign_parameters({"mult": 2}, inplace=False)

    for exp_ind, subcircuits in enumerate(cut_circuit_param.subcircuits):
        for circ_ind, value in enumerate(subcircuits):
            for par in value.params:
                if hasattr(par, "parameters"):
                    for elem in par.parameters:
                        assert "mult" not in elem.name
                else:
                    assert True


observables = SparsePauliOp(["IIIZ", "IIZI", "IZII", "IIZZ"])
cut_experiment = ck.get_experiment_circuits(cut_circuit, observables)

exp_num_qubits = [1, 1, 3]
exp_num_circuits = 144
# group_size is the instruction count of the first subcircuit. The CX cut now uses a
# generated QPD, whose operations carry the KAK local unitaries as explicit `u` gates.
# num_groups and num_circuits are unchanged.
exp_group_size = 5
exp_num_groups = 48
exp_num_obs_groups = 1


def test_cut_experiment_properties():
    assert cut_experiment.num_qubits == exp_num_qubits
    assert cut_experiment.num_circuits == exp_num_circuits
    assert cut_experiment.group_size == exp_group_size
    assert cut_experiment.num_groups == exp_num_groups
    assert cut_experiment.num_obs_groups == exp_num_obs_groups


def test_cut_experiment_assign_parameters():
    cut_experiment_param = cut_experiment.assign_parameters({"mult": 2}, inplace=False)

    for exp_ind, subcircuits in enumerate(cut_experiment_param.experiments):
        for obs_circ in subcircuits:
            for circ in obs_circ.values():
                for circ_ind, value in enumerate(circ):
                    for par in value.params:
                        if hasattr(par, "parameters"):
                            for elem in par.parameters:
                                assert "mult" not in elem.name
                        else:
                            assert True


def test_results_carry_their_own_experiment():
    """The estimator is called on the result alone, with nothing threaded through."""
    from qiskit_aer import AerSimulator

    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.append(**cutGate(CXGate(), 0, 1))
    observables = SparsePauliOp(["IZ", "ZI"])

    cut_circuit = ck.get_locations_and_subcircuits(circuit)
    experiment = ck.get_experiment_circuits(cut_circuit, observables)
    results = ck.run_experiments(
        experiment, backend=AerSimulator(seed_simulator=17), shots=2048
    )

    assert results.experiment is experiment
    assert len(ck.estimate_expectation_values(results)) == len(observables)


def test_results_without_an_experiment_say_so():
    """A hand-built result carries nothing, so the estimator cannot interpret it."""
    import pytest

    from QCut.execution.qcutresult import RawResult

    bare = RawResult([], 1024)
    assert bare.experiment is None
    with pytest.raises(ValueError, match="no experiment"):
        ck.estimate_expectation_values(bare)


def test_every_exported_name_exists():
    """``from QCut import *`` must not raise.

    ``__all__`` listed a name that had been removed, so the star import failed with an
    AttributeError. Nothing else in the suite exercised it.
    """
    missing = [name for name in ck.__all__ if not hasattr(ck, name)]
    assert missing == []
