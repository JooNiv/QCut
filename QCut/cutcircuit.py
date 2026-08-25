"""Class for nicely representing a cut circuit/experiment. Also implements
some of the same functionality as the qiskit QuantumCircuit class for
a group of circuits."""

from __future__ import annotations

from typing import Iterable

from qiskit import QuantumCircuit

from QCut.cutlocation import CutLocation, SingleQubitCutLocation
from QCut.options import CutOptions, resolve


class CutCircuit:
    """Class for representing a cut circuit. Contains the subcircuits, cut locations,
    and mapping of qubits. Also contains some of the same functionality as the qiskit
    QuantumCircuit class for a group of circuits."""

    def __init__(
        self,
        subcircuits: list[QuantumCircuit],
        cut_locations: list[CutLocation | SingleQubitCutLocation],
        map_qubit: dict[int, int],
        backend=None,
        options: CutOptions | None = None,
    ) -> None:
        """Init."""

        self.subcircuits = subcircuits
        self.cut_locations = cut_locations
        self.map_qubit = map_qubit
        self.backend = backend
        self.options = resolve(options)

    def assign_parameters(self, parameters: dict, inplace=False) -> CutCircuit | None:
        """Assign parameters to the circuits. Same as qiskit
        QuantumCircuit.assign_parameters.

        Parameters on a cut gate are bound too. Those live on the cut location rather
        than in any subcircuit, but a generated QPD needs them numeric.
        """
        bound_locations = [
            location.assign_parameters(parameters)
            if isinstance(location, CutLocation)
            else location
            for location in self.cut_locations
        ]

        if inplace:
            for ind, circuit in enumerate(self.subcircuits):
                try:
                    self.subcircuits[ind] = circuit.assign_parameters(parameters)
                except Exception:
                    pass
            self.cut_locations = bound_locations
            return

        else:
            new_circuits = []
            for ind, circuit in enumerate(self.subcircuits):
                try:
                    new_circuits.append(circuit.assign_parameters(parameters))
                except Exception:
                    new_circuits.append(circuit)
            return CutCircuit(
                subcircuits=new_circuits,
                cut_locations=bound_locations,
                map_qubit=self.map_qubit,
                backend=self.backend,
                options=self.options,
            )

    @property
    def num_qubits(self):
        """Number of qubits per subcircuit."""
        return [i.num_qubits for i in self.subcircuits]

    @property
    def num_subcircuits(self):
        """Total number of circuits."""
        return len(self.subcircuits)


class CutExperiment:
    def __init__(
        self,
        experiment_circuits: list[list[dict[int, QuantumCircuit]]],
        cut_locations: list[CutLocation | SingleQubitCutLocation],
        map_qubit: dict[int, int],
        coefficients: Iterable[float],
        observables,
        backend=None,
        options: CutOptions | None = None,
        num_draws: int | None = None,
        plan=None,
        qpd_bits: dict[tuple[int, int, int], tuple[int, int]] | None = None,
    ) -> None:
        """Init.

        ``num_draws`` records how many samples were drawn when the decomposition was
        sampled rather than enumerated. It is informational. The estimator does not need
        it, because the sampled coefficients already carry their multiplicity.

        ``plan`` is a :class:`QCut.qpd_locc.CommunicationPlan` when any wire cut
        exchanges its measured outcome, and None otherwise. Those experiments run in two
        phases, so execution needs to know which bits carry the outcome.

        ``qpd_bits`` says, per circuit, how many qpd measurement bits it writes and how
        many were dropped for going unwritten. Post-processing needs both: the first to
        find those bits without relying on how a backend reports its registers, and the
        second to restore the sign the dropped ones carried.
        """

        self.experiments = experiment_circuits
        self.backend = backend
        self.cut_locations = cut_locations
        self.map_qubit = map_qubit
        self.coefficients = coefficients
        self.observables = observables
        self.options = resolve(options)
        self._num_draws = num_draws
        self.plan = plan
        self.qpd_bits = qpd_bits or {}

    def expv_data(self):
        """Get data for expv calculation."""
        return {
            "cut_locations": self.cut_locations,
            "map_qubit": self.map_qubit,
            "coefficients": self.coefficients,
            "observables": self.observables,
            "num_exp_groups": self.num_groups,
            "qpd_bits": self.qpd_bits,
        }

    def assign_parameters(
        self, parameters: dict, inplace=False
    ) -> CutExperiment | None:
        """Assign parameters to the circuits. Same as qiskit
        QuantumCircuit.assign_parameters."""
        if inplace:
            for exp_ind, subcircuits in enumerate(self.experiments):
                for circ_ind, value in enumerate(subcircuits):
                    for ind, circuit in value.items():
                        try:
                            self.experiments[exp_ind][circ_ind][ind] = (
                                circuit.assign_parameters(parameters)
                            )
                        except Exception:
                            pass
            return
        else:
            new_experiments = []
            for exp_ind, subcircuits in enumerate(self.experiments):
                new_subcircuits = []
                for circ_ind, value in enumerate(subcircuits):
                    new_circuits = {}
                    for ind, circuit in value.items():
                        try:
                            new_circuits[ind] = circuit.assign_parameters(parameters)
                        except Exception:
                            new_circuits[ind] = circuit
                    new_subcircuits.append(new_circuits)
                new_experiments.append(new_subcircuits)
            return CutExperiment(
                experiment_circuits=new_experiments,
                cut_locations=self.cut_locations,
                backend=self.backend,
                map_qubit=self.map_qubit,
                coefficients=self.coefficients,
                observables=self.observables,
                options=self.options,
                num_draws=self._num_draws,
                plan=self.plan,
                qpd_bits=self.qpd_bits,
            )

    @property
    def num_qubits(self):
        """Number of qubits per subcircuit."""
        return [i.num_qubits for i in self.experiments[0][0].values()]

    @property
    def num_circuits(self):
        """Total number of circuits."""
        return sum(
            len(subcircuits) * len(subcircuits[0]) for subcircuits in self.experiments
        )

    @property
    def group_size(self):
        """Number of circuits in a group."""
        return len(self.experiments[0][0][0])

    @property
    def num_groups(self):
        """Number of circuit groups."""
        return len(self.experiments)

    @property
    def num_draws(self):
        """How many samples were drawn, or the group count if fully enumerated."""
        if self._num_draws is None:
            return self.num_groups
        return self._num_draws

    @property
    def communicates(self):
        """Whether any wire cut exchanges its outcome between the partitions."""
        return self.plan is not None

    @property
    def sampled(self):
        """Whether the decomposition was sampled rather than fully enumerated."""
        return self._num_draws is not None

    @property
    def num_obs_groups(self):
        return len(self.experiments[0])
