from qiskit.quantum_info import SparsePauliOp
from qiskit.result import QuasiDistribution

from QCut.execution.postprocess import estimate_expectation_values
from QCut.execution.qcutresult import RawResult


class QuasiProbabilities(dict):
    """A reconstructed distribution over bitstrings, in three views.

    The object is the quasi-probabilities themselves: it is the ``{bitstring: value}``
    dict it appears to be, so it indexes, iterates and plots like one, and
    :meth:`quasi_probabilities` hands back the same numbers as a plain dict.
    :meth:`nearest_probabilities` gives the closest true distribution and :meth:`counts`
    scales that to a shot count.

    The sum of elements is exactly one, but that is by construction rather than evidence
    of anything. The empty subset contributes one and every other subset cancels over
    the bitstrings, so the sum is one however wrong the estimate is. An individual value
    can come out negative, because each is a signed sum of separately estimated
    expectation values. How negative says how far this estimate sits from a
    physical distribution.

    Attributes:
        shots (int | None): what the experiment ran at, carried so :meth:`counts` has a
            scale to work with by default.
    """

    def __init__(self, data, shots: int | None = None):
        """Init."""
        super().__init__(data)
        self.shots = shots

    def quasi_probabilities(self) -> dict[str, float]:
        """The values as reconstructed, negative ones included.

        Returns:
            dict[str, float]: what this object holds, as a plain dict.
        """
        return dict(self)

    def nearest_probabilities(self) -> dict[str, float]:
        """The closest true distribution.

        Bitstrings clipped to zero are kept, so this covers the same keys as the
        quasi-probabilities it came from.

        Returns:
            dict[str, float]: the projected distribution, keyed by bitstring.
        """
        width = len(next(iter(self), ""))
        nearest = QuasiDistribution(dict(self)).nearest_probability_distribution()
        projected = dict.fromkeys(self, 0.0)
        projected.update(
            {format(key, f"0{width}b"): value for key, value in nearest.items()}
        )
        return projected

    def counts(self, shots: int | None = None) -> dict[str, float]:
        """The distribution scaled to a shot count.

        Projected first, since a count cannot be negative, so this is
        :meth:`nearest_probabilities` multiplied through.

        These are not the counts of that many measurements, and reading them as though
        they were overstates how much is known. A cut experiment estimates each value as
        a signed sum over its subexperiments, which inflates the variance by roughly
        ``gamma`` squared against running the circuit whole, so the spread here is far
        wider than the same number of direct shots would give.

        Args:
            shots (int): what to scale by. Defaults to the shots the experiment ran at.

        Returns:
            dict[str, float]: the projected distribution scaled by ``shots``.

        Raises:
            ValueError: no shot count was carried and none was given.
        """
        shots = self.shots if shots is None else shots
        if shots is None:
            raise ValueError(
                "these results carried no shot count, so pass shots to scale by"
            )
        return {
            key: value * shots for key, value in self.nearest_probabilities().items()
        }


def _all_z_paulis_for_subset(
    number_of_qubits: int, qubit_indices: list[int]
) -> SparsePauliOp:
    """Every Pauli Z string over ``qubit_indices``, identity excluded.

    Args:
        number_of_qubits (int): width of the uncut circuit, which the strings span.
        qubit_indices (list[int]): the qubits to measure, in the order their bits are
            read back.

    Returns:
        SparsePauliOp: the ``2**k - 1`` non-identity Z strings, ordered so that the
        ``i``-th carries Z on the qubits named by the set bits of ``i``.

    Raises:
        ValueError: the qubits are empty, repeated, or outside the circuit.
    """
    if not qubit_indices:
        raise ValueError("qubits must name at least one qubit")
    if len(set(qubit_indices)) != len(qubit_indices):
        raise ValueError(f"qubits must not repeat, got {qubit_indices}")
    outside = [q for q in qubit_indices if not 0 <= q < number_of_qubits]
    if outside:
        raise ValueError(
            f"qubits {outside} are outside the {number_of_qubits}-qubit circuit"
        )

    paulis = []
    for i in range(2 ** len(qubit_indices)):
        pauli_str = ["I"] * number_of_qubits
        for j, qubit_index in enumerate(qubit_indices):
            ind = number_of_qubits - qubit_index - 1
            if (i >> j) & 1:
                pauli_str[ind] = "Z"
        paulis.append("".join(pauli_str))
    return SparsePauliOp(paulis[1:])


def _gen_bitstrings(n):
    """Generate all bitstrings of length n."""
    return [format(i, f"0{n}b") for i in range(2**n)]


def _gen_subsets(n):
    """Generate all subsets of a set of size n."""
    subsets = []
    for i in range(2**n):
        subset = [j for j in range(n) if (i & (1 << j))]
        subsets.append(subset)
    return subsets


def _map_expvs_to_subsets(expvs, subsets):
    """Map expectation values to their corresponding subsets."""
    expv_mapping = {(): 1.0}  # Initialize with the identity observable
    subsets_no_empty = subsets[1:]  # Exclude the empty subset
    for subset, expv in zip(subsets_no_empty, expvs):
        expv_mapping[tuple(subset)] = expv
    return expv_mapping


def _parity_bits(bitstring, subset):
    """Calculate the parity of a bitstring for a given subset of indices."""
    bits = bitstring[::-1]
    return sum(int(bits[i]) for i in subset) % 2


def _reconstruct_probs(expv_subset_mapping, bits):
    """Reconstruct the probabilities of each bitstring from the expectation values."""
    probs = {}
    for x in bits:
        total = sum(
            (-1) ** _parity_bits(x, subset) * expv
            for subset, expv in expv_subset_mapping.items()
        )
        probs[x] = total / 2 ** len(bits[0])
    return probs


def estimate_probabilities(result: RawResult) -> QuasiProbabilities:
    """Reconstruct the distribution over the qubits the experiment measured.

    Every Pauli Z over the chosen qubits is estimated and the distribution follows from
    them, so this costs ``2**k`` values for ``k`` qubits. The circuits do not multiply
    with it: Z observables all commute, so they share one measurement setting and the
    experiment is the size it would have been for a single observable. Only this
    reconstruction grows.

    The bitstrings are written with the first of the chosen qubits last, so a subset
    given as ``[2, 0]`` reads qubit 2 as the rightmost character.

    Args:
        result (RawResult): results of an experiment built with ``qubits``.

    Returns:
        QuasiProbabilities: the distribution, which also offers the projected
        distribution and a scaling to counts.

    Raises:
        ValueError: the experiment was built with observables rather than ``qubits``, so
            it does not carry the full set this needs.
    """
    if not result.experiment.can_reconstruct_probabilities:
        raise ValueError(
            "Cannot reconstruct probabilities for this experiment. Pass ``qubits`` to "
            "the ``get_experiment_circuits`` function to enable this feature."
        )

    exps = estimate_expectation_values(result)
    len_subset = len(result.experiment.qubits)
    subsets = _gen_subsets(len_subset)
    expv_mapping = _map_expvs_to_subsets(exps, subsets)
    bits = _gen_bitstrings(len_subset)
    return QuasiProbabilities(
        _reconstruct_probs(expv_mapping, bits), shots=result.shots
    )
