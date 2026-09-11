"""
The separable form a reconstructed distribution takes, and the queries on it.
"""

from __future__ import annotations

import heapq

import numpy as np


class SeparableDistribution:
    r"""A reconstructed distribution, held as a sum of separable terms.

    Args:
        width (int): how many qubits the distribution spans, :math:`k`.
        bits (list[list[int]]): per mode, which bits of the outcome it carries, in
            ascending order. The modes partition ``range(width)``: every bit belongs to
            exactly one, which is what makes the terms separable.
        modes (list[np.ndarray]): per mode, an array of shape
            ``(2**len(bits[i]), groups)``. Entry ``[y, g]`` is the weight group ``g``
            gave the local outcome ``y``, where local bit ``t`` is the outcome bit
            ``bits[i][t]``. Weights carry the qpd signs, so they are not probabilities
            and can be negative.
        coefficients (np.ndarray): one per group, carrying the quasiprobability
            coefficient and the shared wire cut parity.

    Attributes:
        constant (float): the :math:`(1 + \sum_y T[y]) / 2^k` above. Worked out from the
            modes rather than by summing the table, so it costs nothing and the
            distribution sums to one by construction.
    """

    def __init__(
        self,
        width: int,
        bits: list[list[int]],
        modes: list[np.ndarray],
        coefficients: np.ndarray,
    ) -> None:
        """Init."""
        self.width = width
        self.bits = bits
        self.modes = modes
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.constant = (1.0 + self.total()) / (1 << width)

    @property
    def groups(self) -> int:
        """How many groups contributed. A group with an empty result is not one."""
        return len(self.coefficients)

    def total(self) -> float:
        r"""The sum :math:`\sum_y T[y]`, which is separable like everything else."""
        if self.groups == 0:
            return 0.0
        totals = np.prod([mode.sum(axis=0) for mode in self.modes], axis=0)
        return float(self.coefficients @ totals)

    def _outcome(self, blocks: list[int]) -> int:
        """One local index per mode, assembled into the outcome they name together."""
        outcome = 0
        for bits, local in zip(self.bits, blocks):
            for position, bit in enumerate(bits):
                outcome |= ((local >> position) & 1) << bit
        return outcome

    def _blocks(self, outcome: int) -> list[int]:
        """The reverse: an outcome split into one local index per mode."""
        blocks = []
        for bits in self.bits:
            local = 0
            for position, bit in enumerate(bits):
                local |= ((outcome >> bit) & 1) << position
            blocks.append(local)
        return blocks

    def array(self) -> np.ndarray:
        """The whole distribution, indexed by outcome.

        Returns:
            np.ndarray: ``2**width`` quasi-probabilities, entry ``x`` being the
            probability of the outcome whose bit ``b`` is ``(x >> b) & 1``.
        """
        accumulated = np.zeros((2,) * self.width)
        term = np.empty((2,) * self.width)
        for group in range(self.groups):
            term.fill(self.coefficients[group])
            for bits, mode in zip(self.bits, self.modes):
                shape = [1] * self.width
                for bit in bits:
                    shape[self.width - 1 - bit] = 2
                term *= mode[:, group].reshape(shape)
            accumulated += term
        return self.constant - accumulated.reshape(-1)[::-1]

    def value(self, outcome: int) -> float:
        """One outcome's quasi-probability, without building anything.

        Costs one pass over the groups per mode, so it does not grow with ``width``.

        Args:
            outcome (int): the outcome, bit ``b`` being ``(outcome >> b) & 1``.

        Returns:
            float: its quasi-probability, which may be negative.
        """
        if self.groups == 0:
            return self.constant
        weights = self.coefficients
        for mode, local in zip(self.modes, self._blocks(self._flip(outcome))):
            weights = weights * mode[local]
        return self.constant - float(weights.sum())

    def _flip(self, outcome: int) -> int:
        """The complement of an outcome, which is the index into :math:`T`."""
        return outcome ^ ((1 << self.width) - 1)

    def top(self, count: int) -> list[tuple[int, float]]:
        """The most likely outcomes, exactly, without building the whole table.

        Args:
            count (int): how many outcomes to return. Capped at ``2**width``.

        Returns:
            list[tuple[int, float]]: ``count`` pairs of ``(outcome,
            quasi-probability)``, most likely first. Values may be negative, and are not
            projected onto a physical distribution -- that needs the whole table, see
            :meth:`QCut.QuasiProbabilities.nearest_probabilities`.

        Raises:
            ValueError: ``count`` is not positive.
        """
        if count < 1:
            raise ValueError(f"count must be at least one, got {count}")
        count = min(count, 1 << self.width)

        found: list[tuple[int, float]] = []
        if self.groups > 0:
            search = _Search(self, count)
            search.descend(0, self.coefficients, [])
            found = [
                (self._flip(outcome), self.constant - value)
                for value, outcome in ((-key, out) for key, out in search.heap)
            ]

        if len(found) < count:
            taken = {outcome for outcome, _value in found}
            spare = (out for out in range(1 << self.width) if out not in taken)
            found += [(next(spare), self.constant) for _ in range(count - len(found))]

        found.sort(key=lambda pair: -pair[1])
        return found

    def marginal(self, keep: list[int]) -> SeparableDistribution:
        """The distribution over a subset of the bits, summed over the rest.

        Args:
            keep (list[int]): the bits to keep, becoming bits ``0..len(keep)-1`` of the
                result in the order given.

        Returns:
            SeparableDistribution: the marginal, in the same separable form.

        Raises:
            ValueError: ``keep`` is empty, repeats a bit, or names one outside the
                distribution.
        """
        if not keep:
            raise ValueError("keep must name at least one bit")
        if len(set(keep)) != len(keep):
            raise ValueError(f"keep must not repeat, got {keep}")
        outside = [bit for bit in keep if not 0 <= bit < self.width]
        if outside:
            raise ValueError(f"bits {outside} are outside a {self.width}-bit outcome")

        position_of = {bit: position for position, bit in enumerate(keep)}
        new_bits, new_modes = [], []
        for bits, mode in zip(self.bits, self.modes):
            held = sorted(
                (position_of[bit], index)
                for index, bit in enumerate(bits)
                if bit in position_of
            )
            new_bits.append([position for position, _index in held])
            kept = [index for _position, index in held]
            new_modes.append(_summed_over(mode, kept, len(bits)))
        return SeparableDistribution(len(keep), new_bits, new_modes, self.coefficients)


def _summed_over(mode: np.ndarray, held: list[int], width: int) -> np.ndarray:
    """A mode summed over the local bits it is not keeping.

    Args:
        mode (np.ndarray): the mode, shape ``(2**width, groups)``.
        held (list[int]): the local bits to keep, in the order they become the local
            bits of the result.
        width (int): how many local bits ``mode`` is indexed by.

    Returns:
        np.ndarray: shape ``(2**len(held), groups)``.
    """
    if len(held) == width:
        if held == list(range(width)):
            return mode
    locals_ = np.arange(1 << width)
    target = np.zeros(1 << width, dtype=np.intp)
    for position, bit in enumerate(held):
        target |= ((locals_ >> bit) & 1) << position
    summed = np.zeros((1 << len(held), mode.shape[1]))
    np.add.at(summed, target, mode)
    return summed


class _Search:
    """Branch and bound state for :meth:`SeparableDistribution.top`."""

    def __init__(self, distribution: SeparableDistribution, count: int) -> None:
        """Init."""
        self.distribution = distribution
        self.count = count
        #: The best ``count`` found so far, as a max-heap on the value by negation.
        self.heap: list[tuple[float, int]] = []

        # Per mode, per group, the largest magnitude any one outcome can contribute.
        extremes = [np.abs(mode).max(axis=0) for mode in distribution.modes]

        self.tail = [np.ones(distribution.groups) for _ in range(len(extremes) + 1)]
        for level in range(len(extremes) - 1, -1, -1):
            self.tail[level] = self.tail[level + 1] * extremes[level]

    @property
    def limit(self) -> float:
        """How small an entry has to be to still make the list."""
        return -self.heap[0][0] if len(self.heap) >= self.count else np.inf

    def offer(self, outcome: int, value: float) -> None:
        """Keep an entry if it beats the worst one held."""
        if len(self.heap) < self.count:
            heapq.heappush(self.heap, (-value, outcome))
        elif value < -self.heap[0][0]:
            heapq.heapreplace(self.heap, (-value, outcome))

    def leaf(self, weights: np.ndarray, blocks: list[int]) -> None:
        """Read the last mode exactly, every group at once."""
        mode = self.distribution.modes[-1]
        values = mode @ weights
        take = min(self.count, values.size)
        for local in np.argpartition(values, take - 1)[:take]:
            self.offer(
                self.distribution._outcome([*blocks, int(local)]), float(values[local])
            )

    def descend(self, level: int, weights: np.ndarray, blocks: list[int]) -> None:
        """Choose this mode's outcome, best bound first, and stop when it cannot win."""
        if level == len(self.distribution.modes) - 1:
            self.leaf(weights, blocks)
            return
        mode = self.distribution.modes[level]
        bounds = np.abs(mode) @ (np.abs(weights) * self.tail[level + 1])
        for local in np.argsort(-bounds):
            beaten = len(self.heap) >= self.count and bounds[local] <= -self.limit
            if bounds[local] == 0.0 or beaten:
                break
            self.descend(level + 1, weights * mode[local], [*blocks, int(local)])
