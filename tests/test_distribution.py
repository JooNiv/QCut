"""The separable form of a reconstructed distribution, and the queries on it.

Everything here works on modes made up on the spot, so none of it needs a simulator.
The point is that the cheap queries agree with the dense table they avoid building.
"""

import numpy as np
import pytest

from QCut.execution.distribution import SeparableDistribution

#: Exact-arithmetic comparisons. The queries and the table are the same sum in a
#: different order, so they agree to rounding; measured worst case over the cases below
#: is 3e-16, leaving three orders of margin.
EXACT = 1e-12


def _make(width, bits, groups=4, seed=0, signed=True):
    """A distribution over ``width`` bits whose modes carry ``bits``.

    Weights are deliberately signed: a subcircuit's weight carries the qpd measurement
    signs, so it is not a probability vector and nothing may assume it is.
    """
    rng = np.random.default_rng(seed)
    modes = [rng.normal(size=(1 << len(held), groups)) for held in bits]
    if not signed:
        modes = [np.abs(mode) for mode in modes]
    coefficients = rng.normal(size=groups)
    return SeparableDistribution(width, bits, modes, coefficients)


def _dense(distribution):
    """The table, built straight from the definition rather than from ``array``."""
    width = distribution.width
    table = np.empty(1 << width)
    for outcome in range(1 << width):
        flipped = outcome ^ ((1 << width) - 1)
        total = 0.0
        for group in range(distribution.groups):
            term = distribution.coefficients[group]
            for held, mode in zip(distribution.bits, distribution.modes):
                local = 0
                for position, bit in enumerate(held):
                    local |= ((flipped >> bit) & 1) << position
                term *= mode[local, group]
            total += term
        table[outcome] = distribution.constant - total
    return table


#: Layouts worth covering: an even split, a lopsided one, three modes, a mode holding
#: nothing (a subcircuit measuring none of the chosen qubits), and a single mode.
LAYOUTS = [
    (4, [[0, 1], [2, 3]]),
    (4, [[0], [1, 2, 3]]),
    (5, [[0, 2], [1], [3, 4]]),
    (3, [[], [0, 1, 2]]),
    (3, [[0, 1, 2]]),
    (1, [[0]]),
    (6, [[0, 1, 5], [2, 3, 4]]),
]
LAYOUT_IDS = [
    f"w{width}-{len(bits)}modes-{index}" for index, (width, bits) in enumerate(LAYOUTS)
]


@pytest.mark.parametrize(("width", "bits"), LAYOUTS, ids=LAYOUT_IDS)
def test_the_array_is_the_definition(width, bits):
    """``array`` reorders the sum for speed; it must not change it."""
    distribution = _make(width, bits)
    assert np.abs(distribution.array() - _dense(distribution)).max() < EXACT


@pytest.mark.parametrize(("width", "bits"), LAYOUTS, ids=LAYOUT_IDS)
def test_one_outcome_agrees_with_the_whole_table(width, bits):
    """``value`` is the query that never builds anything, so it has to be right."""
    distribution = _make(width, bits)
    table = distribution.array()
    for outcome in range(1 << width):
        assert distribution.value(outcome) == pytest.approx(table[outcome], abs=EXACT)


@pytest.mark.parametrize(("width", "bits"), LAYOUTS, ids=LAYOUT_IDS)
def test_the_distribution_sums_to_one_by_construction(width, bits):
    """The constant is worked out from the modes, so the total cannot drift."""
    distribution = _make(width, bits)
    assert distribution.array().sum() == pytest.approx(1.0, abs=EXACT)


@pytest.mark.parametrize(("width", "bits"), LAYOUTS, ids=LAYOUT_IDS)
def test_the_separable_total_matches_summing_the_table(width, bits):
    distribution = _make(width, bits)
    table = distribution.array()
    # sum_y T[y] = 2**k * constant - 1, rearranged from p = constant - T.
    assert distribution.total() == pytest.approx(
        (1 << width) * distribution.constant - 1.0, abs=EXACT
    )
    assert table.sum() == pytest.approx(1.0, abs=EXACT)


@pytest.mark.parametrize(("width", "bits"), LAYOUTS, ids=LAYOUT_IDS)
@pytest.mark.parametrize("count", [1, 3, 8])
def test_top_returns_the_same_outcomes_as_sorting_the_table(width, bits, count):
    """Ranks may differ on exact ties, so the set and the values are what is pinned."""
    distribution = _make(width, bits)
    table = distribution.array()
    top = distribution.top(count)

    wanted = min(count, 1 << width)
    assert len(top) == wanted
    assert [value for _outcome, value in top] == sorted(
        (value for _o, value in top), reverse=True
    ), "most likely first"

    best = np.sort(table)[::-1][:wanted]
    assert np.abs(np.sort([v for _o, v in top])[::-1] - best).max() < EXACT
    for outcome, value in top:
        assert value == pytest.approx(table[outcome], abs=EXACT)


def test_top_finds_the_peak_of_a_concentrated_distribution():
    """The pruning must not be able to prune away the answer."""
    rng = np.random.default_rng(7)
    # One dominant outcome per mode, negative so that it is the peak of p and not its
    # lowest point: p = constant - T, so the likeliest outcome is the most negative T.
    modes = []
    for held in ([0, 1, 2], [3, 4, 5]):
        mode = rng.normal(size=(1 << len(held), 3)) * 0.01
        mode[5] = -3.0
        modes.append(mode)
    distribution = SeparableDistribution(6, [[0, 1, 2], [3, 4, 5]], modes, np.ones(3))

    table = distribution.array()
    assert distribution.top(1)[0][0] == int(np.argmax(table))


def test_top_fills_up_to_count_when_the_support_is_smaller():
    """A near-empty support leaves ties at the constant, and it still returns ``count``.

    Pinned because the search stops descending into a zero-bound slice, which happens
    exactly while the list is still short.
    """
    modes = [np.zeros((4, 2)), np.zeros((4, 2))]
    modes[0][1] = [-1.0, -1.0]
    modes[1][2] = [1.0, 1.0]
    distribution = SeparableDistribution(4, [[0, 1], [2, 3]], modes, np.ones(2))
    table = distribution.array()

    top = distribution.top(8)
    assert len(top) == 8
    assert (
        np.abs(np.array([value for _o, value in top]) - np.sort(table)[::-1][:8]).max()
        < EXACT
    )
    assert top[0][0] == int(np.argmax(table)), "the one supported outcome leads"
    assert all(value == pytest.approx(distribution.constant) for _o, value in top[1:])


def test_top_is_capped_at_the_number_of_outcomes():
    distribution = _make(3, [[0], [1, 2]])
    assert len(distribution.top(1000)) == 8


def test_top_wants_a_positive_count():
    with pytest.raises(ValueError, match="at least one"):
        _make(3, [[0], [1, 2]]).top(0)


@pytest.mark.parametrize(
    "keep",
    [[0], [1], [0, 1], [1, 0], [0, 2], [2, 0], [0, 1, 2], [3, 1]],
    ids=lambda keep: "keep" + "".join(str(bit) for bit in keep),
)
def test_the_marginal_is_the_table_summed_over_the_dropped_bits(keep):
    distribution = _make(4, [[0, 1], [2, 3]], seed=3)
    table = distribution.array()

    marginal = distribution.marginal(keep)
    assert marginal.width == len(keep)
    reduced = marginal.array()

    expected = np.zeros(1 << len(keep))
    for outcome in range(1 << distribution.width):
        index = 0
        for position, bit in enumerate(keep):
            index |= ((outcome >> bit) & 1) << position
        expected[index] += table[outcome]
    assert np.abs(reduced - expected).max() < EXACT


def test_a_marginal_of_a_marginal_is_the_marginal():
    """The form survives the operation, so it has to compose."""
    distribution = _make(5, [[0, 2], [1], [3, 4]], seed=11)
    once = distribution.marginal([0, 1, 3]).marginal([0, 2])
    twice = distribution.marginal([0, 3])
    assert np.abs(once.array() - twice.array()).max() < EXACT


@pytest.mark.parametrize(
    ("keep", "message"),
    [
        ([], "at least one bit"),
        ([0, 0], "must not repeat"),
        ([4], "outside a 4-bit outcome"),
        ([-1], "outside a 4-bit outcome"),
    ],
)
def test_the_marginal_bits_have_to_exist(keep, message):
    distribution = _make(4, [[0, 1], [2, 3]])
    with pytest.raises(ValueError, match=message):
        distribution.marginal(keep)


def test_an_experiment_where_every_group_was_dropped_is_uniform():
    """Every group can drop out, e.g. when no label came up. Nothing may assume one."""
    distribution = SeparableDistribution(
        3, [[0], [1, 2]], [np.zeros((2, 0)), np.zeros((4, 0))], np.zeros(0)
    )
    assert distribution.total() == 0.0
    assert distribution.constant == pytest.approx(1 / 8)
    assert distribution.array() == pytest.approx(np.full(8, 1 / 8))
    assert distribution.value(3) == pytest.approx(1 / 8)
    assert distribution.top(4) == [(outcome, 1 / 8) for outcome in range(4)]
