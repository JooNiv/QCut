"""User-facing configuration for the cutting workflow.

Options are collected once and carried on :class:`QCut.cutcircuit.CutCircuit`, which
passes them to :class:`QCut.cutcircuit.CutExperiment`. Everything downstream reads them
from there, so adding a knob does not mean widening five signatures.

They have to be fixed before the subcircuits are built, because they decide how the
quasiprobability decomposition is formed and therefore what the coefficients mean. A
:class:`CutOptions` is frozen for that reason.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from QCut.qcuterror import QCutError

#: How the experiment tensor is built. ``"exact"`` enumerates every combination of QPD
#: terms, ``"sample"`` draws from the quasiprobability distribution, and ``"auto"``
#: enumerates while the exact count stays within ``max_exact_groups``.
ExpansionStrategy = Literal["auto", "exact", "sample"]

#: When to merge runs of gates on the same qubit pair. ``"always"`` merges wherever it
#: lowers the cost of that pair on its own, ``"never"`` leaves every gate alone, and
#: ``"auto"`` costs both whole plans and keeps the cheaper. ``True`` and ``False`` are
#: accepted as ``"always"`` and ``"never"``.
ConsolidateStrategy = Literal["auto", "always", "never"]


@dataclass(frozen=True)
class CutOptions:
    """Configuration for a cutting run.

    ``consolidate`` merges runs of gates acting on the same qubit pair into one gate
    before cutting, so the pair costs one cut instead of several. See
    :data:`ConsolidateStrategy`. Merging can cost more than it saves once joint cutting
    is in play, because a merged run of gates about different axes is no longer a
    single-axis rotation and cannot be bundled, so ``"auto"`` compares the two plans
    outright rather than guessing.

    ``joint_rotation_cuts`` cuts parallel two-qubit rotation gates with one joint
    decomposition instead of one each, which costs strictly less in both sampling
    overhead and circuit count.

    ``expansion`` decides how the decomposition becomes experiment circuits, see
    :data:`ExpansionStrategy`. Under ``"auto"``, ``max_exact_groups`` is the largest
    exact group count still enumerated rather than sampled. When sampling,
    ``num_samples`` draws are taken, defaulting to ``max_exact_groups``, and ``seed``
    fixes the sampler for a reproducible experiment set.
    """

    consolidate: ConsolidateStrategy | bool = "auto"
    joint_rotation_cuts: bool = True
    expansion: ExpansionStrategy = "auto"
    max_exact_groups: int = 1000
    num_samples: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate the combination."""
        if self.consolidate not in ("auto", "always", "never", True, False):
            raise QCutError(
                f"unknown consolidate strategy '{self.consolidate}', expected one of "
                "'auto', 'always', 'never', True, False"
            )
        if self.expansion not in ("auto", "exact", "sample"):
            raise QCutError(
                f"unknown expansion strategy '{self.expansion}', expected one of "
                "'auto', 'exact', 'sample'"
            )
        if self.max_exact_groups < 1:
            raise QCutError("max_exact_groups must be at least 1")
        if self.num_samples is not None and self.num_samples < 1:
            raise QCutError("num_samples must be at least 1")

    @property
    def consolidate_mode(self) -> str:
        """The consolidation strategy, with ``True`` and ``False`` normalised."""
        if self.consolidate is True:
            return "always"
        if self.consolidate is False:
            return "never"
        return self.consolidate

    @property
    def sample_count(self) -> int:
        """Number of draws to use when sampling."""
        return (
            self.num_samples if self.num_samples is not None else self.max_exact_groups
        )

    def should_sample(self, exact_groups: int) -> bool:
        """Whether an experiment of ``exact_groups`` combinations should be sampled."""
        if self.expansion == "exact":
            return False
        if self.expansion == "sample":
            return True
        return exact_groups > self.max_exact_groups

    def replace(self, **changes) -> CutOptions:
        """Return a copy with ``changes`` applied."""
        return replace(self, **changes)


#: Used when no options are passed. Reassign to change the defaults process-wide.
DEFAULT_OPTIONS = CutOptions()


def resolve(options: CutOptions | None) -> CutOptions:
    """Return ``options``, falling back to :data:`DEFAULT_OPTIONS`."""
    if options is None:
        return DEFAULT_OPTIONS
    if not isinstance(options, CutOptions):
        raise QCutError(f"options must be a CutOptions, got {type(options).__name__}")
    return options
