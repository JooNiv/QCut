Options
=======

A cutting run is configured with a :class:`~QCut.options.CutOptions`. Pass one to
``find_cuts``, ``get_locations_and_subcircuits`` or ``run`` and it is carried on the
resulting ``CutCircuit`` and ``CutExperiment``, so it only has to be given once.

.. code:: python

   from QCut import CutOptions, find_cuts

   cut_circuit = find_cuts(circuit, num_partitions=2,
                           options=CutOptions(consolidate=False))

Options are fixed before the subcircuits are built, because they decide how the
quasiprobability decomposition is formed and therefore what its coefficients mean. A
``CutOptions`` is frozen for that reason. Use ``options.replace(...)`` for a modified copy,
or reassign ``QCut.DEFAULT_OPTIONS`` to change the defaults process-wide.

Merging gates on the same qubit pair
------------------------------------

``consolidate`` (default ``"auto"``) merges runs of gates acting on the same qubit pair
into one gate before cutting. If nothing else touches those qubits in between, the whole
run is itself one two-qubit unitary, and cutting it once is usually cheaper than cutting
each gate, because the overhead of separate cuts multiplies.

Two ``rzz(0.4)`` gates on a pair cost :math:`\gamma = 3.16` over 36 subexperiments cut
separately, against :math:`\gamma = 2.43` over 6 as the single ``rzz(0.8)`` they compose
to. Two ``cz`` gates compose to the identity, so the merged cut costs
:math:`\gamma = 1`.

Merging is not always the cheaper choice once joint cutting is in play. A run of gates
about *different* axes composes to a generic two-qubit unitary, which is no longer a
single-axis rotation and so can no longer join a joint decomposition. Cutting the gates
separately and bundling each with its parallel partners can beat merging them. In a
randomised search over 550 layered circuits this happened 18 times, costing up to 11% in
:math:`\gamma` and so about 23% in shots.

There is no way to tell which wins from one pair alone, so the three strategies are

``"auto"``
    Split the circuit both ways, cost each plan in full including the bundles it allows,
    and keep the cheaper. The comparison is exact rather than a heuristic, and it was
    optimal in every trial of the search above. The extra work only happens when there
    was something to merge in the first place.

``"always"`` (or ``True``)
    Merge wherever it lowers the cost of that pair considered on its own. This was the
    old behaviour.

``"never"`` (or ``False``)
    Leave every gate alone.

Under ``"auto"``, ``find_cuts`` runs the whole search twice, because consolidating
changes which edges the partitioner sees and the two plans can end up cutting different
gates entirely. Both paths log which way they went and what it saved at INFO level.

A run holding only one two-qubit gate is left alone, since absorbing the surrounding
single-qubit gates cannot change :math:`\gamma` and would only replace a named gate by a
generic unitary.

Note that merging can turn two separately placed cut markers on one pair into a single
cut, which changes ``len(cut_locations)``. That is intended. QCut already requires that
if one gate on a pair is cut then every gate on that pair is cut, so the gates in a run
all have to be cut anyway.

The pass is also available on its own as
:func:`~QCut.consolidate.consolidate_two_qubit_blocks`, for running on a circuit before
placing cuts by hand.

Cutting parallel rotation gates together
----------------------------------------

``joint_rotation_cuts`` (default ``True``) cuts several parallel two-qubit rotation gates
with one decomposition instead of one each. Separate cuts cost the product of their
overheads, which is not optimal for gates equivalent to a rotation about a single Cartan
axis. Cutting :math:`n` of them together costs
:math:`2\prod_s (1 + |\sin\theta_s|) - 1` against
:math:`\prod_s (1 + 2|\sin\theta_s|)`.

Two CNOT gates therefore cost :math:`\gamma = 7` over 30 subexperiments rather than
:math:`\gamma = 9` over 36, and three cost :math:`\gamma = 15` over 132 rather than
:math:`\gamma = 27` over 216. Shot count goes as :math:`\gamma^2`, so three parallel
gates get about three times cheaper and four about seven times. Both the overhead and the
circuit count fall, so there is nothing to trade off, which is why it defaults to on.

The gates have to be equivalent to a single-axis rotation, which covers ``rzz``, ``rxx``,
``ryy``, ``rzx``, the controlled rotations, ``cp``, ``cx``, ``cz`` and ``ecr`` but not
``swap``, ``iswap``, ``dcx`` or ``xx_plus_yy``. They also have to be parallel and to join
the same pair of subcircuits, because each side of the decomposition acts on several
qubits of one subcircuit at once. Anything that does not qualify is cut on its own as
before, so turning this on can only help. ``QCut.bundle.plan_bundles`` reports what it
grouped and what that saved at INFO level.

Note that this changes the number of experiment groups, so a test pinning
``CutExperiment.num_groups`` will see 30 where it saw 36. The derivation is on the
:doc:`joint rotation gate cutting <theory/Joint_rotation_derivation>` page.

Sampling instead of enumerating
-------------------------------

Building every combination of QPD terms costs the product of the per-cut term counts. That
is fine for one or two cuts and hopeless beyond that, since a generic two-qubit gate needs
58 terms. ``expansion`` controls what happens instead.

``"auto"`` (the default) enumerates while the exact count is within ``max_exact_groups``
(default 1000) and samples above it. ``"exact"`` always enumerates, ``"sample"`` always
samples.

When sampling, terms are drawn per cut with probability proportional to :math:`|c|`, and
``num_samples`` draws are taken (defaulting to ``max_exact_groups``). Repeated draws are
collapsed, so drawing 500 times from a six term decomposition still builds six circuits.
Set ``seed`` for a reproducible experiment set.

Sampling is not only a way to make large cases tractable. Enumerating gives every group the
same number of shots and corrects with a weight afterwards, which is higher variance than
drawing proportional to :math:`|c|` in the first place. The estimator is unbiased either
way, and ``CutExperiment.sampled`` says which path was taken.
