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

``consolidate`` (default ``True``) merges runs of gates acting on the same qubit pair into
one gate before cutting. If nothing else touches those qubits in between, the whole run is
itself one two-qubit unitary, and cutting it once is cheaper than cutting each gate,
because the overhead of separate cuts multiplies.

Two ``rzz(0.4)`` gates on a pair cost :math:`\gamma = 3.16` over 36 subexperiments cut
separately, against :math:`\gamma = 2.43` over 6 as the single ``rzz(0.8)`` they compose
to. Two ``cz`` gates compose to the identity, so the merged cut costs
:math:`\gamma = 1`.

A guard compares both costs and keeps the cheaper, so merging never makes a circuit more
expensive. A run holding only one two-qubit gate is left alone, since absorbing the
surrounding single-qubit gates cannot change :math:`\gamma` and would only replace a named
gate by a generic unitary.

Note that this can merge two separately placed cut markers on one pair into a single cut,
which changes ``len(cut_locations)``. That is intended. QCut already requires that if one
gate on a pair is cut then every gate on that pair is cut, so the gates in a run all have
to be cut anyway.

The pass is also available on its own as
:func:`~QCut.consolidate.consolidate_two_qubit_blocks`, for running on a circuit before
placing cuts by hand.

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
