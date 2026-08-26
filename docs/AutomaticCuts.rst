Automatic cuts
==============

QCut comes with functionality for automatically finding good cut locations that can place both wire and gate cuts.
The number of partitions, the size of the partitions, and the type of cuts can be specified.

Under the hood QCut uses `pymetis <https://github.com/inducer/pymetis>`__  to find good cut locations based on the circuit's graph representation.

.. code:: python

   from QCut import find_cuts, CutOptions

   options = CutOptions(
      finder_num_partitions=3,
      finder_cut_mode="both",
   )

   cut_circuit = find_cuts(circuit , options=options)

   estimated_expectation_values = ck.run_cut_circuit(cut_circuit, observables, backend)

How cuts are costed
-------------------

METIS minimises the **sum** of the weights of the edges it cuts, but the sampling
overhead of a set of cuts is the **product** of their :math:`\gamma` values. Taking the
logarithm turns one into the other, so each edge is weighted by
:math:`\log\gamma` rather than by :math:`\gamma`, and the partitioner then optimises
the cost that actually matters.

Weighting by :math:`\gamma` directly, as QCut used to, systematically undervalues cheap
cuts. Two ``cz`` cuts cost :math:`\gamma = 9` and one ``swap`` cut costs
:math:`\gamma = 7`, so the ``swap`` is cheaper, yet adding gammas says the opposite
because :math:`3 + 3 < 7`.

A wire cut weighs :math:`\log 4`, and a gate cut weighs the logarithm of its own
:math:`\gamma`, so an ``rzz`` with a small angle is correctly treated as nearly free.
A cut whose :math:`\gamma` is exactly 1 costs no shots at all, but it still adds
circuits, so its weight floors at one rather than zero.

One thing the weights cannot express is that parallel rotation gates get cheaper when
cut together, since that is a property of a set of edges rather than of any one edge.
Choosing between candidate partitions
--------------------------------------

METIS minimises the weighted edge cut. With ``log gamma`` weights that is exactly the
true cost of a plan, right up until two cuts share a decomposition: a joint rotation
bundle or a communicating block of wires costs less than the product of its parts, and
METIS cannot know that. It also returns only the partitioning that scored best by its
own measure, discarding the rest.

So the finder asks for one partitioning per seed, builds each one out in full, and costs
the finished plans with :func:`~QCut.qpd_operations.plan_cost`, which does see the
bundles. The cheapest wins. ``finder_candidates`` (default 5) sets how many to try, and
because the seeds are consecutive a larger set contains the smaller one, so raising it
cannot give a worse answer.

The seeds are fixed. Before that they were drawn at random, which made the finder
non-deterministic: two runs on the same circuit could return plans whose overheads
differed by three orders of magnitude, and there was no way to get the good one back.
``seed`` shifts the whole candidate set if you want a different draw.

Meeting a qubit budget
-----------------------

A budget has to be asked for rather than repaired for. The graph METIS partitions has
one node per wire segment, so several nodes belong to one qubit and balancing node
counts says nothing about how many qubits a partition ends up holding. Each of a
qubit's nodes therefore carries an equal share of one qubit's worth, which makes a
partition's weight its qubit count, and ``max_qubits`` becomes the share each partition
may hold. A straddling qubit counts in both, which is what a wire cut costs anyway.

Without that, the partition was chosen with no regard for the budget and then repaired
to fit by moving whole qubits across, paying in cuts for each one. On nearest-neighbour
circuits at 16 to 24 qubits, where the best balanced split is simply the contiguous one,
that repair cost between 17 and 600 times the sampling overhead of the optimum. Asking
the partitioner for the right thing finds the optimum outright.

The weighting only applies when there is a budget. With none, an unbalanced split is
usually cheaper, so forcing balance would make that case worse. The repair still runs as
a fallback for whatever the partitioner cannot satisfy.

``find_cuts`` still applies joint cutting to whatever cuts it settles on, it just does
not steer the partitioner towards cut sets that would bundle well.
