Automatic cuts
==============

QCut comes with functionality for automatically finding good cut locations that can place both wire and gate cuts.
The number of partitions, the size of the partitions, and the type of cuts can be specified.

Under the hood QCut uses `pymetis <https://github.com/inducer/pymetis>`__  to find good cut locations based on the circuit's graph representation.

.. code:: python

   cut_circuit = find_cuts(circuit , 3, cuts="both")
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
``find_cuts`` still applies joint cutting to whatever cuts it settles on, it just does
not steer the partitioner towards cut sets that would bundle well.
