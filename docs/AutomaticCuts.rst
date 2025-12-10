Automatic cuts
==============

QCut comes with functionality for automatically finding good cut locations that can place both wire and gate cuts.
The number of partitions, the size of the partitions, and the type of cuts can be specified.

Under the hood QCut uses `pymetis <https://github.com/inducer/pymetis>`__  to find good cut locations based on the circuit's graph representation.

.. code:: python

   cut_locations, subcircuits, map_qubit = find_cuts(circuit , 3, cuts="both")
   estimated_expectation_values = ck.run_cut_circuit(subcircuits, cut_locations, observables, map_qubit, backend)