Gate cuts
=========

Gate cuts can be used to cut multi-qubit gates instead of cutting wires. This is done by inserting special gate cut instructions into the circuit.
Currently only CZ gate cuts are supported so special care needs to be taken to transform other gates into CZ gates before inserting the cuts.

.. code:: python

   cut_circuit = QuantumCircuit(3)
   cut_circuit.h(0)
   cut_circuit.h(1)
   cut_circuit.append(cutCZ, [0,1])
   cut_circuit.h(1)
   cut_circuit.cx(1,2)

   cut_circuit.draw("mpl")

After this the circuit can be processed as usual with QCut (take a look at the Usage documentation for more details).