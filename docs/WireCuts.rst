Wire cuts
=========

Wire cuts can be used to cut the wires of qubits in a circuit. This is done by inserting special cut instructions into the circuit.

.. code:: python

   cut_circuit = QuantumCircuit(3)
   cut_circuit.h(0)
   cut_circuit.cx(0,1)
   cut_circuit.append(cut, [1])
   cut_circuit.cx(1,2)

   cut_circuit.draw("mpl")

After this the circuit can be processed as usual with QCut (take a look at the Usage documentation for more details).