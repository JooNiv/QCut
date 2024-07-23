Multiple cuts
============

Performing multiple cuts works the exact same way as a single cut. One just has to be careful to 
properly allocate the extra qubis needed by wire-cutting.

**1: Insert cut_wire operations to the circuit to denote where we want
to cut the circuit**

Note that here we don’t insert any measurements. Measurements will be
automatically handled by QCut.

.. code:: python

   cut_circuit = QuantumCircuit(4)
   cut_circuit.h(0)
   cut_circuit.cx(0,1)
   cut_circuit.append(cut_wire, [1,2])
   cut_circuit.cx(2,3)

   cut_circuit.draw("mpl")

.. image:: /images/circ5.png

**2: Continue like with single cut**

.. code:: python

   backend = AerSimulator()
   observables = [0,1,2, [0,2]]
   error = 0.03

   estimated_expectation_values = ck.run(cut_circuit, observables, error, backend)