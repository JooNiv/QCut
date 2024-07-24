Multiple cuts
============

Performing multiple cuts works the exact same way as a single cut. Since eats cut introduces an extra qubit one just has to 
be careful to properly allocate the extra qubits needed by wire-cutting.

**1: Define the circuit we want to cut**

.. code:: python
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0,1)
    circuit.cx(1,2)
    circuit.cx(2,3)
    circuit.measure_all()

    circuit.draw("mpl")

Now let's say we want to cut this circuit into two pieces with two cuts.

**2: Insert cut_wire operations to the circuit to denote where we want
to cut the circuit**

.. code:: python

   cut_circuit = QuantumCircuit(6)
   cut_circuit.h(0)
   cut_circuit.cx(0,1)
   cut_circuit.append(cut_wire, [1,2])
   cut_circuit.cx(2,3)
   cut_circuit.append(cut_wire, [3,4])
   cut_circuit.cx(4,5)

   cut_circuit.draw("mpl")

.. image:: /images/circ5.png

**3: Continue like with single cut**

.. code:: python

   backend = AerSimulator()
   observables = [0,1,2, [0,2]]
   error = 0.03

   estimated_expectation_values = ck.run(cut_circuit, observables, error, backend)

More examples on how the cuts should be placed.
------------

**1: Cutting two consequent wires**

Initial circuit:

.. code:: python
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0,1)
    circuit.cx(0,2)
    circuit.cx(1,2)
    circuit.cx(1,3)
    circuit.measure_all()

    circuit.draw("mpl")

Circuit with cuts:

.. code:: python
    cut_circuit = QuantumCircuit(6)
    cut_circuit.h(0)
    cut_circuit.cx(0,1)
    cut_circuit.cx(0,2)
    cut_circuit.append(cut_wire, [1,3])
    cut_circuit.append(cut_wire, [2,4])
    cut_circuit.cx(3,4)
    cut_circuit.cx(3,5)

    cut_circuit.draw("mpl")

**2: Two cosequent cuts on same wire**

Under construction. Once done the cuts should be placed as follows:

Initial cirucit:

.. code:: python
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0,1)
    circuit.cx(1,2)
    circuit.cx(0,1)
    circuit.measure_all()

    circuit.draw("mpl")

Circuit with cuts:

.. code:: python
    cut_circuit = QuantumCircuit(5)
    cut_circuit.h(0)
    cut_circuit.cx(0,1)
    cut_circuit.append(cut_wire, [1,3])
    cut_circuit.cx(3,4)
    cut_circuit.append(cut_wire, [3,2])
    cut_circuit.cx(0,2)

    cut_circuit.draw("mpl")
