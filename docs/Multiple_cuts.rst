Multiple cuts
=============

Performing multiple cuts works the exact same way as a single cut. Since eats cut introduces an extra qubit one just has to 
be careful to properly allocate the extra qubits needed by wire-cutting.

Cutting to more than two parts
------------------------------

**1: Define the circuit we want to cut**

.. code:: python

    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0,1)
    circuit.cx(1,2)
    circuit.cx(2,3)
    circuit.measure_all()

    circuit.draw("mpl")

.. image:: _static/images/circ6.png

Now let's say we want to cut this circuit into three pieces with two cuts.

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

.. image:: _static/images/circ5.png

**3: Continue like with single cut**

.. code:: python

   backend = AerSimulator()
   observables = [0,1,2, [0,2]]

   estimated_expectation_values = ck.run(cut_circuit, observables, backend)

Click :download:`here <examples/QCutCutToThreeParts.ipynb>` to download example notebook.

More examples on how the cuts should be placed.
-----------------------------------------------

**1: Cutting two subsequent wires**

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

.. image:: _static/images/circ9.png

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

.. image:: _static/images/circ10.png

Click :download:`here <examples/QCutCutSubsequentWires.ipynb>` to download example notebook.

**2: Two consequent cuts on the same wire**

Initial circuit:

.. code:: python

    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0,1)
    circuit.cx(1,2)
    circuit.cx(0,1)
    circuit.measure_all()

    circuit.draw("mpl")

.. image:: _static/images/circ7.png

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

.. image:: _static/images/circ8.png

Click :download:`here <examples/QCutMultipleCutsOnSingleWire.ipynb>` to download example notebook.
