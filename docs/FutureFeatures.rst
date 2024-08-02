Possible future features
========================

This list is not a roadmap but simpy highlights some interesting features that could be added in the future
depending on time constraints.

Streamline calculating other than Z observable expectation values
-----------------------------------------------------------------

Make it possible to define observables as an dictionary where keys correspond to a Pauli observable
and values to qubit indices, like:

.. code:: python

    observables = {'z': [[0,1,2], [0,2]], 'x': [0,1]}

This could then be passed on as in the current version but would automatically calculate all the needed expectation values.
Note that just like in the current implelemtation this would cause multiple experiment runs behind the scenes and would just
streamline the user experience.

Automatic allocation for extra qubits
-------------------------------------

Allow cuts to be placed without manually inserting extra qubits to the circuit and instead have it be done automatically.

Likely first implelemtation would be just for the case when there are multiple cuts on a single wire. This way the current way:

.. code:: python

    cut_circuit = QuantumCircuit(5)
    cut_circuit.h(0)
    cut_circuit.cx(0,1)
    cut_circuit.append(cut_wire, [1,3])
    cut_circuit.cx(3,4)
    cut_circuit.append(cut_wire, [3,2])
    cut_circuit.cx(0,2)

    cut_circuit.draw("mpl")

would simplify to:

.. code:: python

    cut_circuit = QuantumCircuit(4)
    cut_circuit.h(0)
    cut_circuit.cx(0,1)
    cut_circuit.append(cut_wire, [1,2])
    cut_circuit.cx(2,3)
    cut_circuit.append(cut_wire, [2,1])
    cut_circuit.cx(0,1)

    cut_circuit.draw("mpl")

However behind the scenes this would still be transformed so that each cut moves following operations onto a new wire,
which could possibly cause confusion.

Support for cutting with reset gates
------------------------------------

Currently not an important feature since reset gates pose large errors but could be implemented in the future.
Allowing reset gates would reduce the number of extra qubits needed.