Other than Z observables
========================

Currently the observables passed to methods in QCut are always Z observables.
If one wants to calculate expectation values in some other basis, like the X basis for example, this needs to be done by manually modifying the circuit.

Further, if one wants to calculate expectation values on multiple bases for a single qubit, multiple experiment runs with different circuit are needed.

This may change in a future release.

**1: X observable expectation values**

.. code:: python

    circuit_x = QuantumCircuit(3)
    circuit_x.h(0)
    circuit_x.cx(0,1)
    circuit_x.cx(1,2)
    circuit_x.h(2)
    circuit_x.measure_all()

    circuit_x.draw("mpl")

.. code:: python

    cut_circuit_x = QuantumCircuit(4)
    cut_circuit_x.h(0)
    cut_circuit_x.cx(0,1)
    cut_circuit_x.append(cut_wire, [1,2])
    cut_circuit_x.cx(2,3)
    cut_circuit_x.h(3)

    cut_circuit_z.draw("mpl")

.. code:: python

    observables = [0, 1, 2]

We have manually transformed qubit 2 to the X basis so now the expectation value returned by QCut will also be the X observable expectation value.
The other qubits are stull in the Z basis so the expectation value will be the Z observable expectation value.

**2: Z and X observable expectation values for a single qubit**

In addition to the circuits above let's define the "normal" Z basis circuits.

.. code:: python

    circuit_z = QuantumCircuit(3)
    circuit_z.h(0)
    circuit_z.cx(0,1)
    circuit_z.cx(1,2)
    circuit_z.measure_all()

    circuit_z.draw("mpl")

.. code:: python

    cut_circuit_z = QuantumCircuit(4)
    cut_circuit_z.h(0)
    cut_circuit_z.cx(0,1)
    cut_circuit_z.append(cut_wire, [1,2])
    cut_circuit_z.cx(2,3)

    cut_circuit_z.draw("mpl")

.. code:: python

    observables_x = [2]

Now to get expectation values for both X and Z basis for qubit 2 the QCut circuit knitting sequence can be executed on both Z and X basis circuits.

