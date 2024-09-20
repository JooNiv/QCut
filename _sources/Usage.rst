Usage
=====

QCut is a quantum circuit knitting package for performing wire cuts
especially designed to not use reset gates or mid-circuit since on early
NISQ devices they pose significant errors, if they are even available.

QCut has been designed to work with IQM’s qpus, and therefore on the
Finnish Quantum Computing Infrastructure
(`FiQCI <https://fiqci.fi/>`__), and tested with an IQM Adonis 5-qubit
qpu. Additionally, QCut is built on top of Qiskit 0.45.3 which is the
current supported Qiskit version of IQM’s Qiskit fork iqm_qiskit.

QCut was built as a part of a summer internship at CSC - IT Center for
Science (Finnish IT Center for Science).

Installation
------------

| **Pip:**
| Installation should be done via ``pip``

.. code:: python

   pip install QCut

Using pip is the recommended install method.

| **Install from source**
| It is also possible to use QCut by cloning this repository and
  including it in your project folder.

Basic usage
-----------

**1: Import needed packages**

.. code:: python

   import QCut as ck
   from QCut import cut_wire
   from qiskit import QuantumCircuit
   from qiskit_aer import AerSimulator
   from qiskit_aer.primitives import Estimator

**2: Start by defining a QuantumCircuit just like in Qiskit**

.. code:: python

   circuit  =  QuantumCircuit(3)
   circuit.h(0)
   circuit.cx(0,1)
   circuit.cx(1,2)
      
   circuit.measure_all()

   circuit.draw("mpl")

.. image:: _static/images/circ1.png

**3: Insert cut_wire operations to the circuit to denote where we want
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

.. image:: _static/images/circ2.png

**4. Extract cut locations from cut_circuit and split it into
independent subcircuit.**

.. code:: python

   cut_locations, subcircuits = ck.get_locations_and_subcircuits(cut_circuit)

Now we can draw our subcircuits.

.. code:: python

   subcircuits[0].draw("mpl")

.. image:: _static/images/circ3.png

.. code:: python

   subcircuits[1].draw("mpl")

.. image:: _static/images/circ4.png

**5: Generate experiment circuits by inserting operations from a
quasi-probability distribution for the identity channel**

.. code:: python

   experiment_circuits, coefficients, id_meas = ck.get_experiment_circuits(subcircuits, cut_locations)

**6: Run the experiment circuits**

Here we are using the qisit AerSimulator as our backend but since QCut
is backend independent you can choose whatever backend you want as long
as you transpile the experiment circuits accordingly. QCut provides a
function ``transpile_experiments()`` for doing just this.

Since QCut is a circuit knitting package the results are approximations
of the actual values.

.. code:: python

   backend = AerSimulator()
   results = ck.run_experiments(experiment_circuits, cut_locations, id_meas, backend=backend)

**7. Define observables and calculate expectation values**

Observables are Pauli-Z observables and are defined as a list of qubit
indices. Multi-qubit observables are defined as a list inside the
observable list.

If one wishes to calculate other than Pauli-Z observable expectation
values currently this needs to be done by manually modifying the initial
circuit to perform the basis transform.

.. code:: python

   observables = [0,1,2, [0,2]]
   expectation_values = ck.estimate_expectation_values(results, coefficients, cut_locations, observables)

**8: Finally calculate the exact expectation values and compare them to
the results calculated with QCut**

.. code:: python

   paulilist_observables = ck.get_pauli_list(observables, 3)

   estimator = Estimator(run_options={"shots": None}, approximation=True)
   exact_expvals = (
       estimator.run([circuit] * len(paulilist_observables),  # noqa: PD011
                     list(paulilist_observables)).result().values
   )

.. code:: python

   import numpy as np

   np.set_printoptions(formatter={"float": lambda x: f"{x:0.6f}"})

   print(f"QCut expectation values:{np.array(expectation_values)}")
   print(f"Exact expectation values with ideal simulator :{np.array(exact_expvals)}")

``QCut circuit knitting expectation values: [0.007532 0.007532 -0.003662 1.010128]``

``Exact expectation values with ideal simulator :[0.000000 0.000000 0.000000 1.000000]``

As we can see QCut is able to accurately reconstruct the expectation
values. (Note that since this is a probabilistic method the results vary
a bit each run)

Click :download:`here <examples/QCutBasicUsage.ipynb>` to download example notebook.


Basic usage shorthand
---------------------

For convenience, it is not necessary to go through each of the
aforementioned steps individually. Instead, QCut provides a function
``run()`` that executes the whole wire-cutting sequence.

The same example can then be run like this:

.. code:: python

   backend = AerSimulator()
   observables = [0,1,2, [0,2]]

   estimated_expectation_values = ck.run(cut_circuit, observables, backend)

Running on IQM fake backends
----------------------------

To use QCut with IQM’s fake backends it is required to install `Qiskit
IQM <https://github.com/iqm-finland/qiskit-on-iqm>`__. QCut supports
version 13.7. Installation can be done with pip:

.. code:: python

   pip install qiskit-iqm

After installation just import the backend you want to use:

.. code:: python

   from iqm.qiskit_iqm import IQMFakeAdonis()
   backend = IQMFakeAdonis()

To tranpile experiment circuits to the backend one can either manually call qiskit
transpile in a loop or use QCut's ``transpile_experiments()`` function:

.. code:: python

   transpiled_experiments = ck.transpile_experiments(experiment_circuits, backend)

Now one can proceed like before.

Running on FiQCI
----------------

For running on real IQM hardware through the Lumi supercomputer’s FiQCI
partition follow the instructions
`here <https://docs.csc.fi/computing/quantum-computing/helmi/running-on-helmi/>`__.
If you are used to using Qiskit on jupyter notebooks it is recommended
to use the `Lumi web
interface <https://docs.lumi-supercomputer.eu/runjobs/webui/>`__.

Running on other hardware
-------------------------

Running on other providers such as IBM is untested at the moment but as
long as the hardware can be accessed with Qiskit version < 1.0 the QCut
should be compatible.
