Basic Usage
=====

QCut is a quantum circuit knitting package capable of efficiently partitioning
quantum circuits with wire and gate cuts using advanced LOCC and joint rotation decompositions
along with the standard local decompositions. It is designed to be compatible with Qiskit and should
be compatible with any Qiskit programmable backend but has been especially designed to be compatible
with IQM’s qpus and the Finnish Quantum Computing Infrastructure (`FiQCI <https://fiqci.fi/>`__).

QQCut has been built at CSC - IT Center for Science (Finnish IT Center for Science)

Installation
------------

For installation a UNIX-like system is currently needed due to pymetis being used for automatic cut finding. On Windows use WSL

| **Pip:**
| Installation should be done via ``uv``

.. code:: bash

   uv pip install QCut
   #or
   uv add QCut

If using other than the default Qiskit version (newest) it is recommended to install Qiskit first before installing QCut.

Using uv is the recommended install method.

| **Install from source**
| It is also possible to use QCut by cloning this repository and
  including it in your project folder.

Basic usage
-----------

**1: Import needed packages**

.. code:: python

   import QCut as ck
   from QCut import cut, cutGate
   from qiskit import QuantumCircuit, transpile
   from qiskit.circuit.library import CXGate
   from qiskit.quantum_info import SparsePauliOp
   from qiskit.circuit.library import CXGate
   from qiskit_aer import AerSimulator
   from qiskit.primitives import StatevectorEstimator, BackendEstimatorV2 as BackendEstimator
   from iqm.qiskit_iqm import IQMFakeAdonis

**2: Start by defining a QuantumCircuit just like in Qiskit**

.. code:: python

   circuit  =  QuantumCircuit(4)

   mult = 1.635
   circuit.r(mult*0.46262, mult*0.1446, 0)
   circuit.cx(0,1)
   circuit.cx(1,2)
   circuit.cx(2,3)
      
   circuit.draw("mpl")

.. image:: _static/images/circ1.png

**3: Insert cuts to the circuit to denote where we want
to cut the circuit**

Note that here we don’t insert any measurements. Measurements will be
automatically handled by QCut.

.. code:: python
   
   from qiskit.circuit.library import CXGate

   cut_circuit = QuantumCircuit(4)

   mult = 1.635
   cut_circuit.r(mult*0.46262, mult*0.1446, 0)
   cut_circuit.append(**cutGate(CXGate(), 0, 1)) 
   cut_circuit.append(cut(), [1])
   cut_circuit.cx(1,2)
   cut_circuit.cx(2,3)

   cut_circuit.draw("mpl")

.. image:: _static/images/circ2.png

**Note** that currently QCut only supports cutting CZ, SWAP, and iSWAP gates so all two qubit gates get decomposed to them, hence some cuts resulting in extra gates.

**4. Extract cut locations from cut_circuit and split it into
independent subcircuit.**

.. code:: python

   cut_circuit = ck.get_locations_and_subcircuits(cut_circuit)

Now we can draw our subcircuits.

.. code:: python

   cut_circuit.subcircuits[0].draw("mpl")

.. image:: _static/images/circ3.png

.. code:: python

   cut_circuit.subcircuits[1].draw("mpl")

.. image:: _static/images/circ4.png

.. code:: python

   cut_circuit.subcircuits[2].draw("mpl")

.. image:: _static/images/circ11.png

**5 Define backend and transpile the cut circuit**

.. code:: python

   fake = IQMFakeAdonis() #noisy
   sim = AerSimulator() #ideal

   transpiled = ck.transpile_subcircuits(cut_circuit, fake, optimization_level=3)

**6: Define observables and Generate experiment circuits**

   Observables are defined using the SparsePauliOp class from Qiskit. The expectation values are then 
   calculated using the estimate_expectation_values function from QCut.

.. code:: python

   observables = SparsePauliOp(["IIIZ", "IIZI", "IZII", "IIZZ"])

   cut_experiment = ck.get_experiment_circuits(transpiled, observables)

**7: Run the experiment circuits**

.. code:: python

   results = ck.run_experiments(cut_experiment, backend=fake)

**8. Calculate expectation values**

.. code:: python

   expectation_values = ck.estimate_expectation_values(results)

**9: Finally calculate the exact expectation values and compare them to
the results calculated with QCut**

.. code:: python

   obs = [ob.to_label() for ob in observables.paulis]

   estimator = Estimator()
   exact_expvals = [e.data.evs for e in
      estimator.run([(x) for x in zip([circuit] * len(obs), obs)]).result()
   ]


   tr = transpile(circuit, backend=fake)

   tr_obs = observables.apply_layout(tr.layout)

   tr_obs_separate = [
      SparsePauliOp(pauli.to_label()) for pauli in tr_obs.paulis
   ]

   fake_estimator = BackendEstimator(backend=fake)
   exps = [e.data.evs for e in
      fake_estimator.run([(x) for x in zip([tr] * len(tr_obs_separate), tr_obs_separate)]).result()
   ]

.. code:: python

   import numpy as np

   np.set_printoptions(formatter={"float": lambda x: f"{x:0.6f}"})

   print(f"QCut expectation values:{np.array(expectation_values)}")
   print(f"Noisy expectation values with fake backend:{np.array(exps)}")
   print(f"Exact expectation values with ideal simulator :{np.array(exact_expvals)}")

``QCut expectation values:[0.704039 0.615275 0.554269 0.808868]``

``Noisy expectation values with fake backend:[0.587891 0.669922 0.500000 0.777344]``

``Exact expectation values with ideal simulator :[0.727323 0.727323 0.727323 1.000000]``

As we can see QCut is able to accurately reconstruct the expectation values and be more accurate that just using the fake backend as is. (Note that since this is a probabilistic method the results vary a bit each run)

Additionally we can execute QCut using the ideal Aer simulator and see that we get (practically) exact results:

``QCut expectation values:[0.699436 0.713172 0.713172 0.979377]``


Click :download:`here <examples/QCutBasicUsage.ipynb>` to download example notebook.


Basic usage shorthand
---------------------

For convenience, it is not necessary to go through each of the
aforementioned steps individually. Instead, QCut provides a function
``run()`` that executes the whole wire-cutting sequence.

The same example can then be run like this:

.. code:: python

   sim = AerSimulator()
   observables = SparsePauliOp(["IIIZ", "IIZI", "IZII", "IIZZ"])

   estimated_expectation_values = ck.run(cut_circuit, observables, sim)

Automatic cuts
--------------

QCut comes with functionality for automatically finding good cut locations that can place both wire and gate cuts.

.. code:: python

   from QCut import find_cuts, CutOptions

   options = CutOptions(
      finder_num_partitions=3,
      finder_cut_mode="both",
   )

   cut_circuit = find_cuts(circuit , options=options)

   estimated_expectation_values = ck.run_cut_circuit(cut_circuit, observables, sim)

   np.set_printoptions(formatter={"float": lambda x: f"{x:0.6f}"})

   print(f"QCut expectation values:{np.array(estimated_expectation_values)}")
   print(f"Exact expectation values with ideal simulator :{np.array(exact_expvals)}")


``QCut expectation values:[0.729648 0.745609 0.702871 0.992620]``

``Exact expectation values with ideal simulator :[0.727323 0.727323 0.727323 1.000000]``

Running on IQM fake backends
----------------------------

QCut ships an extra for this, so the adapter comes with it:

.. code:: bash

   uv pip install "QCut[iqm]"

That installs `IQM client <https://docs.meetiqm.com/iqm-client>`__ with its Qiskit
adapter, which covers Qiskit 1.0 up to but not including 2.2. Installing the extra will
therefore hold Qiskit below 2.2, put it in an environment whose Qiskit is already in
range if you would rather the resolver left it alone.

After installation just import the backend you want to use:

.. code:: python

   from iqm.qiskit_iqm import IQMFakeAdonis
   backend = IQMFakeAdonis()

Running on FiQCI
----------------

For running on real hardware through the Lumi supercomputer’s FiQCI
partition follow the instructions
`here <https://docs.csc.fi/computing/quantum-computing/helmi/running-on-helmi/>`__.
If you are used to using Qiskit on jupyter notebooks it is recommended
to use the `Lumi web
interface <https://docs.lumi-supercomputer.eu/runjobs/webui/>`__.

Running on other hardware
-------------------------

Running on other providers such as IBM is untested at the moment but as
long as the hardware can be accessed with Qiskit version > 1.0 the QCut
should be compatible.
