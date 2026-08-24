Usage
=====

QCut is a quantum circuit knitting package for performing wire cuts
especially designed to not use reset gates or mid-circuit since on early
NISQ devices they pose significant errors, if they are even available.

QCut has been designed to work with IQM’s qpus, and therefore on the
Finnish Quantum Computing Infrastructure
(`FiQCI <https://fiqci.fi/>`__), and tested with an IQM Adonis 5-qubit
qpu. Additionally, QCut is built to be compatible with IQM’s Qiskit fork iqm_qiskit.

QCut was built as a part of a summer internship at CSC - IT Center for
Science (Finnish IT Center for Science).

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

**6: Generate experiment circuits**

.. code:: python

   observables = SparsePauliOp(["IIIZ", "IIZI", "IZII", "IIZZ"])

   cut_experiment = ck.get_experiment_circuits(transpiled, observables)

**7: Run the experiment circuits**

.. code:: python

   results = ck.run_experiments(cut_experiment, backend=fake)

**8. Define observables and calculate expectation values**

Observables are Pauli-Z observables and are defined as a list of qubit
indices. Multi-qubit observables are defined as a list inside the
observable list.

If one wishes to calculate other than Pauli-Z observable expectation
values currently this needs to be done by manually modifying the initial
circuit to perform the basis transform.

.. code:: python

   observables = [0,1,2, [0,1]]
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

   from QCut import find_cuts

   cut_circuit = find_cuts(circuit , 3, cuts="both")

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
therefore hold Qiskit below 2.2; put it in an environment whose Qiskit is already in
range if you would rather the resolver left it alone.

The adapter used to be a separate `Qiskit
IQM <https://github.com/iqm-finland/qiskit-on-iqm>`__ package, capped at Qiskit 1.2.
Everything it supported is supported by the extra above, so it is only worth installing
directly if you need that older adapter's API:

.. code:: bash

   uv pip install qiskit-iqm==17.8

After installation just import the backend you want to use:

.. code:: python

   from iqm.qiskit_iqm import IQMFakeAdonis
   backend = IQMFakeAdonis()

Let QCut do the transpiling
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Transpile the subcircuits, before the experiment circuits are built. This is both
cheaper, since each subcircuit is translated once rather than once per group, and the
path that knows about the placeholders standing in for the cuts:

.. code:: python

   cut_circuit = ck.transpile_subcircuits(cut_circuit, backend, optimization_level=3)
   experiment = ck.get_experiment_circuits(cut_circuit, observables)

``transpile_experiments`` does the same job after the fact, if the experiment circuits
already exist. Either takes a ``transpile_options`` dict, passed through to Qiskit's
transpiler.

This works on star-topology devices such as Adonis, on Apollo, and on resonator devices
such as Deneb. Three things it has to take care of, which are easy to get wrong by hand:

- **The layout.** Transpiling lays a subcircuit out on physical qubits and pads it to
  the device width, so the qubit at index ``i`` afterwards is not the one that was there
  before. Expectation values are read by position, so the qubits are put back in their
  original order afterwards.
- **Borrowed wires.** If the layout puts two of a subcircuit's qubits somewhere the
  device does not connect, routing borrows a third wire to bridge them. That wire is
  kept, since the circuit needs it, but it holds nothing to measure.
- **Non-standard device gates.** A resonator machine lists a ``move`` operation, which
  Qiskit will not accept as a basis gate. It is not needed for translating a basis
  change, so it is left out.

IQM's own transpiler
~~~~~~~~~~~~~~~~~~~~

``transpile_subcircuits`` uses ``transpile_to_IQM`` by itself whenever the backend is an
IQM one and the adapter is installed, which is where the shallower circuits above come
from. Pass ``use_iqm_transpiler=False`` to force the ordinary Qiskit path instead.

Two of that function's defaults are inverted for QCut, and it is worth knowing why:

``remove_final_rzs=False``
    A Z rotation is a virtual frame on IQM hardware, and dropping a trailing one is
    harmless only if the qubit is then measured in the Z basis. QCut adds the basis
    rotations for X and Y observables *later*, so a frame removed now is one that should
    have been rotated then, and those expectation values come out wrong with nothing to
    indicate it.

``perform_move_routing=False``
    This is the step that turns a simplified-architecture circuit into a real Star
    architecture one, introducing the resonator and its MOVE gates. It rebuilds the
    classical registers on the way and loses ``qpd_meas`` wherever a term does not write
    to it, which invalidates the reconstruction. It also has to: IQM's own documentation
    describes going the other way, with ``transpile_remove_moves``, precisely so a
    circuit can be handled by tools that do not support the MOVE gate. QCut is one of
    those, so it works on the simplified architecture and the MOVEs are inserted
    afterwards, at submission, by ``iqm-client``. A move-routed circuit also cannot be
    checked against a local simulator at all, since Aer does not implement the gate.

Both can still be overridden through ``transpile_options``, which is passed on to
``transpile_to_IQM`` along with anything else it takes. Overriding these two will give
wrong answers.

The placeholders standing in for the cuts survive this because they are swapped for
barriers carrying their name as a label first, and swapped back afterwards. A transpiler
that builds its own target has no way of being told about a custom instruction and
refuses to synthesise it, while a barrier is a directive it passes through untouched.
This is also the more honest instruction to give it: a placeholder stands for an
operation that has not been chosen yet, so merging the gates on either side across the
gap would not be a valid simplification.

Transpiling before the cuts are marked
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The other way round works too, if you would rather IQM's transpiler saw the whole circuit
at once rather than each subcircuit:

.. code:: python

   from iqm.qiskit_iqm import transpile_to_IQM

   native = transpile_to_IQM(
       circuit,
       backend,
       remove_final_rzs=False,      # for the reason above
       perform_move_routing=False,
       optimization_level=3,
   )
   # insert the cut markers into `native`, then cut and run as usual

Two things to keep in mind on that path. The native circuit is padded to the device
width, so QCut sees the padding as ordinary qubits and splits accordingly, and results
have to be read back through ``native.layout.final_index_layout()`` to line up with the
qubits of the circuit you started from.

The same warning applies to any pre-transpiled input, whatever produced it: if a circuit
reached you already translated and stripped of its trailing Z frames, asking QCut for X
or Y observables will give wrong answers.

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
