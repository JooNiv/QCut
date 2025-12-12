Gate cuts
=========

Gate cuts can be used to cut two-qubit gates instead of cutting wires. This is done by inserting special gate cut instructions into the circuit.
Currently only CZ gate cuts are supported so all cut gates get transformed into cut CZ gates with appropriate transpilation.

.. code:: python
   
   from qiskit.circuit.library import CXGate
   from qiskit import QuantumCircuit
   from QCut import cutGate

   cut_circuit = QuantumCircuit(3)
   cut_circuit.h(0)
   cut_circuit.append(**cutGate(CXGate(), 0, 1)) 
   cut_circuit.cx(1,2)

   cut_circuit.decompose(["CutGate"]).draw("mpl")

After this the circuit can be processed as usual with QCut (take a look at the Usage documentation for more details).