Gate cuts
=========

Gate cuts can be used to cut two-qubit gates instead of cutting wires. This is done by inserting special gate cut instructions into the circuit.

**Any** two-qubit gate can be cut. CZ, SWAP and iSWAP use hand-derived decompositions. For every other gate a quasiprobability decomposition is generated from the gate's own matrix via its KAK decomposition (see :doc:`Theory`). Gates on more than two qubits are still transpiled down first, which turns them into several cuts.

Generating the decomposition rather than transpiling to CZ matters most for parametrised gates, whose sampling overhead depends on the angle. ``rzz(0.3)`` costs :math:`\gamma = 1 + 2|\sin\theta| \approx 1.59` over 6 subexperiments as itself, against :math:`\gamma = 9` over 36 subexperiments as two CZ cuts. The shot count scales as :math:`\gamma^2`, so that is a factor of roughly 32.

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