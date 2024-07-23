Examples
============

More advanced examples on how QCut works.

Multiple cuts
-----

**1: Import needed packages**

.. code:: python

   import QCut as ck
   from QCut import cut_wire
   from qiskit import QuantumCircuit
   from qiskit_aer import AerSimulator
   from qiskit_aer.primitives import Estimator