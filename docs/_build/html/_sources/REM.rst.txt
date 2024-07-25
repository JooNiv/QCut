Readout error mitigation
============

QCut comes with an optional readout error mitigation flag for ``run_experiments()``.
To use readout error mitigation one only needs to set ``mitgate=True``.

Now the mitigated circuit knitting sequence can be executed with:

.. code:: python

   backend = AerSimulator()
   observables = [0,1,2, [0,2]]
   error = 0.03

   estimated_expectation_values = ck.run(cut_circuit, observables, error, backend, mitigate=True)
