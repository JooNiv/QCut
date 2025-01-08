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

Support for cutting with reset gates
------------------------------------

Currently not an important feature since reset gates pose large errors but could be implemented in the future.
Allowing reset gates would reduce the number of extra qubits needed and allow gate cuts.