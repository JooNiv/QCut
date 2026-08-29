=========
Changelog
=========

**Version 2.1.0**
=================

Reconstructing a probability distribution
-----------------------------------------

- :code:`get_experiment_circuits()` takes :code:`qubits` in place of
  :code:`observables`, and :code:`estimate_probabilities()` then reconstructs the
  distribution over those qubits by an inverse Walsh-Hadamard transform. Costs
  :code:`2**k` observables but no extra circuits, since they share one measurement
  setting.
  See `the derivation <https://jooniv.github.io/QCut/theory/Probability_reconstruction.html>`__.
- The result is a :code:`dict` of quasi-probabilities, which can be negative.
  :code:`nearest_probabilities()` gives the closest true distribution and
  :code:`counts()` scales it by the shots the experiment ran at, or by one of your own.
- A circuit carrying final measurements can be cut. They are removed on a copy, so the
  circuit passed in is left as it was.
- :code:`RawResult.shots` and :code:`CutExperiment.can_reconstruct_probabilities` are
  public, and the types the public functions return are importable from :code:`QCut`.

**Version 2.0.0**
=================

Breaking changes
----------------

- :code:`RawResult` no longer takes a :code:`samples` argument. It is now
  :code:`RawResult(results, shots, experiment=None)` and carries the experiment itself.
- :code:`RawResult.result()` keeps its shape but each subcircuit now holds a
  :code:`CircuitResult` containing what the backend or sampler returned, plus the shot scale and
  the label selection that group takes from it rather than a counts dict. Call
  :code:`.counts()` on one for the counts.
- Modules are grouped into subpackages: :code:`QCut.qpd`, :code:`QCut.cutting`,
  :code:`QCut.execution`, :code:`QCut.errors` and :code:`QCut.utils`. The names exported
  from :code:`QCut` are unchanged; code importing a module directly has to be updated.
- :code:`find_cuts()` is now deterministic and costs several candidate partitions before
  choosing, so it returns different, cheaper plans than 1.3.2 did for the same circuit.
- :code:`find_cuts()` now reads its configuration from a :code:`CutOptions` object rather than from keyword arguments. See
  `Options <https://jooniv.github.io/QCut/Options.html>`__ and `Automatic cuts <https://jooniv.github.io/QCut/examples/AutomaticCuts.html>`__ for details.
- Cut edge weights are :code:`log gamma` rather than :code:`gamma`, which also changes
  which cuts are chosen. See `Automatic cuts <https://jooniv.github.io/QCut/examples/AutomaticCuts.html>`__.
- Subcircuits are no longer given empty classical registers, and an unused
  :code:`qpd_meas` register is dropped. Code reading registers by position rather than by
  name has to be updated.
- :code:`CutExperiment.expv_data()` is gone and :code:`estimate_expectation_values()`
  takes only the results, which now carry the experiment they came from. Replace
  :code:`estimate_expectation_values(results, experiment.expv_data())` with
  :code:`estimate_expectation_values(results)`.
  See `Usage <https://jooniv.github.io/QCut/Usage.html>`__.
- :code:`get_experiment_circuits()` no longer modifies the :code:`CutCircuit` it is
  given, so one can be reused for several observable sets.
- :code:`transpile_subcircuits()` raises rather than quietly overriding when
  :code:`remove_final_rzs` or :code:`optimize_single_qubits` is passed for an IQM
  backend. Those rewrite a circuit that still carries cut placeholders; use
  :code:`transpile_experiments()` instead.
- :code:`perform_move_routing` now defaults to whether the backend needs it, on for a
  resonator device and off otherwise, and an explicit value is honoured either way.

Cutting arbitrary two-qubit gates
---------------------------------

- Any two-qubit gate can be cut, with the decomposition derived from its KAK
  coordinates. See `Gate cuts <https://jooniv.github.io/QCut/examples/GateCuts.html>`__ and
  `Theory <https://jooniv.github.io/QCut/Theory.html>`__.

Joint cutting of parallel rotation gates
----------------------------------------

- Single-axis rotation gates running in parallel between the same two partitions share
  one decomposition: two parallel :code:`rzz` cost 30 subexperiments instead of 36, three
  cost 132 instead of 216. On by default.
  See `the derivation <https://jooniv.github.io/QCut/theory/Joint_rotation_derivation.html>`__.

Wire cuts with classical communication
--------------------------------------

- A block of parallel wire cuts can exchange the measured outcome, taking the overhead
  from :code:`4**n` to :code:`2**(n+1) - 1` and two wires from 64 subexperiments to 28.
  These run in waves. Used by default for blocks of two or more.
  See `the derivation <https://jooniv.github.io/QCut/theory/LOCC_wire_derivation.html>`__.

Gate consolidation
------------------

- Runs of gates on the same qubit pair are merged before cutting, so the pair costs one
  cut. Runs need not be contiguous, gates that commute with the run are moved out of the
  way. On by default, and compared against not merging.
  See `Options <https://jooniv.github.io/QCut/Options.html>`__.

Sampling instead of enumerating
-------------------------------

- The experiment can be sampled from the quasiprobability distribution rather than
  enumerated, which bounds the number of circuits when the exact count is out of reach.
  Automatic above 1000 groups. See `Options <https://jooniv.github.io/QCut/Options.html>`__.

Configuration
-------------

- Added :code:`CutOptions`, collected once and carried through the run. Covers
  consolidation, joint cuts, wire cut communication, expansion strategy, sampling and the
  cut finder. See `Options <https://jooniv.github.io/QCut/Options.html>`__.

Knowing what a cut costs
------------------------

- :code:`CutCircuit.gamma` is the sampling overhead of a split and
  :code:`CutCircuit.optimal_gamma` the least those same cuts could cost with every
  decomposition available. Both are closed form, so the cost of a plan can be read
  before any experiment circuits are built. :code:`CutExperiment` carries both forward.

Running on real hardware
------------------------

- Improved and fixed bugs in transpilation for real backends.
- Better IQM support: :code:`pip install "QCut[iqm]"`, and both transpile helpers use IQM's own
  transpiler for IQM backends, resonator machines included. Pass
  :code:`use_iqm_transpiler=False` to opt out.
  See `Usage <https://jooniv.github.io/QCut/Usage.html>`__.
- Resonator devices are supported by both transpile helpers. Their MOVE gates are routed
  while the subcircuits still carry cut placeholders, which the routing pass used to
  drop along with the classical registers and the layout.
  :code:`use_iqm_transpiler=False` raises for them, since standard qiskit has no MOVE gate.
- :code:`transpile_experiments()` works on IQM backends.
- A block of parallel wire cuts is never bundled for a resonator device, which reports
  its qubits as fully coupled but has no two-qubit gate that avoids its resonator. Those
  cuts fall back to one block per wire rather than to the local decomposition.
- :code:`run_experiments()` also takes a V2 sampler as its :code:`backend`, on both the
  plain and the communicating execution path. Note that
  :code:`qiskit.primitives.StatevectorSampler` cannot be used, since it refuses
  mid-circuit measurements.
- Every batch of a wave is submitted before any of it is collected, so a run queues all
  of its jobs at once rather than waiting out each batch in turn.
- :code:`run_experiments()` takes :code:`run_options`, passed on to every :code:`run`
  call. A target that batches on its own account, such as
  `fiqci-ems <https://github.com/FiQCI/fiqci-ems>`__ does, is also given QCut's
  :code:`max_batch_size`, so it does not split a batch QCut has already sized.
- :code:`run()` and :code:`run_cut_circuit()` now take :code:`shots`.

Other
-----

- :code:`from QCut import *` no longer raises.
- Test suite split into tiers. See :code:`CONTRIBUTING.md`.
- Dropped pickle, experiment generation is much faster.
- :code:`finder_max_qubits` no longer raises when given as a list.
- Benchmarks against IBM's cutting addon, in :code:`benchmarks/`.

**Version 1.3.2**
=================
- Fix bug in how weights for different gates were being handled by `QCutFind`.

**Version 1.3.1**
=================
- Fix bug in `find_cuts()` where on circuits that already had the desired partition early return condition would return incorrect type.

**Version 1.3.0**
=================

Support for cutting SWAP and iSWAP gates
----------------------------------------

- Added support for cutting SWAP and iSWAP gates in addition to CZ gates and wire cuts.
    * Uses optimal QPDs instead of naively decomposing into CZ gates, resulting in significantly fewer subcircuits needed for the same number of cuts.
    * QCutFind also supports finding optimal cut locations for cutting SWAP and iSWAP gates.
    * Refactored codebase to support cutting arbitrary gates in a more modular way, making it easier to add support for cutting more gates in the future.
    * Check documentation for details on how to use.

**Version 1.2.0**
=================

Refactor `run_experiments()`
----------------------------

- Refactor `run_experiments()` to improve performance, flexibility, and code quality.
- `run_experiments()` now batches circuits into groups for more efficient execution on real hardware.
    * By default, circuits are grouped into batches of 100, but this can be adjusted with the `max_batch_size` parameter in `run_experiments()`.
- `run_experiments()` now returns a `RawResults` object instead of a list of preprocessed results. This is to allow for more flexible postprocessing of results, including support for custom postprocessing functions.
    * post processing now automatically happens when calling `get_expectation_values()` that now takes a `RawResults` object as an argument.

Add `RawResults` class
-----------------------

- The `RawResults` object contains the raw results from the backend, as well as metadata such as the number of shots and samples used.
    * This allows for more flexible postprocessing of results, including support for custom postprocessing functions.
    * Like a qiskit result object the `RawResults` has a `result()` method that returns the raw results from the backend. The format is currently quite messy and could be improved in future releases. 


Add logging
------------

- Added logging to the codebase to improve debuggability and provide more information about the execution of the code.
- Examples on how to use the logging can be found in the documentation.

Version 1.1.1
=============
- Fix `find_cut()` not passing `max_qubits` parameter to `get_locations_and_subcircuits()`, which caused incorrect cut_circuits to be returned.

Version 1.1.0
=============
- Small syntax change for placing wire cuts or cz cuts directly
    * Instead of :code:`cut_circuit.append(cut, [1])` now use :code:`cut_circuit.append(cut(), [1])`
    * Instead of :code:`cut_circuit.append(cutCZ, [0,1])` now use :code:`cut_circuit.append(**cutGate(CZGate(), 0, 1))`
    * Check documentation for details on how to use.
- Comprehensive refactor of codebase and documentation to improve readability and maintainability.

Version 1.0.2
=============
- Fix bug in transpile_experiments()
- Adjust supported qiskit and python versions to better match iqm-client
    * Drop Python 3.10 support
    * Drop qiskit 1.0 support
    * Supported versions now Python >= 3.11, < 3.13 and qiskit >= 1.1, < 3.0

Version 1.0.1
=============
- Fix typo in pyproject.toml

Version 1.0.0
=============
- Yanked due to pyproject.toml typo. Please use 1.0.1 instead.
- Support for Qiskit 2.x
- QCut now supports Qiskit 1.0+

Version 0.9.2
=============
- Minor fix

Version 0.9.1
=============
- Fixes for issues for subcircuit construction with Qiskit > 1.2
    - Qiskit version requirement updated to >= 1.0, < 2.0

Version 0.9.0
=============
- Migrate from index based Z-observables to Qiskits SparsePauliOps
    * The observables parameter for all functions now takes a list of Qiskits :code:`SparsePauliOp` objects instead of lists of qubit indices.
    * This allows for more general observables to be calculated, including multi-qubit observables and observables with different Pauli operators.
    * Check documentation for details on how to use.

Version 0.8.0
=============
- Support for cutting 2 qubit gates
    * Added :code:`cutGate` function for cutting 2 qubit gates directly.
    * Cutting done by transpiling the 2 qubit gate into a cut CZ gate with appropriate basis changes.
    * For non CZ family gates this results in suboptimal decompositions. More optimised decompositions will be added in future releases.
    * Check documentation for details on how to use.

Version 0.7.0
=============
- Rework transpilation workflow
    * Transpilation now done per subcircuit instead of per experiment circuit.
         * Per experiment transpilation still provided for more control.
         * Users can of course still manually transpile circuits before passing to QCut.
    * This greatly reduces the number of transpilation calls needed, improving performance.
    * Check documentation for details on how to use.
- Bug fixes

Version 0.6.0
=============
- Support cutting CZ gates
    * Added support for cutting CZ gates in addition wire cuts.
    * Check documentation for details on how to use.
- Added automatic cut finding feature
    * Added :code:`QCutFind` module for automatically finding good cut locations in a circuit.
    * Check documentation for details on how to use.
- Bug fixes
- Drop windows support for the time being due to METIS issues
    * Windows users should use WSL

Version 0.3.0
=============
- Support for IQM Qiskit 17.8
    * Added support for IQM Qiskit 17.8.
- Removed built in :code:`mitigate` flag
- Bug fixes
- Remove old two qubit gate :code:`CutWire` operation
    * Users using the old method can consult documentation for migration help

Version 0.2.4
=============
- Bugfix for incorrect partitioning for cases where there are multipe cuts on a single wire

Version 0.2.3
=============
- Major optimisation on :code:`_move_to_new_wire` method.
    * Old version took around 11s for a random circuit with depth of 50 and 50 qubits.
    * New version takes around 0.2s for the same circuit.

Version 0.2.2
=============
- Bugfix for array overflow in :code:`get_experiment_circuits`

Version 0.2.1
=============
- Hotfix for incorrect version of qiskit-aer in pyproject.toml.
- Fix pypi workflow

Version 0.2.0
=============
- Single qubit cut gate now the default cut method
    * Greatly simplifies placing cuts.
    * Old two qubit gate deprecated and will be removed soon.
    * Check out documentation for migration help.

Version 0.1.3
=============
- Hotfix for source files not included in pypi build.
    * 0.1.0 - 0.1.2 not installable.

Version 0.1.2
=============
- Add Qiskit 1.0 support.
    * Supported versions now >= 0.45.3, < 1.2.
    * No worflow changes. No migration required.
    * Compatible with qiskit-iqm 13.15
- Add Python 3.11 support.
    * Supported versions now >= 3.9, < 3.12.
- Fix bug in :code:`_get_bounds()` method.
- Use :code:`pickle.loads(pickle.dumps())` instead of :code:`deepcopy()` in :code:`_get_experiment_circuits()`.
    * Slight performance improvement.
- Revert back to vx.x.x versioning scheme.


Version 0.1.1
=============

- Code quality improvements:
    * Move to pyproject.toml. Contents of ruff.toml and setup.py now live in pyproject.toml.
    * Relative imports are now absolute imports.
    * All images are now located under the _static folder.
- Added Github workflows:
    * Added github actions workflows for building the documentation and testing that the documentation can be built.
    * Added github actions workflows for testing the code.
    * Added github actions workflows for publishing to pypi.
    * Added github actions workflows for linting.
- Revamped documentation:
    * Documentation now uses the Book theme.
    * Fixed all warnings from sphinx when building the documentation.
    * Fixed spelling mistakes.
    * Added a new page for the changelog.
- README now contains the information to build the docs.
- Change versioning scheme to "x.x.x" instead of "vx.x.x".

Version 0.1.0
=============

- First release
