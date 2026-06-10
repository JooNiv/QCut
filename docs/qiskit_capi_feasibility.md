# Feasibility: porting QCut circuit generation to the Qiskit C API

**Question.** Can QCut's experiment-circuit generation
(`circuit_knitting.get_experiment_circuits` and `circuit_preparation`) be moved to
C/C++ via the Qiskit C API for a speed-up, as part of the LUMI HPC effort?

**Short answer.** The C API in the *installed* Qiskit (2.4.1) is much more capable
than earlier releases — it has circuit construction, **DAG manipulation**, and
**bidirectional Python↔C circuit conversion**, and the Python-extension build needs
no separate `libqiskit` to link. So a port is *technically feasible*. But the
dominant cost at scale is **circuit execution**, which the MPI task-farm already
parallelizes, and the cheapest generation win (replacing the pickle deep-copy with
`QuantumCircuit.copy()`) was landed for a measured **3.1× copy speed-up** with no C
code. **Recommendation: do not port now. Keep the C-API port as a targeted,
benchmarked prototype of the innermost copy+splice loop only.**

## Environment (measured)

| Fact | Value |
|---|---|
| Installed Qiskit | 2.4.1 (`qiskit >= 1.1, < 3.0` per pyproject) |
| C umbrella header | present: `qiskit/capi/include/qiskit.h` |
| Generated sub-headers | present: `types.h`, `funcs.h`, `funcs_py.h`, … |
| `libqiskit.so` to link against | **absent** — not needed for the Python-extension path |
| Build pattern | `#define QISKIT_PYTHON_EXTENSION` + `qk_import()`; symbols resolve at runtime from the already-loaded `qiskit._accelerate` module |

So a CPython extension that calls into Qiskit's Rust core is buildable today against
the wheel: compile with the bundled headers, call `qk_import()` in the module init,
and load the `.so` after `import qiskit`. No source build of Qiskit required.

## Capability matrix: what QCut needs vs what the C API exposes

| QCut generation step (file) | C API support (2.4.1) | Verdict |
|---|---|---|
| Deep-copy a base subcircuit per variant (`circuit_knitting.py:225`) | `qk_circuit_copy` | ✅ supported (but `QuantumCircuit.copy()` in Python already gives 3.1×) |
| Take a Python `QuantumCircuit` into C and return one | `qk_circuit_borrow_from_python`, `qk_circuit_convert_from_python`, `qk_circuit_to_python(_full)` | ✅ the key enabler — no manual marshalling |
| Split into independent subcircuits via DAG (`_separate_subcircuits`) | `qk_circuit_to_dag`, `qk_dag_*`, `qk_dag_to_circuit` | ✅ full DAG API incl. `substitute_node_with_dag`, `compose`, `predecessors/successors` |
| Append standard gates / measure / reset / unitary | `qk_circuit_gate` (fixed `QkGate` enum), `qk_circuit_measure`, `qk_circuit_reset`, `qk_circuit_unitary`, `qk_circuit_parameterized_gate` | ✅ for standard gates |
| Splice a pre-built QPD sub-instruction sequence at an index (`_insert_*_qpd`) | express as `qk_dag_substitute_node_with_dag` / `qk_dag_compose` | ✅ feasible via DAG |
| **Custom named placeholder instructions** `Meas_i`, `Init_i`, `CutGate`, `obs_i` | no API to append an arbitrary *named opaque* instruction (gate set is the `QkGate` enum + unitary/pauli-product) | ⚠️ **gap** — would require redesigning placeholders, or keeping placeholder bookkeeping in Python |
| Observable basis transform via `PassManager` (`basis_transform.py`) | `qk_circuit_inst_pauli_product_measurement/_rotation`, `qk_transpile_layout_to_python`; no general Python `PassManager` from C | ⚠️ partial — Pauli-product instructions help, custom passes do not port |

The one hard blocker is QCut's reliance on **custom *named* instructions** as
placeholders. Everything else (DAG surgery, copy, conversion) is now available.

## Measured Python-side numbers (no C required)

Microbench (`docs/genbench.py`), 20 000 deep-copies of one cut subcircuit:

```
pickle.loads(pickle.dumps(circ))   3.49 s
circ.copy()                        1.12 s     → 3.1× faster
```

The `.copy()` change is already merged into `get_experiment_circuits`; all 64 tests
pass. This captures most of the per-copy overhead the C port would target, at zero
build/maintenance cost.

## Where the time actually goes

For a real cut workload the circuit *count* is `∏(QPD terms) × obs-groups ×
subcircuits` and each circuit is then **simulated** (shots × statevector). Simulation
dominates generation by orders of magnitude and is exactly what `QCut.mpi_run`
parallelizes across LUMI ranks. Optimizing generation in C without parallelizing
execution would move the wrong bottleneck.

## Go / no-go

**No-go for a full port now.** Reasons: (1) execution, not generation, is the scaling
cost and is already addressed by MPI; (2) the cheap `.copy()` win already removed the
top generation hotspot; (3) the custom-named-instruction gap forces a placeholder
redesign that touches `circuit_preparation`, `qpd_operations`, and `basis_transform`.

**Conditional go for a *prototype spike*** if generation is later shown (via profiling
under MPI at scale) to be a rank-0 bottleneck. Scope it to the innermost
copy-and-splice loop only:

1. Build a CPython extension (`genc`) against the bundled headers with
   `-DQISKIT_PYTHON_EXTENSION`; call `qk_import()` in `PyInit_genc`.
2. `qk_circuit_borrow_from_python` the base subcircuit → `qk_circuit_to_dag` →
   splice the QPD DAG fragments with `qk_dag_substitute_node_with_dag` →
   `qk_dag_to_circuit` → `qk_circuit_to_python`. Avoid named placeholders: pass the
   splice indices and fragment specs from Python.
3. Benchmark C-built vs `.copy()`-in-Python on one representative subcircuit ensemble;
   require a ≥3× wall-clock win on *generation* before investing further.

Note for LUMI: building the extension on LUMI-C needs the cray compiler wrappers
(`cc`) and the Qiskit headers from the same wheel used at runtime — pin the Qiskit
version between build and run.
