- [QCut](#qcut)
- [Installation](#installation)
- [Usage](#usage)
  - [Manual Usage](#manual-usage)
  - [Usage shorthand](#usage-shorthand)
  - [Automatic cuts](#automatic-cuts)
  - [Running on IQM fake backends](#running-on-iqm-fake-backends)
  - [Running on FiQCI](#running-on-fiqci)
  - [Running on other hardware](#running-on-other-hardware)
- [Documentation](#documentation)
- [Acknowledgements](#acknowledgements)
- [License](#license)


# QCut

QCut is a quantum circuit knitting package for gate cuts and resrtless wire cuts. QCut has been designed to work with IQM's qpus, and therefore on the Finnish Quantum Computing Infrastructure ([FiQCI](https://fiqci.fi/)), and tested with an IQM qpus. Additionally, QCut is built to be combatible with IQM’s Qiskit fork iqm_qiskit.

QCut has been built at CSC - IT Center for Science (Finnish IT Center for Science).

Check out [jooniv.github.io/QCut/](https://jooniv.github.io/QCut/) for documentation and more examples.

# Installation

For installation a UNIX-like system is currently needed due to pymetis being used for automatic cut finding. On Windows use WSL

**Pip:**  
Installation should be done via `uv`

```bash
uv pip install QCut
#or
uv add QCut
```

Uv can be installed with

```bash
#Linux / mac
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Note: for drawing circuits you might have to install pylatexenc. This can also be done with uv.

```bash
uv pip install pylatexenc
#or
uv add pylatexenc
```

**Install from source**  
It is also possible to use QCut by cloning this repository and including it in your project folder.

```bash
cd QCut
uv pip install .
#or
uv sync --no-dev

#or with dev deps
uv sync
```

# Usage

## Manual Usage

**1: Import needed packages**

```python
import QCut as ck
from QCut import cut, cutCZ
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Estimator
```

**2: Start by defining a QuantumCircuit just like in Qiskit**

```python
circuit  =  QuantumCircuit(4)

mult = 1.635
circuit.r(mult*0.46262, mult*0.1446, 0)
circuit.cx(0,1)
circuit.cx(1,2)
circuit.cx(2,3)
   
circuit.measure_all()

circuit.draw("mpl")
```

![](./docs/_static/images/circ1.png)

**3: Insert cut operations to the circuit to denote where we want to cut the circuit**

Note that here we don't insert any measurements. Measurements will be automatically handled by QCut.

```python
cut_circuit = QuantumCircuit(4)

mult = 1.635
cut_circuit.r(mult*0.46262, mult*0.1446, 0)
cut_circuit.h(1)
cut_circuit.append(cutCZ, [0,1])
cut_circuit.h(1)
cut_circuit.append(cut, [1])
cut_circuit.cx(1,2)
cut_circuit.cx(2,3)

cut_circuit.draw("mpl")
```

**Note** that currently QCut only supports cutting Cz gates so transformation have to be done manually for the time being (hence the added H gates)


![](./docs/_static/images/circ2.png)

**4: Extract cut locations from cut_circuit and split it into independent subcircuit.**

```python
cut_locations, subcircuits, map_qubit = ck.get_locations_and_subcircuits(cut_circuit)
```

Now we can draw our subcircuits.

```python
subcircuits[0].draw("mpl")
```

![](./docs/_static/images/circ3.png)

```python
subcircuits[1].draw("mpl")
```

![](./docs/_static/images/circ4.png)

```python
subcircuits[2].draw("mpl")
```

![](./docs/_static/images/circ11.png)

**5 Define backend and transpile the cut circuit**

```python
fake = IQMFakeAdonis() #noisy
sim = AerSimulator() #ideal
```

```python
transpiled = ck.transpile_subcircuits(subcircuits, cut_locations, fake, optimization_level=3)
```

**6: Generate experiment circuits**

```python
experiment_circuits, coefficients, id_meas = ck.get_experiment_circuits(transpiled, cut_locations)
```

**7: Run the experiment circuits**

```python
results = ck.run_experiments(experiment_circuits, cut_locations, id_meas, backend=fake)
```

**8: Define observables and calculate expectation values**

Observables are Pauli-Z observables and are defined as a list of qubit indices. Multi-qubit observables are defined as a list inside the observable list.

If one wishes to calculate other than Pauli-Z observable expectation values currently this needs to be done by manually modifying the initial circuit to perform the basis transform.

```python
observables = [0,1,2, [0,1]]
expectation_values = ck.estimate_expectation_values(results, coefficients, cut_locations, observables, map_qubit)
```

**9: Finally calculate the exact and noisy expectation values of the original circuit and compare them to the results calculated with QCut**

```python
paulilist_observables = ck.get_pauli_list(observables, circuit.num_qubits)

estimator = Estimator(run_options={"shots": None}, approximation=True)
exact_expvals = (
    estimator.run([circuit] * len(paulilist_observables),  # noqa: PD011
                  list(paulilist_observables)).result().values
)

tr = transpile(circuit, backend=fake)
counts, exps = ck.run_and_expectation_value(tr, fake, observables, shots=2048)
```

```python
import numpy as np

np.set_printoptions(formatter={"float": lambda x: f"{x:0.6f}"})

print(f"QCut expectation values:{np.array(expectation_values)}")
print(f"Noisy expectation values with fake backend:{np.array(exps)}")
print(f"Exact expectation values with ideal simulator :{np.array(exact_expvals)}")
```

`QCut circuit knitting expectation values: [0.007532 0.007532 -0.003662 1.010128]`

`Noisy expectation values with fake backend:[0.478516 0.621094 0.558594 0.689453]`

`Exact expectation values with ideal simulator :[0.000000 0.000000 0.000000 1.000000]`

As we can see QCut is able to accurately reconstruct the expectation values and be more accurate that just using the fake backend as is. (Note that since this is a probabilistic method the results vary a bit each run)

Additionally we can execute QCut using the ideal Aer simulator and see that we get (practically) exact results:


`QCut ideal expectation values:[0.727968 0.727968 0.727968 1.015832]`


## Usage shorthand

For convenience, it is not necessary to go through each of the aforementioned steps individually. Instead, QCut provides a function `run()` that executes the whole wire-cutting sequence.

The same example can then be run like this:

```python
sim = AerSimulator()
observables = [0,1,2, [0,2]]

estimated_expectation_values = ck.run(cut_circuit, observables, sim)
```

## Automatic cuts

QCut comes with functionality for automatically finding good cut locations that can place both wire and gate cuts.

```python
from QCut import find_cuts

cut_locations, subcircuits, map_qubit = find_cuts(circuit , 3, cuts="both")

estimated_expectation_values = ck.run_cut_circuit(subcircuits, cut_locations, observables, map_qubit, sim)

np.set_printoptions(formatter={"float": lambda x: f"{x:0.6f}"})

print(f"QCut expectation values:{np.array(estimated_expectation_values)}")
print(f"Exact expectation values with ideal simulator :{np.array(exact_expvals)}")
```

`QCut expectation values:[0.729648 0.745609 0.702871 0.992620]`

`Exact expectation values with ideal simulator :[0.727323 0.727323 0.727323 1.000000]`


## Running on IQM fake backends

To use QCut with IQM's fake backends it is required to install [Qiskit IQM](https://github.com/iqm-finland/qiskit-on-iqm). QCut supports version 17.8. Installation can be done with uv:

```bash
uv pip install qiskit-iqm==17.8
#or
uv add qiskit-iqm==17.8

```

After installation just import the backend you want to use:

```python
from iqm.qiskit_iqm import IQMFakeAdonis()
backend = IQMFakeAdonis()
```

## Running on FiQCI

For running on real hardware through the Lumi supercomputer's FiQCI partition follow the instructions [here](https://docs.csc.fi/computing/quantum-computing/helmi/running-on-helmi/). If you are used to using Qiskit on jupyter notebooks it is recommended to use the [Lumi web interface](https://docs.lumi-supercomputer.eu/runjobs/webui/).

## Running on other hardware

Running on other providers such as IBM is untested at the moment but as long as the hardware can be accessed with Qiskit QCut should be compatible.

# Documentation

Check out [jooniv.github.io/QCut/](https://jooniv.github.io/QCut/) for documentation and more examples.

The docs are built with sphinx using the sphinx book theme. To build the docs:

```bash
cd docs
uv pip install -r requirements-docs.txt
sphinx-build -v -b html . build/sphinx/html -W
```

HTML files can then be found under `build/sphinx/html/`

# Acknowledgements

This project is built on top of [Qiskit](https://github.com/Qiskit/qiskit) which is licensed under the Apache 2.0 license.

# License

[Apache 2.0 license](https://github.com/JooNiv/QCut/blob/main/LICENSE)
