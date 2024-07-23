## QCut

QCut is a quantum circuit knitting package for performing wire cuts especially designed to not use reset gates or mid-circuit since on early NISQ devices they pose significant errors, if they are even available.

QCut has been designed to work with IQM's qpus, and therefore on the Finnish Quantum Computing Infrastructure ([FiQCI](https://fiqci.fi/)), and tested with an IQM Adonis 5-qubit qpu. Additionally, QCut is built on top of Qiskit 0.45.3 which is the current supported Qiskit version of IQM's Qiskit fork iqm\_qiskit.

QCut was built as a part of a summer internship at CSC - IT Center for Science (Finnish IT Center for Science).

## Installation

**Pip:**  
Installation should be done via `pip`

```python
pip install QCut
```

Using pip is the recommended install method.

**Install from source**  
It is also possible to use QCut by cloning this repository and including it in your project folder.

## Usage

**1: Import needed packages**

```python
import QCut as ck
from QCut import cut_wire
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Estimator
```

**2: Start by defining a QuantumCircuit just like in Qiskit**

```python
circuit  =  QuantumCircuit(3)
circuit.h(0)
circuit.cx(0,1)
circuit.cx(1,2)
   
circuit.measure_all()

circuit.draw("mpl")
```

![](https://33333.cdn.cke-cs.com/kSW7V9NHUXugvhoQeFaf/images/de618d0f193a9532d9ee849bac62e319cd5df3e35a4f9331.png)

**3: Insert cut\_wire operations to the circuit to denote where we want to cut the circuit**

Note that here we don't insert any measurements. Measurements will be automatically handled by QCut.

```python
cut_circuit = QuantumCircuit(4)
cut_circuit.h(0)
cut_circuit.cx(0,1)
cut_circuit.append(cut_wire, [1,2])
cut_circuit.cx(2,3)

cut_circuit.draw("mpl")
```

![](https://33333.cdn.cke-cs.com/kSW7V9NHUXugvhoQeFaf/images/97085d3d332653534f045ebd4bceb473b28a1c93014c06c8.png)

**4\. Extract cut locations from cut\_circuit and split it into independent subcircuit.**

```python
cut_locations, subcircuits = ck.get_locations_and_subcircuits(cut_circuit)
```

Now we can draw our subcircuits.

```python
subcircuits[0].draw("mpl")
```

![](https://33333.cdn.cke-cs.com/kSW7V9NHUXugvhoQeFaf/images/168872c3d19110c76d6295dce1a9641156e38a9b31de2008.png)

```python
subcircuits[1].draw("mpl")
```

![](https://33333.cdn.cke-cs.com/kSW7V9NHUXugvhoQeFaf/images/38cef69b2bf031fb8d88f791932bf5af9b53ba796ac22fd1.png)

**5: Generate experiment circuits by inserting operations from a quasi-probability distribution for the identity channel**

```python
experiment_circuits, coefficients, id_meas = ck.get_experiment_circuits(subcircuits, cut_locations)
```

**6: Run the experiment circuits**

Here we are using the qisit AerSimulator as our backend but since QCut is backend independent you can choose whatever backend you want as long as you transpile the experiment circuits accordingly. QCut provides a function `transpile_experiments()` for doing just this.

Since QCut is a circuit knitting package the results are approximations of the actual values. Error is the error in the approximation.

```python
backend = AerSimulator()
error = 0.03
results = ck.run_experiments(experiment_circuits, cut_locations, id_meas, error=error, backend=backend, mitigate=True)
```

**7\. Define observables and calculate expectation values**

Observables are Pauli-Z observables and are defined as a list of qubit indices. Multi-qubit observables are defined as a list inside the observable list.

If one wishes to calculate other than Pauli-Z observable expectation values currently this needs to be done by manually modifying the initial circuit to perform the basis transform.

```python
observables = [0,1,2, [0,2]]
expectation_values = ck.estimate_expectation_values(results, coefficients, cut_locations, observables, error)
```

**8: Finally calculate the exact expectation values and compare them to the results calculated with QCut**

```python
paulilist_observables = ck.get_pauli_list(observables, 3)

estimator = Estimator(run_options={"shots": None}, approximation=True)
exact_expvals = (
    estimator.run([circuit] * len(paulilist_observables),  # noqa: PD011
                  list(paulilist_observables)).result().values
)
```

```python
import numpy as np

np.set_printoptions(formatter={"float": lambda x: f"{x:0.6f}"})

print(f"QCut expectation values:{np.array(expectation_values)}")
print(f"Exact expectation values with ideal simulator :{np.array(exact_expvals)}")
```

`QCut expectation values:[-0.018534 -0.018534 -0.012826 0.998416]`

`Exact expectation values with ideal simulator :[0.000000 0.000000 0.000000 1.000000]`

As we can see QCut is able to accurately reconstruct the expectation values. (Note that since this is a probabilistic method the results vary a bit each run)

## Usage shorthand

For convenience, it is not necessary to go through each of the aforementioned steps individually. Instead, QCut provides a function `run()` that executes the whole wire-cutting sequence.

The same example can then be run like this:

```python
backend = AerSimulator()
observables = [0,1,2, [0,2]]
error = 0.03

estimated_expectation_values = ck.run(cut_circuit, observables, error, backend, mitigate=True)
```

## Running on IQM fake backends

To use QCut with IQM's fake backends it is required to install [Qiskit IQM](https://github.com/iqm-finland/qiskit-on-iqm). QCut supports version 13.7. Installation can be done with pip:

```python
pip install qiskit-iqm
```

After installation just import the backend you want to use:

```python
from iqm.qiskit_iqm import IQMFakeAdonis()
backend = IQMFakeAdonis()
```

## Running on FiQCI

For running on real IQM hardware through the Lumi supercomputer's FiQCI partition follow the instructions [here](https://docs.csc.fi/computing/quantum-computing/helmi/running-on-helmi/). If you are used to using Qiskit on jupyter notebooks it is recommended to use the [Lumi web interface](https://docs.lumi-supercomputer.eu/runjobs/webui/).

## Running on other hardware

Running on other providers such as IBM is untested at the moment but as long as the hardware can be accessed with Qiskit version \< 1.0 then QCut should be compatible.

## Acknowledgements

This project is built on top of [Qiskit](https://github.com/Qiskit/qiskit) which is licensed under the Apache 2.0 license.

## License

[Apache 2.0 license](https://github.com/JooNiv/QCut/blob/main/LICENSE)
