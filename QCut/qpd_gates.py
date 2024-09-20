"""Helper gates for circuit knitting."""

from qiskit import QuantumCircuit

# define measurements for different bases
xmeas = QuantumCircuit(1, 1, name="x-meas")
xmeas.h(0)
xmeas.measure(0, 0)

ymeas = QuantumCircuit(1, 1, name="y-meas")
ymeas.sdg(0)
ymeas.h(0)
ymeas.measure(0, 0)

idmeas = QuantumCircuit(1, name="id-meas")

zmeas = QuantumCircuit(1, 1, name="z-meas")
zmeas.measure(0, 0)

# define the cut location marker
"""Cut-instruction: two qubit gate. 0-qubit is the measure channel
    and 1-qubit the initialize channel."""
cut_wire = QuantumCircuit(2, name="Cut")
cut_wire = cut_wire.to_instruction()

# define initialization operations
zero_init = QuantumCircuit(1, name="0-init")

one_init = QuantumCircuit(1, name="1-init")
one_init.x(0)

plus_init = QuantumCircuit(1, name="'+'-init")
plus_init.h(0)

minus_init = QuantumCircuit(1, name="'-'-init")
minus_init.h(0)
minus_init.z(0)

i_plus_init = QuantumCircuit(1, name="'i+'-init")
i_plus_init.h(0)
i_plus_init.s(0)

i_minus_init = QuantumCircuit(1, name="'i-'-init")
i_minus_init.h(0)
i_minus_init.z(0)
i_minus_init.s(0)
