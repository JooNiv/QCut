"""Helper gates for circuit knitting."""

from qiskit import QuantumCircuit

#define measurements for different bases
c = QuantumCircuit(1,1, name="x-meas")
c.h(0)
c.measure(0,0)
xmeas = c.to_instruction()

c = QuantumCircuit(1,1, name="y-meas")
c.sdg(0)
c.h(0)
c.measure(0,0)
ymeas = c.to_instruction()

c = QuantumCircuit(1, name="id-meas")
idmeas = c.to_instruction()

c = QuantumCircuit(1,1, name="z-meas")
c.measure(0,0)
zmeas = c.to_instruction()

#define the cut location marker
cut = QuantumCircuit(2, name="Cut")
cut_wire = cut.to_instruction()

#define initialization instructions for the eigenstates
c = QuantumCircuit(1, name="0-init")
zero_init = c.to_instruction()

c = QuantumCircuit(1, name="1-init")
c.x(0)
one_init = c.to_instruction()

c = QuantumCircuit(1, name="'+'-init")
c.h(0)
plus_init = c.to_instruction()

c = QuantumCircuit(1, name="'-'-init")
c.h(0)
c.z(0)
minus_init = c.to_instruction()

c = QuantumCircuit(1, name="'i+'-init")
c.h(0)
c.s(0)
i_plus_init = c.to_instruction()

c = QuantumCircuit(1, name="'i-'-init")
c.h(0)
c.z(0)
c.s(0)
i_minus_init = c.to_instruction()
