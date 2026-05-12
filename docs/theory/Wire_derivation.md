# Deriving Optimal QPD for Wire Cuts

**Here we will briefly go over where the decomposition used for wire cuts comes from and how that is transfromed to operations implementable on a quantum computer. The derivation is based on [[1][2]](#wire_ref).**

## Decompose Single Qubit Density Matrix

Any single qubit density matrix can be decomposed as:

$$
\rho=\frac{I}{2}+\frac{Tr[X\rho]}{2}X+\frac{Tr[Y\rho]}{2}Y+\frac{Tr[Z\rho]}{2}Z
$$

We now want to express this as a sum of measure-prepare channels (measure in a pauli basis, prepare a pauli basis state).

Start by ignoring the $\frac{I}{2}$ term for now. Each term has the form $\frac{1}{2}Tr[\sigma\rho]\sigma$. Now expand this using the spectral decomposition $\sigma=P^+-P^-$ where $P^\pm=\frac{I+\sigma}{2}$ are the projectors to the $\pm 1$ eigenstates of $\sigma$. Similarly $Tr[P^+\rho]-Tr[P^-\rho]$.

Expanding the product:

$$
\begin{aligned}
    Tr[\sigma\rho]\sigma &= (Tr[P^+\rho]-Tr[P^-\rho])(P^+-P^-) \\
    &= (Tr[P^+\rho]-Tr[P^+\rho])P^+ - (Tr[P^-\rho]-Tr[P^-\rho])P^- \\
    &= Tr[\sigma\rho]P^+ - Tr[\sigma\rho]P^-
\end{aligned}
$$

We can now observe that the $\frac{1}{2}Tr[\sigma\rho]\sigma$ gives two measure-prepare channels, one per eigenstate:

| Measure | Prepare              | Coef |
| :-----  | :-----:              | ---: |
| $\sigma$| $\frac{I+\sigma}{2}$ | 1/2  |
| $\sigma$| $\frac{I-\sigma}{2}$ | -1/2 |

Now looking at the $\frac{I}{2}$ term we can write it as:

$$
\frac{I}{2}=\frac{1}{2}Tr[I\rho]\ket{0}\bra{0}+\frac{1}{2}Tr[I\rho]\ket{1}\bra{1}
$$

These are again two measure-prepare channels.

Now writing out all the measure prepare channels for I, X, Y, Z:

| Measure | Prepare            | Coef |
| :-----  | :-----:            | ---: |
| I       | $\ket{0}\bra{0}$   | 1/2  |
| I       | $\ket{1}\bra{1}$   | 1/2  |
| Z       | $\ket{0}\bra{0}$   | 1/2  |
| Z       | $\ket{1}\bra{1}$   | -1/2 |
| X       | $\ket{+}\bra{+}$   | 1/2  |
| X       | $\ket{-}\bra{-}$   | -1/2 |
| Y       | $\ket{i}\bra{i}$   | 1/2  |
| Y       | $\ket{-i}\bra{-i}$ | -1/2 |

Note that here the identity basis measurement (I) stands for a measurement that always reutrns the $\ket{0}$ state. Practically this means not measuring and leaving a classical bit in the initial 0 position.

We have now decomposed any single qubit density matrix $\rho$ as $\sum_{i=1}^8c_iTr[O_i\rho]\rho_i$.

Summing over the absolute coefficient we get $\gamma=4$ which is optimal (without communication).

(wire_ref)=
## References
1. T. Peng, A. W. Harrow, M. Ozols, and X. Wu, ‘Simulating Large Quantum Circuits on a Small Quantum Computer’, Phys. Rev. Lett., vol. 125, no. 15, p. 150504, Oct. 2020, doi: 10.1103/PhysRevLett.125.150504.
2. H. Harada, K. Wada, and N. Yamamoto, ‘Doubly Optimal Parallel Wire Cutting without Ancilla Qubits’, PRX Quantum, vol. 5, no. 4, p. 040308, Oct. 2024, doi: 10.1103/PRXQuantum.5.040308.