# Deriving optimal QPD for the CZ gate

**
Here we will briefly go over where the decomposition for the CZ gate used comes from and how that is transfromed to operations implementable on a quantum computer. The derivation is based on [1].
**

## Background

The goal is to decompose a quantum channel implementing the CZ gate into a set of local channels implementable as single qubits operations. This decomposed channel can then be sampled to approximate the expectation values of the original channel.

## Factor the CZ gate

The CZ gate can (upto global phase) be decomposed as:
$$ 
\begin{equation}
CZ = \exp(i\pi Z \otimes I/4) \cdot \exp(i\pi I \otimes Z/4) \cdot \exp(-i\pi Z \otimes Z/4) 
\end{equation}
$$ 

We can immediately see that the first two channels are just local single qubit Z rotations. The only non-local part is then the third term $\exp(-i\pi Z \otimes Z/4)$ and so has the form $\exp(i\theta A_1 \otimes A_2)$ with $A_1 = A_2 = Z, \theta = -\pi/4$

## Apply Lemma 1 from [1]

The lemma states that we can decompose the super operator $S(\exp(i\theta A_1 \otimes A_2))$ as:
$$
\begin{aligned}
S(\exp(i\theta A_1 \otimes A_2)) &= \cos^2(\theta)S(I \otimes I) + \sin^2(\theta)S(A_1 \otimes A_2) + 1/8 \cos(\theta)\sin(\theta) \\ 
& \sum_{(\alpha_1, \alpha_2) \in \{\plusmn1\}} \alpha_1 \alpha_2 [S((I+\alpha_1A_1) \otimes S(I+i\alpha_2A_2)) + \\
& S((I+i\alpha_1A_1) \otimes S(I+\alpha_2A_2))] 
\end{aligned}
$$

if $A_1^2 = A_2^2 = I$, which is the case for the pauli operators.

## Transform to gates

In less mathematical terms we can decompose any gate of form $S(\exp(i\theta A_1 \otimes A_2))$ into a set of 6 single qubit operations. Looking at the decomposition we can see that term 1 is just applying identity on both qubits aka doing nothing. The second term is applying $A_1$ on qubit 1 and $A_2$ on qubit 2, remember that A are pauli operators. The third term is a bit more complex so let us look at it in detail.

### Term 3

Let us look at the structure that repeats twice in the sum:
$S((I+\alpha_1A_1) \otimes S(I+i\alpha_2A_2))$

We can see that the superoperator acts separately on each qubit so we can look at in in two parts. By calculating out the matrices we can see that the imaginary term is just the S gate (for $\alpha=+1$) and the $S^\dag$ gate for $\alpha=-1$. Similarly we can see that the real term is the $\ket{0}\bra{0}$ projector for $\alpha=+1$ and the $\ket{1}\bra{1}$ for $\alpha=-1$. The total operation from the third term then becomes either measure + apply S/$S^\dag$ or apply S/$S^\dag$ + measure.

----

Now we can get the full decomposition by applying the extra terms from equation 1, which are also just S gates and calculating the coefficients. The full decomposition is then:

| Qubit 1 | Qubit 2 | Coef |
| :---    |  :---:  | ---: |
| S       | S       |  1/2 |
| Z S     | Z S     |  1/2 |
| Meas    | Z       | -1/2 |
| Meas    | I       |  1/2 | 
| Z       | Meas    | -1/2 |
| I       | Meas    |  1/2 | 

Here Meas is the computational basis measurement and Z S means first applying the Z gate and the the S gate. I is the identity gate.

## References
1. K. Mitarai and K. Fujii, "Constructing a virtual two-qubit gate by sampling single-qubit operations", New J. Phys., vol. 23, no. 2, p. 023021, Feb. 2021, doi: 10.1088/1367-2630/abd7bc.