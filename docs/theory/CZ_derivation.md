# Deriving optimal QPD for the CZ gate

**Here we will briefly go over where the decomposition for the CZ gate used comes from and how that is transformed to operations implementable on a quantum computer. The derivation is based on [[1]](#cz-ref).**

## Background

The goal is to decompose a quantum channel implementing the CZ gate into a set of local channels implementable as single qubits operations. This decomposed channel can then be sampled to approximate the expectation values of the original channel.

## Factor the CZ gate

The CZ gate can (up to global phase) be decomposed as:

(eq1)=
\begin{equation}
CZ = \exp(i\pi Z \otimes I/4) \cdot \exp(i\pi I \otimes Z/4) \cdot \exp(-i\pi Z \otimes Z/4) 
\end{equation}

We can immediately see that the first two channels are just local single qubit Z rotations, S gates in this case. The only non-local part is then the third term $\exp(-i\pi Z \otimes Z/4)$ and so has the form $\exp(i\theta A_1 \otimes A_2)$ with $A_1 = A_2 = Z, \theta = -\pi/4$

## Apply Lemma 1 from [[1]](#cz-ref)

The lemma states that we can decompose the super operator $S(\exp(i\theta A_1 \otimes A_2))$ as:

(eq2)=
\begin{equation}
\begin{aligned}
S(\exp(i\theta A_1 \otimes A_2)) &= \cos^2(\theta)S(I \otimes I) + \sin^2(\theta)S(A_1 \otimes A_2) + 1/8 \cos(\theta)\sin(\theta) \\ 
& \sum_{(\alpha_1, \alpha_2) \in \{\pm 1\}} \alpha_1 \alpha_2 [S((I+\alpha_1A_1) \otimes S(I+i\alpha_2A_2)) + \\
& S((I+i\alpha_1A_1) \otimes S(I+\alpha_2A_2))] 
\end{aligned}
\end{equation}

if $A_1^2 = A_2^2 = I$, which is the case for the pauli operators.

## Transform to gates

In less mathematical terms we can decompose any gate of form $S(\exp(i\theta A_1 \otimes A_2))$ into a set of 6 single qubit operations. Looking at the decomposition we can see that term 1 is just applying identity on both qubits aka doing nothing. The second term is applying $A_1$ on qubit 1 and $A_2$ on qubit 2, remember that A are pauli operators and the total operations of term 2 is then the Z gate on both qubits. The third term is a bit more complex so let us look at it in detail.

### Term 3

Let us look at the structure that repeats twice in the sum:
$S((I+\alpha_1A_1) \otimes S(I+i\alpha_2A_2))$

We can see that the superoperator acts separately on each qubit so we can look at in in two parts. By calculating out the matrices [Appendix 1](#app1) we can see that the imaginary term is just the S gate (for $\alpha=-1$) and the $S^\dagger$ gate for $\alpha=+1$. Similarly we can see that the real term is the $\ket{0}\bra{0}$ projector for $\alpha=+1$ and the $\ket{1}\bra{1}$ for $\alpha=-1$. The total operation from the third term then becomes either measure + apply S/$S^\dagger$ or apply S/$S^\dagger$ + measure.

----

Now we can get the full decomposition by applying the extra terms from [equation 1](#eq1), which are also just S gates and calculating the coefficients [Appendix 2](#app2). The full decomposition is then:

| Qubit 1 | Qubit 2 | Coef |
| :---    |  :---:  | ---: |
| S       | S       |  1/2 |
| Z S     | Z S     |  1/2 |
| Meas    | Z       | -1/2 |
| Meas    | I       |  1/2 | 
| Z       | Meas    | -1/2 |
| I       | Meas    |  1/2 | 

Here Meas is the computational basis measurement and Z S means first applying the Z gate and the the S gate. I is the identity gate. Z S could also be simplified to the $S^\dagger$ gate here since they are equal.

Summing over the absolute coefficients we get $\gamma=3$ which is optimal (without communication).

(cz-ref)=
## References
1. K. Mitarai and K. Fujii, "Constructing a virtual two-qubit gate by sampling single-qubit operations", New J. Phys., vol. 23, no. 2, p. 023021, Feb. 2021, doi: 10.1088/1367-2630/abd7bc.

## Appendix

(app1)=
### 1. Explicitly calculate term 3 matrices

$
\begin{aligned}
Z =& \begin{pmatrix}
    1 & 0 \\ 0 & -1
    \end{pmatrix} \\
\end{aligned}
$,
$
\begin{aligned}
I =& \begin{pmatrix}
    1 & 0 \\ 0 & 1
    \end{pmatrix} \\
\end{aligned}
$

#### Imaginary term
$I+i\alpha A$, with

A = Z, $\alpha = \pm 1$

Expanding gives:
$
\begin{aligned}
& \begin{pmatrix}
    i\pm 1 & 0 \\ 0 & i\mp 1
    \end{pmatrix} \\
&=  (1 \pm i)\begin{pmatrix}
    1 & 0 \\ 0 & \pm i
    \end{pmatrix} \\
\end{aligned}
$

This is exactly equal to the S/$S^\dagger$ gate (up to global phase, which gets absorbed by the coefficient).

#### Real term
$I+\alpha A$, with

A = Z, $\alpha = \pm 1$

Expanding gives:
For $\alpha = +1$

$
\begin{aligned}
& 2\begin{pmatrix}
    1 & 0 \\ 0 & 0
    \end{pmatrix} \\
&= 2\ket{0}\bra{0}
\end{aligned}
$

For $\alpha = -1$

$
\begin{aligned}
& 2\begin{pmatrix}
    0 & 0 \\ 0 & 1
    \end{pmatrix} \\
&= 2\ket{1}\bra{1}
\end{aligned}
$

These are exactly the 0 and 1 projectors (the factor of 2 gets absorbed by the coefficient)

(app2)=
### 2. Calculating coefficients

The coefficients follow immediately from the coefficients of [equation 2](#eq2). Let us go term wise:

#### Term 1

$\cos^2(\pi /4)=\frac{1}{2}$

#### Term 2

$\sin^2(\pi /4)=\frac{1}{2}$

#### Term 3

$\frac{1}{8}\cos(\pi/4)\sin(\pi/4) = \frac{1}{16}$

Nos this gets multiplied by the extra factors from the terms in Appendix A (since they can be moved out of the sum). This then gives us:

$\frac{1}{16}*2^2*|1 + i|^2 = \frac{1}{16}*4*2=\frac{1}{2}$

And finally this is multiplied by the $\alpha_1 \alpha_2$ factor from the sum yielding $\pm \frac{1}{2}$ as the coefficient for term 3 depending on the $\alpha's$.

