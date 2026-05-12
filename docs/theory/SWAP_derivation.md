# Swap derivation

**Naively one could decompose a SWAP gate into 3 CZ gates (plus local unitaries) and decompose those using the optimal CZ QPD giving a decomposition with $\gamma=3^3=27$, sampling overhead of $27^2$, and 216 elements in the QPD. However, luckily this is far from optimal. Utilising KAK decomposition and some tricks from [[1]](#ref) we can achieve a decomposition with $\gamma=7$, sampling overhead of 49, and 34 elements in QPD. In this document we'll go over this decomposition. The same method can also be used to optimally decompose othe two qubit gates.** 

## Pauli expansion

We start by expressing the SWAP gate in Pauli tensor basis:

(eq1)=
\begin{equation}
    SWAP = \sum_{\alpha,\beta=0}^{3}c_{\alpha\beta}\sigma_\alpha\otimes\sigma_\beta
\end{equation}

where $c_\alpha\beta = \frac{1}{4}Tr[(\sigma_\alpha \otimes \sigma\beta)SWAP]$

Evaluating the coefficients:

$$
\begin{aligned}
    Tr[(\sigma_\alpha \otimes \sigma\beta)SWAP] &= \sum_{ij}\bra{i}\sigma_\alpha\ket{j}\bra{j}\sigma_\beta\ket{i} \\
    &= \sum_{ij}(\sigma_\alpha)_{ij}(\sigma_\beta)_ji \\
    &= Tr[\sigma_\alpha\sigma_\beta] \\
    &= 2\delta_{\alpha\beta} \\
    &=  \begin{cases}
            0 \text{, if } \alpha \neq \beta\\
            2 \text{, if } \alpha = \beta
        \end{cases}
\end{aligned}
$$

Therefore $c_{\alpha\beta}=\frac{1}{4}2\delta_{\alpha\beta}=\frac{1}{2}\delta_{\alpha\beta}$

Now expanding out [equation 1](#eq1) gives:

(eq2)=
\begin{equation}
    SWAP = \frac{1}{2}(I\otimes I + X\otimes X + Y\otimes Y + Z\otimes Z)
\end{equation}

Note that SWAP is already diagonal in the Pauli basis.

## KAK decomposition

Any two qubit unitary can be written as [[2]](#ref):

(eq3)=
\begin{equation}
    U = (K_1 \otimes K_2)\exp[-i(\theta_1X\otimes X + \theta_2Y\otimes Y + \theta_3Z\otimes Z)](K_3 \otimes K4)
\end{equation}

Where $K_i$ are local unitaries and $\theta_i$ are the non-local paramters. For swap gate the local parts are trivial. Since SWAP is already diagonal in the Pauli basis $K_i=I$. [Equation 3](#eq3) then simplifies to just $\exp[-i(\theta_1X\otimes X + \theta_2Y\otimes Y + \theta_3Z\otimes Z)]$. To evaluate this we note that the Pauli terms commute and they all square to identity and write:

\begin{equation}
    \begin{aligned}
        & \exp[-i(\theta_1X\otimes X + \theta_2Y\otimes Y + \theta_3Z\otimes Z)] \\
        &= e^{-\theta_1XX}e^{-\theta_2YY}e^{-\theta_3ZZ} \\
        &= \prod^3_{k=1}[\cos(\theta_k)II-i\sin(\theta_k)\sigma_k\sigma_k] \\
        & \text{Now using } (\sigma_j \otimes \sigma_j)(\sigma_k \otimes \sigma_k)=(\sigma_j\sigma_k)\otimes(\sigma_j\sigma_k) \\
        & \text{and the Pauli multiplication rules} \\
        &= u_0 II + u_1 XX + u_2 YY + u_3 ZZ
    \end{aligned}
\end{equation}

Where:
$$
u_0 = c_1c_2c_3-is_1s_2s_3 \\
u_1 = -is_1c_2c_3 + c_1s_2s_3 \\
u_2 = s_1c_2s_3 - ic_1s_2c_3 \\
u_3 = s_1s_2c_3 - ic_1c_2s_3
$$

with $s_i=\sin\theta_i$ and $c_i=\cos\theta_i$

Now from [equation 2](#eq2) we know that we want $u_\alpha=1/2$ for all $\alpha$. By symmetry if $\theta_1=\theta_2=\theta_3$ then $u_1=u_2=u_3$ and the coefficients simplify:

$$
u_0 = c^3-is^3 \\
u_1 = -isc^2 + cs^2 \\
u_2 = s^2c - ic^2s \\
u_3 = s^2c - ic^2s
$$

So now $u_1=u_2=u_3=cs^2-ic^2s$

Now by taking $\theta=\pi/4$ for all we get:
$$
u_0 = \cos(\pi/4)^3 - i \sin(\pi/4)^3 = \frac{1}{\sqrt(2)^3} - \frac{i}{\sqrt(2)^3} = \frac{1}{2}e^{-i\pi/4}
$$

Similarly $u_1=u_2=u_3=\cos(\pi/4)\sin(\pi/4)^2-i\cos(\pi/4)^2\sin(\pi/4) =  \frac{1}{2}e^{-i\pi/4}$

So $u_\alpha=\frac{1-i}{2\sqrt(2)}$

## Superoperator expansion

We can expand a superoperator acting on a two qubit unitary as:

(eq4)=
\begin{equation}
    \begin{aligned}
        U(\rho) &= U\rho U^\dagger \\
        &=\sum_{\alpha,\alpha'}u_\alpha u_{\alpha'}^*(\sigma_\alpha \otimes \sigma\alpha)\rho(\sigma_\alpha' \otimes \sigma\alpha') \\
        & \text{Define single qubit superoperator } U_{\alpha\alpha'}\rho=\sigma_\alpha\rho\sigma{_\alpha'} \\
        &= \sum_{\alpha, \alpha'}u_\alpha u_{\alpha'}^*U_{\alpha\alpha'}^{\otimes 2}
    \end{aligned}
\end{equation}

The issue now becomes that $U_{\alpha\alpha'}$ is not a physical channel and thus needs to be decomposed further into implementable operations. We can do this by splitting it into symmetric and antisymmetric parts:

$$
U_{\alpha\alpha'} = A_{\alpha\alpha'}+iB_{\alpha\alpha'}
$$

where:
$$
A_{\alpha\alpha'}\rho = \frac{1}{2}(\sigma_\alpha\rho\sigma_{\alpha'}+ \sigma_{\alpha'}\rho\sigma_{\alpha}), \\
B_{\alpha\alpha'}\rho = \frac{1}{2i}(\sigma_\alpha\rho\sigma_{\alpha'}- \sigma_{\alpha'}\rho\sigma_{\alpha}),
$$

Each of these can be decomposed to rank 1 Kraus maps:

$$
A_{\alpha\alpha'} = A_{\alpha\alpha'}^+ - A_{\alpha\alpha'}^- \\
A_{\alpha\alpha'}^\pm\rho = \frac{\sigma_\alpha\pm\sigma_{\alpha'}}{2}\rho\frac{\sigma_{\alpha}\pm\sigma_{\alpha'}}{2}
$$

$$
B_{\alpha\alpha'} = B_{\alpha\alpha'}^+ - B_{\alpha\alpha'}^- \\
B_{\alpha\alpha'}^\pm\rho = \frac{\sigma_\alpha\pm i\sigma_{\alpha'}}{2}\rho\frac{\sigma_{\alpha}\pm i\sigma_{\alpha'}}{2}
$$

Now each $A^\pm$ and $B^\pm$ is a physical channel that we can interpret as local operations.

## Kraus Operators to Gates

We can split the operators in to four families:

### Family 1: $A^\pm$, $\alpha=0$

The operarator is now $A_{0\alpha'} = \frac{I+\sigma_{\alpha'}}{2}$. These are projectors to the eigenstate of $\sigma_{\alpha'}$:

| Channel     | Operator                                            |  Circuit |
| :-----      | :-----:                                             |  -----:  |
| $A^\pm_{0Z}$  | $\ket{0}\bra{0} \\ \text{or} \ket{1}\bra{1}$      |  Meas    |
| $A^\pm_{0X}$  | $\ket{+}\bra{+} \\ \text{or} \ket{-}\bra{-}$      | H Meas H |
| $A^\pm_{0Y}$  | $\ket{+i}\bra{+i} \\ \text{or} \ket{-i}\bra{-i}$  | Sx Meas $Sx^\dagger$ |

Since the measurements implement both $\pm$ channels in single operation this family contributes 3 terms to the total.

### Family 2: $B^\pm$, $\alpha=0$

Operators are $\frac{I\pm i\sigma_{\alpha'}}{2}$. These are $\pm\pi/2$ rotations around axis $\alpha'$,  $\frac{I\pm i\sigma_{\alpha'}}{2} \propto \exp(\pm i \pi/4 \sigma_{\alpha'}) = R_{\alpha'}(\mp\pi/2)$.

| Channel     | Operator                       |   Circuit   |
| :-----      | :-----:                        |   -----:    |
| $B^+_{0Z}$  |  $\propto \exp(i \pi/4 Z)$     | $S^\dagger$ |
| $B^-_{0Z}$  |  $\propto \exp(- i \pi/4 Z)$   | $S$         |
| $B^+_{0X}$  |  $\propto \exp(  i \pi/4 X)$   | $SX^\dagger$|
| $B^-_{0X}$  |  $\propto \exp( -i \pi/4 X)$   | $SX$        |
| $B^+_{0Y}$  |  $\propto \exp(  i \pi/4 Y)$   | $Ry(-\pi/2)$|
| $B^-_{0Y}$  |  $\propto \exp( -i \pi/4 Y)$   | $Ry( \pi/2)$|


Since $B_{0\alpha'} = B_{0\alpha'}^+ + B_{0\alpha'}^-$, and there are three $\pm$ pairs with possible signs of ++, +-, -+, -- making the total number of terms contributed is 12.

### Family 3: $A^\pm$, $\alpha\neq\alpha'\neq 0$

Now the operators are $\frac{\sigma_\alpha\pm\sigma_{\alpha'}}{2}$. These are again single qubit rotations.

| Channel     | Operator             | Circuit  |
| :-----      | :-----:              |-----:    |
| $A^+_{XY}$  |  $\frac{X+Y}{2}$     |   S Y    |
| $A^-_{XY}$  |  $\frac{X-Y}{2}$     | Z S Y    |
| $A^+_{XZ}$  |  $\frac{X+Z}{2}$     |   H      |
| $A^-_{XZ}$  |  $\frac{X-Z}{2}$     |   Y H    |
| $A^+_{YZ}$  |  $\frac{Y+Z}{2}$     |   SX Z   |
| $A^-_{YZ}$  |  $\frac{Y-Z}{2}$     | X SX Z   |

Total number of contributed terms is 12 as above.

### Family 4: $B^\pm$, $\alpha\neq\alpha'\neq 0$

The operators are $\frac{\sigma_\alpha \pm i\sigma_{\alpha'}}{2}$. Using $\sigma_\alpha\sigma_{\alpha'}=i\epsilon_{\alpha\alpha'\beta}\sigma_\beta$ we can write the operators as:

\begin{align}
    \frac{\sigma_\alpha \pm i\sigma_{\alpha'}}{2} &= \frac{\sigma_\alpha(I \pm i \sigma_\alpha\sigma_{\alpha'})}{2} \\
    &= \sigma_\alpha \frac{I \pm \epsilon_{\alpha \alpha' \beta}\sigma_\beta}{2} \\
    & \text{For cyclic ordering } \epsilon=+1 \\
    &= \sigma_\alpha \frac{I \pm \sigma_\beta}{2}
\end{align}

This is now equal to measuring in the $\sigma_\beta$ basis and applying $\sigma_\alpha$.

E.g $B^\pm_{XY}$: $\frac{X\pm Y}{2}=X\frac{I \pm Z}{2}=X\ket{0}\bra{0} \text{ or } X\ket{1}\bra{1}$ 

| Channel     | Operator                                      | Circuit     |
| :-----      | :-----:                                       |-----:       |
| $B^\pm_{XY}$  |  $X\ket{0}\bra{0}$ or $X\ket{1}\bra{1}$     |   Meas X    |
| $B^\pm_{yz}$  |  $Y\ket{+}\bra{+}$ or $Y\ket{-}\bra{-}$     | H Meas H Y  |
| $B^\pm_{xz}$  |$Z\ket{+i}\bra{+i}$ or $Z\ket{-i}\bra{-i}$   | SX Meas $SX^\dagger$ Z|

Since the measurements implement both $\pm$ channels in single operation this family contributes 3 terms to the total.

## Applying equation 19 of [[1]](#ref)

From [[1]](#ref) equation 19 with $u_\alpha u_{\alpha'}^*$=1/4 (all real and equal):

\begin{equation}
    U=\sum_\alpha|u_\alpha|^2\sigma_\alpha^{\otimes 2} + \sum_{\alpha<\alpha'}2Re(u_\alpha u_{\alpha'}^*)(A_{\alpha\alpha'}^{\otimes2}-B_{\alpha\alpha}^{\otimes 2})
\end{equation}

Note that here we omit the last term of the equation since for SWAP it is 0. However if deriving the QPD for e.g an iSWAP gate the last term would also be relevant.

The first term is simple for it just applies a Pauli gate to both qubits with coefficient of $|u_\alpha|^2=\frac{1}{4}$. The entries are then:

| Qubit 1 | Qubit 2 | Coef |
| :---    |  :---:  | ---: |
| I       | I       |  1/4 |
| Z       | Z       |  1/4 |
| Y       | Y       |  1/4 |
| X       | X       |  1/4 | 

For term two we can again treat it as four Kraus families. Note that the base coefficient here is $2 u_\alpha u_{\alpha'}^*=1/2$.

### Family 1

Operator normalised, no extra terms. Coeffcient is 1/2:

| Qubit 1              | Qubit 2               | Coef|
| :-----               | :-----:               | ---:|
| Meas                 |Meas                   | 1/2 |
|H Meas H              | H Meas H              | 1/2 |
| Sx Meas $Sx^\dagger$ | Sx Meas $Sx^\dagger$  | 1/2 |

### Family 2

Expanding $(B^+-B^-)$ gives $B^+B^+-B^+B^--B^-B^++B^-B^-$, the $\pm$ factor in the terms influence the respective coefficient. Additionally each $B^\pm$ Kraus operator $K=(I+i\sigma)/2$ satisfies $K^\dagger K=I/2$ which contributes a factor of 1/2 per qubit so 1/4 total. Additionally the $B$ term has a -1 prefactor to take into account in coefficient. The full coefficient is then $\pm1/8$. For each $\sigma$ we get 4 terms, 12 in total

| Qubit 1 | Qubit 2 | Coef |
| :---    |  :---:  | ---: |
| S       | S       |  -1/8 |
| S       | $S^\dagger$       | 1/8 |
| $S^\dagger$       | S       |  1/8 |
| $S^\dagger$       | $S^\dagger$       |  -1/8 | 
| SX       | SX       |  -1/8 |
| SX       | $SX^\dagger$       | 1/8 |
| $SX^\dagger$       | SX       |  1/8 |
| $SX^\dagger$       | $SX^\dagger$       |  -1/8 | 
| $Ry(-\pi/2)$       | $Ry(-\pi/2)$       |  -1/8 |
| $Ry(-\pi/2)$       | $Ry(\pi/2)$       | 1/8 |
| $Ry(\pi/2)$       | $Ry(-\pi/2)$       |  1/8 |
| $Ry(\pi/2)$       | $Ry(\pi/2)$       |  -1/8 | 

### Family 3

Similarly as for family two we exapand and get $\pm1/8$ just without the extra -1 prefactor so the terms are:

| Qubit 1     | Qubit 2     | Coef     |
| :-----      | :-----:     |-----:    |
| S Y         |  S Y        |   1/8    |
| S Y         |  Z S Y      |  -1/8    |
|  Z S Y      |  S Y        |  -1/8    |
|  Z S Y      |   Z S Y     |   1/8    |
| H           |  H          |   1/8    |
| H           |  Y H        | -1/8     |
|  Y H        |  H          |  -1/8    |
|  Y H        |   Y H       |   1/8    |
| SX Z        |  SX Z       |   1/8    |
| SX Z        |  X SX Z     |  -1/8    |
|   X SX Z    |  SX Z       |  -1/8    |
|  X SX Z     |   X SX Z    |   1/8    |


### Family 4

Similarly as for family one but with extra -1 prefactor:

| Qubit 1              | Qubit 2               | Coef |
| :-----               | :-----:               | ---: |
| Meas X               | Meas X                | -1/2 |
|H Meas H Y            | H Meas H Y            | -1/2 |
|Sx Meas $Sx^\dagger$ Z|Sx Meas $Sx^\dagger$ Z | -1/2 |


Thus we have derived the whole 34 term QPD with $\gamma=7$ which is optimal (without communication). As a sanity check we can see that the sum of absolute coefficients is 7: $\sum|c| = 3*1/2+12*1/8+12*1/8+3*1/2+4*1/4=7$

(ref)=
## References
1. K. Mitarai and K. Fujii, ‘Overhead for simulating a non-local channel with local channels by quasiprobability sampling’, 2020, doi: 10.48550/ARXIV.2006.11174.
2. R. R. Tucci, ‘An Introduction to Cartan’s KAK Decomposition for QC Programmers’, 2005, arXiv. doi: 10.48550/ARXIV.QUANT-PH/0507171.