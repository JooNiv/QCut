# Deriving a QPD for an arbitrary two-qubit gate

**The [SWAP derivation](SWAP_derivation.md) is already almost the general recipe, but it stops one term short. Restoring that term and taking the local unitaries from the KAK decomposition gives a QPD for any two-qubit gate, generated from the gate's matrix. This page covers what changes relative to the SWAP page, the two conventions that have to be pinned down, and how far the result is from the proven optimum. The implementation is `QCut/qpd_generate.py`, with the parametrised-gate machinery in `QCut/qpd_analytic.py`.**

## The term SWAP lets us drop

The SWAP page applies equation 19 of [[1]](#references) in the form

(eq19)=
$$
U=\sum_\alpha|u_\alpha|^2\sigma_\alpha^{\otimes 2} + \sum_{\alpha<\alpha'}2\mathrm{Re}(u_\alpha u_{\alpha'}^*)(A_{\alpha\alpha'}^{\otimes2}-B_{\alpha\alpha'}^{\otimes 2})
$$

and notes that the last term of the equation is omitted because it vanishes for SWAP. In full, equation 19 has a third line.

(eq19-3)=
$$
U=\sum_\alpha|u_\alpha|^2\sigma_\alpha^{\otimes 2} + \sum_{\alpha<\alpha'}2\mathrm{Re}(u_\alpha u_{\alpha'}^*)(A_{\alpha\alpha'}^{\otimes2}-B_{\alpha\alpha'}^{\otimes 2}) + \sum_{\alpha<\alpha'}\mathrm{Im}(u_\alpha u_{\alpha'}^*)(A_{\alpha\alpha'}\otimes B_{\alpha\alpha'} + B_{\alpha\alpha'}\otimes A_{\alpha\alpha'})
$$

Why it disappears for SWAP is due to SWAP having KAK coordinates of $(\pi/4,\pi/4,\pi/4)$, and the SWAP page computes $u_\alpha = \frac{1}{2}e^{-i\pi/4}$ for all four $\alpha$. All four coefficients carry the same phase, so every pair product $u_\alpha u_{\alpha'}^*$ is real and positive and $\mathrm{Im}(u_\alpha u_{\alpha'}^*)=0$ for all six pairs.

That is special to SWAP. For iSWAP, $u = (\tfrac12, -\tfrac{i}{2}, -\tfrac{i}{2}, \tfrac12)$, the phases differ and four of the six pair products are purely imaginary. CZ is the opposite extreme, where the only non-zero pair product is imaginary, so CZ lives entirely in the third line.

The four Kraus families and the operator-to-gate dictionary in the SWAP page are unchanged. The third line just pairs an $A$ family against a $B$ family on the two qubits instead of pairing a family with itself.

## Where the coefficients come from

Only the first two steps of the SWAP derivation change. Rather than matching $u_\alpha$ against a known Pauli expansion, we read the KAK decomposition off the gate,

(kak-decomp)=
$$
U = e^{i\phi}(K_{1l}\otimes K_{1r})\exp[i(a\, X\otimes X + b\, Y\otimes Y + c\, Z\otimes Z)](K_{2l}\otimes K_{2r})
$$

where Qiskit's `TwoQubitWeylDecomposition` supplies $a,b,c$ and the four local unitaries. The $u_\alpha$ then follow from the same identity the SWAP page derives,

(swap-id)=
$$
\exp[i(a\,XX + b\,YY + c\,ZZ)] = u_0 II + u_1 XX + u_2 YY + u_3 ZZ
$$

which is diagonalised by writing the exponent's summation as the real symmetric matrix

(diag-matrix)=
$$
\begin{pmatrix} 0&a&b&c \\ a&0&-c&-b \\ b&-c&0&-a \\ c&-b&-a&0 \end{pmatrix}
$$

whose eigenvectors do not depend on $a,b,c$, so only the eigenvalues need exponentiating.

The local unitaries are prepended and appended to every row of the resulting table, with $K_{2r}$ and $K_{1r}$ going on the operations acting on the gate's first qubit and $K_{2l}$ and $K_{1l}$ on the second. The global phase $e^{i\phi}$ cancels because a QPD represents a channel.

## Two conventions to note

These are invisible for SWAP but change the answer for other gates.

**Sign of the KAK exponent.** The SWAP page writes $\exp[-i(\theta_1 XX + \theta_2 YY + \theta_3 ZZ)]$ with a minus sign, while `TwoQubitWeylDecomposition` uses a plus. The two differ by complex conjugation of $u$, which negates every $\mathrm{Im}(u_\alpha u_{\alpha'}^*)$ and flips the sign of the whole third line. For SWAP this is unobservable because those terms are all zero. `QCut.qpd_generate` works in the $+i$ convention throughout.

**Plus and minus labelling of the Kraus pairs.** The rank-one maps come in $\pm$ pairs, and in `QCut/qpd_gates.py` the members of the $A_{XY}$ and $A_{YZ}$ pairs are labelled the opposite way round from the paper. The circuit named `axyp` implements $A^-_{XY}$, and likewise `ayzp` implements $A^-_{YZ}$. Everything else matches directly, including `azxp`/`azxm`, the $B_{0\alpha}$ pairs, and all the measurement-based operations.

This matters because the second and third lines of [equation 1](#eq19) treat the label differently. The second line contains

(eq1-2)=
$$
(A^+-A^-)^{\otimes 2} = A^+A^+ - A^+A^- - A^-A^+ + A^-A^-
$$

which is invariant under exchanging the $+$ and $-$ labels, so a crossed label is undetectable there. The third line contains terms like $\mathrm{Im}(u_\alpha u_{\alpha'}^*)\,(A^+_{\alpha\alpha'} - A^-_{\alpha\alpha'})\otimes B_{\alpha\alpha'}$, which is antisymmetric in the label and changes sign if the pair is crossed.

## Sampling overhead, and how close it is to optimal

Summing the absolute coefficients of [equation 1](#eq19), and using $\sum_\alpha|u_\alpha|^2=1$, gives

(gamma1)=
$$
\gamma_{\mathrm{MF}} = 1 + 4\sum_{\alpha<\alpha'}\left(|\mathrm{Re}(u_\alpha u_{\alpha'}^*)| + |\mathrm{Im}(u_\alpha u_{\alpha'}^*)|\right)
$$

The provably optimal value, from [[2]](#gen-ref), is

(gamma2)=
$$
\gamma_{\mathrm{opt}} = 2\left(\sum_\alpha|u_\alpha|\right)^2 - 1 = 1 + 4\sum_{\alpha<\alpha'}|u_\alpha u_{\alpha'}^*|
$$

Comparing term by term, the two agree exactly when every pair product is purely real or purely imaginary. Otherwise $\gamma_{\mathrm{MF}}$ is larger, bounded by $|\mathrm{Re}\,z| + |\mathrm{Im}\,z| \le \sqrt2|z|$, so

(gamma3)=
$$
\gamma_{\mathrm{MF}} - 1 \le \sqrt2\,(\gamma_{\mathrm{opt}} - 1)
$$

The equality case covers every gate one is likely to cut.

| gate | terms | $\gamma$ | optimal |
| :--- | ---: | ---: | :--- |
| CZ, CX | 6 | 3 | yes |
| SWAP | 34 | 7 | yes |
| iSWAP, DCX | 30 | 7 | yes |
| $R_{ZZ}(\theta)$ | 6 | $1+2\lvert\sin\theta\rvert$ | yes |
| $CR_Z(\theta)$ | 6 | $1+2\lvert\sin(\theta/2)\rvert$ | yes |
| $XX+YY$ | 30 | | yes |
| generic $SU(4)$ | 58 | | no |

So only a generic $SU(4)$ pays for the simplicity. A decomposition attaining $\gamma_{\mathrm{opt}}$ in that case exists in [[2]](#gen-ref) but is not implemented here because it introduces two extra ancilla qubits per cut. `QCut.qpd_generate.optimality_gap` reports how much a given gate would stand to gain, which is a cheap closed-form check that does not require building the table.

## Parametrised gates and the Weyl chamber

A QPD's coefficients depend on the gate's numeric parameters, so a gate with an unbound parameter cannot be decomposed directly. QCut still lets you split such a circuit into subcircuits, because the split replaces the gate with opaque placeholders and is parameter-agnostic. Only the experiment circuits require the parameter to be bound.

(gen-ref)=
## References

1. K. Mitarai and K. Fujii, 'Overhead for simulating a non-local channel with local channels by quasiprobability sampling', Quantum 5, 388 (2021), doi: 10.48550/ARXIV.2006.11174. Equation 19, section 2.3.
2. L. Schmitt, C. Piveteau and D. Sutter, 'Cutting circuits with multiple two-qubit unitaries', Quantum 9, 1634 (2025), doi: 10.22331/q-2025-02-18-1634.
3. R. R. Tucci, 'An Introduction to Cartan's KAK Decomposition for QC Programmers', 2005, arXiv. doi: 10.48550/ARXIV.QUANT-PH/0507171.
