# Cutting parallel rotation gates together

**Cutting gates one at a time costs the product of their sampling overheads. For gates equivalent to a rotation about a single Cartan axis that is not optimal, and one joint decomposition over `n` of them costs strictly less. Two CNOT gates cost $\gamma = 7$ rather than $9$, three cost $15$ rather than $27$. The number of experiment circuits drops as well. No ancilla qubits and no classical communication are needed. The implementation is `QCut/qpd_joint.py`, with the grouping logic in `QCut/bundle.py`.**

## Why separate cuts are not optimal

Every other page here decomposes one gate. QCut then combines the per-cut tables with a Cartesian product and multiplies their coefficients, so `n` cuts cost $\prod_s\gamma_s$. That is the best possible when each cut is treated on its own, but the quantity being decomposed is the joint channel $\bigotimes_s R_{zz}(\theta_s)$, not `n` separate ones, and the joint channel has a cheaper decomposition.

Writing the two-qubit rotation gate as

$$
R_{zz}(\theta) = \cos(\theta/2)\, I\otimes I - i\sin(\theta/2)\, Z\otimes Z
$$

the single-gate optimum is $\gamma = 1 + 2|\sin\theta|$, which the [general two-qubit derivation](General_2q_derivation.md) already attains. Cutting `n` of them jointly instead costs

(joint-gamma)=
$$
\gamma_{\mathrm{joint}} = 2\prod_{s=1}^{n}\bigl(1 + |\sin\theta_s|\bigr) - 1
\;<\;
\prod_{s=1}^{n}\bigl(1 + 2|\sin\theta_s|\bigr) = \gamma_{\mathrm{ind}}
$$

and [[1]](#joint-ref) proves this is optimal. For $n=1$ the two agree, as they must. The saving grows with the angle, so it is largest exactly where cutting hurts most. At $\theta = \pi/2$, where the gate is CNOT-equivalent, the effective cost per gate falls from $3$ to $2$.

Shot count scales as $\gamma^2$, so three parallel CNOT gates become $(27/15)^2 \approx 3.2$ times cheaper and four become $(81/31)^2 \approx 6.8$ times cheaper.

## The construction

Index the gates by $s$ and write $j\in\{0,1\}^n$ for a bit per gate, with

$$
c_j = \prod_{s=1}^n c_{j_s},\qquad
c_{j_s} = \begin{cases}\cos(\theta_s/2) & j_s = 0\\ \sin(\theta_s/2) & j_s = 1\end{cases}
$$

Let $T_{ij} = \{s : i_s \neq j_s\}$ be the gates where two bit patterns differ, and $\nu_{ij} = \sum_s (j_s - i_s)$. Define three families of local operations, each acting on one partition's qubits.

$\mathcal{Z}_j$ applies $Z$ to the qubit of every gate with $j_s = 1$.

$\mathcal{R}_{ij}$ is built from the multi-qubit rotation

$$
R_{ij}(\pm\pi/2) = \frac{I^{\otimes n} \mp i\, Z^{\otimes T_{ij}}}{\sqrt 2},
\qquad
\mathcal{R}_{ij} = \frac{\mathcal{R}_{ij}(\pi/2) - \mathcal{R}_{ij}(-\pi/2)}{2}
$$

$\mathcal{P}_{ij}$ is the parity measurement over $T_{ij}$, meaning $\mathcal{P}_{ij} = \mathcal{P}^0_{ij} - \mathcal{P}^1_{ij}$ with projectors $P^k_{ij} = (I^{\otimes n} + (-1)^k Z^{\otimes T_{ij}})/2$.

Equation (C16) of [[1]](#joint-ref) is then

(eqc16)=
$$
\bigotimes_s R_{zz}(\theta_s)
= \sum_{j} c_j^2\, \mathcal{Z}_j\otimes\mathcal{Z}_j
+ \sum_{i>j} 2 c_i c_j\, [\mathcal{Z}_i\otimes\mathcal{Z}_i]\circ\mathcal{B}_{ij}
$$

$$
\mathcal{B}_{ij} =
\begin{cases}
(-1)^{\nu_{ij}/2}\bigl[\mathcal{P}_{ij}\otimes\mathcal{P}_{ij} - \mathcal{R}_{ij}\otimes\mathcal{R}_{ij}\bigr] & \nu_{ij}\ \text{even}\\[2pt]
(-1)^{(\nu_{ij}-1)/2}\bigl[\mathcal{R}_{ij}\otimes\mathcal{P}_{ij} + \mathcal{P}_{ij}\otimes\mathcal{R}_{ij}\bigr] & \nu_{ij}\ \text{odd}
\end{cases}
$$

where the ordering $i > j$ reads each bit pattern as an integer with gate $1$ most significant. The ordering matters, because swapping $i$ and $j$ flips the sign of $\nu_{ij}$ and changes the $\mathcal{Z}_i$ prefactor.

Summing the absolute coefficients gives [equation above](#joint-gamma). Both $\mathcal{R}_{ij}$ and $\mathcal{P}_{ij}$ cost $\gamma = 1$ of their own, the first because it is a difference of two unitary channels weighted $\pm 1/2$, the second because the measurement outcome supplies the sign in post-processing. That is the same trick the `Bxy`, `Byz` and `Bzx` primitives already use.

The paper arranges the ancilla qubits so that the second partition carries the reversed bit string $\tilde{\jmath} = (j_n,\dots,j_1)$. That is bookkeeping for its figure. Gate $s$ contributes the same bit $j_s$ to both of its qubits, so once operations are indexed by gate rather than by qubit position the reversal disappears.

## Term counts

The double sum runs over $\binom{2^n}{2}$ pairs. A pair with even $\nu_{ij}$ contributes five terms and one with odd $\nu_{ij}$ contributes four, on top of the $2^n$ diagonal terms.

| gates | joint terms | separate terms | $\gamma$ at $\theta=\pi/2$ |
| :--- | :--- | :--- | :--- |
| 1 | 6 | 6 | 3 against 3 |
| 2 | 30 | 36 | 7 against 9 |
| 3 | 132 | 216 | 15 against 27 |
| 4 | 552 | 1296 | 31 against 81 |

So joint cutting lowers the circuit count as well as the overhead. Nothing has to be traded off, which is why it is on by default.

## Getting a gate into the frame

A gate qualifies when its Cartan coordinates have a single non-zero entry, which covers `rzz`, `rxx`, `ryy`, `rzx`, the controlled rotations, `cp`, `cx`, `cz` and `ecr`. `SWAP`, `iSWAP`, `DCX` and `xx_plus_yy` have two or three non-zero coordinates and are cut individually.

`TwoQubitWeylDecomposition` returns the interaction as $\exp[i\,a\,X\otimes X]$ for such a gate, so a Hadamard on both qubits turns it into $\exp[i\,a\,Z\otimes Z] = R_{zz}(-2a)$. That basis change is folded into the local unitaries alongside $K_{1l},K_{1r},K_{2l},K_{2r}$, exactly as in the [general derivation](General_2q_derivation.md). The Weyl chamber folds the coordinate into $[0,\pi/4]$, which changes the angle reported but not the gate, and $\gamma$ is a local invariant so the fold cannot change the cost.

Because $Z\otimes Z$ is symmetric under exchanging its two qubits, a cut whose two sides run the opposite way round to its neighbours still joins the bundle. Only its own two local unitaries swap over.

## What has to line up

The joint operations act on several qubits of one partition at once, which constrains when a bundle can form.

Every gate in a bundle must join the **same pair of subcircuits**, so that each side's operation has one subcircuit to live in. Two cuts that happen to leave the circuit in four disconnected pieces cannot be bundled, because there is no subcircuit holding both of one side's qubits.

The gates must be **parallel**, meaning schedulable in one time slice. QCut checks the equivalent condition on the split subcircuits, that every one of the bundle's placeholders can slide to a common point without crossing anything on its own qubit. Both the earliest and the latest placeholder are tried as that common point, since either can be the one that works. A local gate sitting between two cuts on a shared wire blocks both directions and splits the bundle.

## Relation to consolidation

[Merging gates on the same qubit pair](../Options.rst) and joint cutting are different optimisations that mostly compose. Consolidation merges gates that act on the **same** pair one after another, turning several cuts into one. Joint cutting groups gates on **disjoint** pairs that act at the same time, turning several cuts into one decomposition.

They do conflict in one case. Merging a run of gates about *different* axes composes them into a generic two-qubit unitary, which is no longer a single-axis rotation and so can no longer join a bundle. Cutting them separately and bundling each with its parallel partners can then be cheaper than merging. Two examples on four qubits, with pair $(0,2)$ carrying the run and pair $(1,3)$ a parallel partner:

| run on $(0,2)$ | partner on $(1,3)$ | merged | separate and bundled |
| :--- | :--- | :--- | :--- |
| `rzz(0.4)`, `rzz(0.4)` | none | $\gamma = 2.43$ | $\gamma = 3.16$ |
| `cp(1.15)`, `ryy(0.758)` | `rzz(1.768)` | $\gamma = 12.47$ | $\gamma = 11.87$ |

Which way wins depends on the whole circuit, so `consolidate="auto"` splits it both ways, costs each plan in full including the bundles it allows, and keeps the cheaper. The comparison is exact rather than a heuristic, because `QCut.qpd_operations.plan_cost` gets every $\gamma$ from a closed form and never has to build a QPD table to do it. In a randomised search over 550 layered circuits, merging was the wrong call 18 times and `"auto"` picked correctly every time.

## References

(joint-ref)=
1. C. Ufrecht, L. S. Herzog, D. D. Scherer, M. Periyasamy, S. Rietsch, A. Plinge and C. Mutschler, *Optimal joint cutting of two-qubit rotation gates*, Phys. Rev. A **109**, 052440 (2024), [arXiv:2312.09679](https://arxiv.org/abs/2312.09679).
2. L. Schmitt, C. Piveteau and D. Sutter, *Cutting circuits with multiple two-qubit unitaries*, Quantum **9**, 1634 (2025), [arXiv:2312.11638](https://arxiv.org/abs/2312.11638), which independently observes that joint cutting beats the product of single-gate costs.
