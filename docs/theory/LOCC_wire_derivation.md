# Cutting wires with classical communication

**Cutting a wire costs $\gamma = 4$, and cutting $n$ of them costs $4^n$. That is already optimal for local operations alone, so cutting wires as a block buys nothing by itself. Letting the two sides exchange the measured outcome changes it to $2^{n+1} - 1$, and drops the number of channels from $8^n$ to $2^n + 1$. The implementation is `QCut/qpd_locc.py`. What that gains in practice is not the whole $\gamma$ story, and the last section is about why.**

## Why local operations cannot do better

The [wire cut derivation](Wire_derivation.md) decomposes the identity channel into eight measure-and-prepare terms with $\gamma = 4$. That is provably the best possible without communication, and unlike gate cuts it stays that way for a block. Proposition 4.1 of [[1]](#locc-ref) shows

$$
\gamma_{\mathrm{LO}}\bigl(\mathrm{Id}^{\otimes n}\bigr) = 4^n
$$

exactly, so applying the single-wire table $n$ times is already optimal. There is nothing to gain from treating a block jointly, which is the opposite of the situation for [parallel rotation gates](Joint_rotation_derivation.md).

Communication changes that. Proposition 4.2 of the same work gives

(locc-gamma)=
$$
\gamma_{\mathrm{LOCC}}\bigl(\mathrm{Id}^{\otimes n}\bigr) = 2^{n+1} - 1
$$

so one wire costs 3 rather than 4, two cost 7 rather than 16, three cost 15 rather than 64. Note that even a single wire improves, which does not happen for gate cuts.

## The construction

The protocol that attains it and needs no ancilla qubits is Theorem 3 of [[2]](#locc-ref2),

(theorem3)=
$$
\mathrm{Id}^{\otimes n}(\cdot) =
\sum_{i=1}^{2^n}\sum_{j\in\{0,1\}^n}
  \mathrm{Tr}\bigl[U_i|j\rangle\langle j|U_i^\dagger\,(\cdot)\bigr]\;
  U_i|j\rangle\langle j|U_i^\dagger
\;-\;(2^n-1)\sum_j \mathrm{Tr}\bigl[|j\rangle\langle j|\,(\cdot)\bigr]\,\rho_j
$$

with $\rho_j$ the uniform mixture over the computational basis states other than $|j\rangle$. In words, **measure in one of the $2^n+1$ mutually unbiased bases and prepare the state you just measured**, except in the computational basis where you prepare any of the others and flip the sign. Summing the absolute coefficients gives $2^n + (2^n - 1)$, which is [the optimum above](#locc-gamma).

The prepared state depends on the measured outcome. That dependence *is* the classical communication, and it is what makes these cuts run in waves.

### Getting the bases

The $2^n+1$ mutually unbiased bases correspond to a partition of the $4^n-1$ non-identity Pauli strings into $2^n+1$ sets of $2^n-1$ commuting strings, each a maximal isotropic subspace of $\mathbb{F}_2^{2n}$ under the symplectic form. Identifying $\mathbb{F}_2^n$ with $\mathrm{GF}(2^n)$ produces them all at once as

$$
L_\alpha = \{(x \mid \alpha x) : x \in \mathrm{GF}(2^n)\}, \qquad
L_\infty = \{(0 \mid y)\}
$$

one for each field element plus the point at infinity. The subspaces are isotropic because multiplication in the field commutes.

There is a trap here worth stating, because it is invisible at small $n$. Those subspaces are isotropic for the field's **trace form** $\mathrm{Tr}(xy')$, not for the coordinate dot product $a\cdot b'$ that Pauli commutation actually uses. The two agree only up to $n = 2$. Writing the $Z$ component in the trace-dual basis, so its coordinates are $\mathrm{Tr}(e_l\,y)$ rather than the plain polynomial-basis coordinates, turns the trace form into the dot product and makes the construction correct at every width. `QCut.qpd_locc` checks the result rather than trusting it: it verifies that the sets really do commute, really are maximal, and really partition the Paulis.

Each basis is then the joint eigenbasis of its subspace, pinned down uniquely by weighting the $n$ generators by distinct powers of two before diagonalising, so that every sign pattern lands on its own eigenvalue.

### Where the terms come from

The measured label $j$ is an **outcome**, not a term of the decomposition. Its $2^n$ values share one channel's coefficient between them, because $\sum_j \mathrm{Tr}[\Pi_{ij}\rho] = 1$. `locc_wire_qpd` still emits one entry per $(\text{channel}, \text{label}, \text{prepared state})$, giving $2^n(2^{n+1}-1)$ entries, and gives each a coefficient of $1/2^n$ of its channel's. The estimator recovers the missing factor by weighting each label by how often it actually came up. That keeps the coefficient one-norm equal to $2^{n+1}-1$ and lets everything downstream treat these like any other terms.

| wires | channels | circuits | $\gamma$ | without communication |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 3 | 6 | 3 | 8 circuits, $\gamma = 4$ |
| 2 | 5 | 28 | 7 | 64 circuits, $\gamma = 16$ |
| 3 | 9 | 120 | 15 | 512 circuits, $\gamma = 64$ |
| 4 | 17 | 496 | 31 | 4096 circuits, $\gamma = 256$ |

## Running it in waves

A communicating cut forces its measuring side to run before its preparing side, so the subcircuits acquire a dependency, and those dependencies chain. Cutting a circuit into $A$, $B$ and $C$ so that $A$ feeds $B$ and $B$ feeds $C$ needs **three** waves rather than two, because $B$'s own measured outcome is what decides what $C$ prepares. The wave of a subcircuit is the longest chain reaching it. If wire cuts point both ways between the same two subcircuits the dependencies form a cycle, no order can satisfy them, and one direction keeps the non-communicating table.

Wave one holds everything no label can affect. Groups that differ only in which label they answer share those circuits, so they run once for the whole set rather than once per group. Each later wave splits its shots in proportion to how often the labels it depends on came up, which is the allocation that minimises the variance of the sum. A label that came up at all still gets at least one shot, or its contribution would go missing and bias the estimate.

The waves do not get equal shares of the budget. That sharing is exactly what makes a measuring shot worth more than a preparing one: it buys precision for every group answering that circuit at once, while a preparing shot buys it for one group. Dividing the budget evenly over the subcircuits, which is what a wave-agnostic rule amounts to, therefore over-funds the measuring wave — and by a factor of one over the number of pieces the circuit was cut into, so worst at two. Scans at one, two and three wires, across cuts into two and three pieces, all put the best measuring share near a sixth wherever they were run from, so `MEASURE_SHARE` is a constant rather than a formula. Redistributing to it is worth between a third and a factor of two and a half, and leaves the run's total untouched. How the rest divides between the later waves makes no measurable difference, so they divide it evenly.

Nothing here runs one shot at a time. Within a wave the circuits are sorted by how many shots they want and grouped into batches, each running at the mean of what its own circuits asked for. A batch closes on whichever comes first of `max_batch_size` circuits or a spread of more than a factor of two in what they asked for. The size limit keeps a job within what the backend will take; the spread limit stops a generous `max_batch_size` from collapsing a whole wave into one job at one shot count, which would be uniform allocation and would discard the proportional split. Uniform allocation is never better: by Cauchy-Schwarz its variance $G\sum_i p_i^2/T$ is at least the proportional $(\sum_i p_i)^2/T$, and the gap widens exactly as the label distribution skews.

Both limits together leave a wave costing a handful of jobs rather than one per group, which is what matters when a job's wall time is dominated by setting the device up rather than by its shots. Rounding the shares to a coarse grid instead would be worse, since a bucket overflowing the size limit is split into several jobs anyway and the coarseness bought nothing.

## What it actually buys

The circuit count improvement is real and large: 28 against 64 at two wires, 120 against 512 at three.

The shot cost is more subtle, and $\gamma$ overstates it. The protocol assumes the prepared state can follow the measured outcome **shot by shot**. Batched runs have to emulate that by post-selection, so a group's measuring side is read from only the shots whose label matched, rescaled by $2^n$. Its variance is then $2^n/N$ rather than $1/N$, and only the sharing of that circuit between the channel's $2^n$ groups pays for the difference. It pays for it exactly, leaving nothing over: the measuring side comes out at parity, not ahead.

That is why the realised gain falls short of $\gamma$. The $\gamma$ of a decomposition bounds the shot cost on the assumption that every term's per-shot estimator is capped at one. The local terms are products of two unconditioned Pauli expectations and sit far below that cap, while the rescaling above puts the communicating terms near it, so the two decompositions approach their own bounds by different margins and the ratio of the $\gamma$s overpredicts the ratio of the costs.

How far short is circuit dependent and this page does not put a number on it. Measurements taken by sampling the estimator's variance were too noisy to support one — repeats of a single configuration moved by factors approaching two — and settling it wants the variance evaluated in closed form from exact term values rather than sampled. What is not in doubt is that a single wire costs more in shots than the local tables while saving only a quarter of the circuits, and that blocks of two and more save on shots, circuits and jobs together.

That is why `wire_cut_communication` defaults to `"auto"`, which uses communication only for blocks of two or more wires. A single wire gains a quarter off its circuit count and pays for it in shots, which is not a trade worth making by default. See [Options](../Options.rst).

Removing the $2^n$ rescaling altogether would need genuine per-shot feed-forward between the two halves, mid-circuit measurement with the preparation conditioned on it, which is exactly the thing batched hardware is bad at.

## References

(locc-ref)=
1. L. Brenner, C. Piveteau and D. Sutter, *Optimal wire cutting with classical communication*, [arXiv:2302.03366](https://arxiv.org/abs/2302.03366). Proposition 4.1 for $\gamma_{\mathrm{LO}} = 4^n$ and Proposition 4.2 for $\gamma_{\mathrm{LOCC}} = 2^{n+1}-1$. Its own protocol is teleportation based and needs one ancilla per wire.

(locc-ref2)=
2. H. Harada, K. Wada and N. Yamamoto, *Doubly optimal parallel wire cutting without ancilla qubits*, PRX Quantum **5**, 040308 (2024), [arXiv:2303.07340](https://arxiv.org/abs/2303.07340). Theorem 3 is the decomposition used here, optimal in both the overhead and the number of channels, and free of ancillas.
