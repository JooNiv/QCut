# Reconstructing a Probability Distribution

**A cut experiment estimates expectation values, not counts. This page derives how a distribution over a chosen set of qubits is recovered from them, why it costs $2^k$ values but no extra circuits, and why the values it returns can come out negative. The transform is the inverse Walsh–Hadamard transform of [[1]](#prob-ref).**

## The problem

Running a circuit whole gives counts, a bitstring per shot, and a distribution by summing them. Cutting removes that. The circuit is split into subcircuits whose results are combined as a signed sum over quasiprobability terms, and what that sum estimates is the expectation value of an observable. No shot of the original circuit is taken anywhere in the experiment, so there is nothing to sum.

What can be recovered is the distribution over any chosen subset of qubits, because a distribution over $k$ bits is completely determined by the expectation values of the $2^k$ Pauli $Z$ strings on those bits.

## The transform

Write $x \in \{0,1\}^k$ for a bitstring and $S \subseteq \{1,\dots,k\}$ for a subset of the chosen qubits. Let

$$
Z_S = \bigotimes_{i \in S} Z_i
$$

be the Pauli $Z$ string supported on $S$, with $Z_\emptyset = I$. Its expectation value is a signed sum over the distribution,

$$
\langle Z_S \rangle = \sum_{x} (-1)^{x \cdot S} \, p(x),
\qquad x \cdot S = \sum_{i \in S} x_i \bmod 2 ,
$$

since $Z_S$ has eigenvalue $+1$ on bitstrings of even parity over $S$ and $-1$ on the odd ones.

The map $p \mapsto \langle Z_\bullet \rangle$ is the **Walsh–Hadamard transform** of $p$, written in the $\pm 1$ character basis $\chi_S(x) = (-1)^{x \cdot S}$ of the group $\mathbb{Z}_2^k$. Those characters are orthogonal,

$$
\sum_{x} \chi_S(x)\,\chi_T(x) = 2^k \, \delta_{S,T} ,
$$

so the transform is its own inverse up to the factor $2^k$, and the distribution comes back as

$$
p(x) = \frac{1}{2^k} \sum_{S} (-1)^{x \cdot S} \, \langle Z_S \rangle .
$$

That is the **inverse Walsh–Hadamard transform**, and it is the whole of the reconstruction. Every value QCut returns is one evaluation of that sum.

Nothing about this is specific to cutting. It is the Boolean Fourier expansion of [[1]](#prob-ref), and it works on top of any estimator that can return $\langle Z_S \rangle$.

## What it costs

The sum runs over all $2^k$ subsets, but $\langle Z_\emptyset \rangle = \langle I \rangle = 1$ is known without measuring anything. So $2^k - 1$ expectation values are estimated, and $2^k$ probabilities come out.

The cost is **not** paid in circuits. Every $Z_S$ commutes with every other, so they share one measurement setting. QCut folds them into a single observable group and the experiment is exactly the size it would have been for one observable. Measured on a gate cut, the circuit count is flat while the observable count grows:

| $k$ | observables | measurement settings | circuits |
| :-- | ----------: | -------------------: | -------: |
| 2   | 3           | 1                    | 12       |
| 4   | 15          | 1                    | 12       |
| 8   | 255         | 1                    | 12       |

Note that even though the number of circuits does not grow the cost of postprocessing grows exponentially with $k$.

## Why values come out negative

Nothing in the sum constrains $p(x) \geq 0$. Each $\langle Z_S \rangle$ is an independent estimate carrying its own error, and the inverse transform mixes $2^k$ of them with alternating signs, so a small true probability easily lands below zero. What comes back is a **quasiprobability distribution** that is real, summing to one, but not necessarily non-negative.

How negative the smallest value is measures how far the estimate is from any physical distribution. QCut therefore returns the quasiprobabilities as they come, and offers the projection onto the nearest true distribution.

(prob-ref)=
## References

1. R. O'Donnell, *Analysis of Boolean Functions*, Cambridge University Press (2014), [arXiv:2105.10386](https://arxiv.org/abs/2105.10386). Chapter 1 develops the Fourier expansion over $\mathbb{Z}_2^k$ in the character basis $\chi_S(x) = (-1)^{x \cdot S}$, of which the identity used here is the inverse transform.