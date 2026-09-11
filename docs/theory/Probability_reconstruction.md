# Reconstructing a Probability Distribution

**A cut experiment estimates expectation values, not counts. This page derives how a distribution over a chosen set of qubits is recovered anyway, why it costs no extra circuits and nothing of size $2^k$ to compute, and why the values it returns can come out negative.**

## The problem

Running a circuit whole gives counts, a bitstring per shot, and a distribution by summing them. Cutting removes that. The circuit is split into subcircuits whose results are combined as a signed sum over quasiprobability terms, and what that sum estimates is the expectation value of an observable. No shot of the original circuit is taken anywhere in the experiment, so there is nothing to sum.

What can be recovered is the distribution over any chosen subset of qubits, and it comes straight out of the decomposition.

## The reconstruction

A quasiprobability decomposition [[1]](#prob-ref) [[2]](#prob-ref) writes the cut circuit's state as a signed combination of product states,

$$
\rho = \sum_g c_g \, \bigotimes_i \rho_i^g ,
$$

Each chosen qubit lives in exactly one subcircuit, so measuring them all is a product measurement, and the distribution over them is the same signed combination of the subcircuits' own distributions:

$$
p(x) = \sum_g c_g \prod_i p_i^g(x_i) .
$$

That is the whole reconstruction. The same step, for a Mitarai–Fujii gate cut, is Theorem 1 of [[3]](#prob-ref), which also bounds the sampling error on every one of the $2^n$ values.

Two conventions turn it into what QCut computes. A wire cut's mid-circuit measurement contributes a sign, so the per-subcircuit factor is a **weight** $w_i$ rather than a distribution, it can be negative and it does not sum to one. And the signs the estimator carries collect into a complement of the outcome plus a constant:

$$
T[y] = \sum_g (-1)^{w+1} c_g \prod_i w_i(y_i),
\qquad
p(x) = \frac{1 + \sum_y T[y]}{2^k} - T[\bar{x}]
$$

where $w$ counts the wire cuts, $y_i$ is $y$ restricted to the bits subcircuit $i$ holds, and $\bar{x}$ is the bitwise complement of $x$. Nothing here is of size $2^k$: the normalising sum is separable too, being $\sum_g (-1)^{w+1} c_g \prod_i \sum_{y_i} w_i(y_i)$.

## What it costs

Not circuits. Every chosen qubit is measured in the computational basis, so the whole distribution needs one measurement setting (all qubits of interested measured in computational basis) and the experiment is exactly the size it would have been for a single observable, however many qubits are asked for. Measured on a gate cut:

| qubits in the distribution | measurement settings | circuits |
| :-- | -------------------: | -------: |
| 2   | 1                    | 12       |
| 4   | 1                    | 12       |
| 8   | 1                    | 12       |

Not postprocessing either. Each subcircuit is read over its own $2^{k_i}$ outcomes rather than over $2^k$, and $\sum_i k_i = k$, so the intermediate is exponentially smaller than the answer would be. And the answer need never be expanded:

- **one bitstring** costs a pass over the groups, since $T[\bar{x}]$ is a product of one lookup per subcircuit;
- **a marginal** over fewer qubits is the same form with each $w_i$ summed over the bits dropped, so it stays separable;
- **the most likely bitstrings** are found by branch and bound, because $p(x)$ decreases with $T[\bar{x}]$ and $T$ is a sum of separable terms with one mode per subcircuit.

Only asking for every value at once costs $2^k$.

## Why values come out negative

Nothing in the sum constrains $p(x) \geq 0$. The weights $w_i$ carry the signs of the qpd measurements, so they are not probabilities, and the coefficients $c_g$ alternate in sign leading to small true probability easily landing below zero. What comes back is a **quasiprobability distribution** that is real, summing to one, but not necessarily non-negative.

How negative the smallest value is measures how far the estimate is from any physical distribution. QCut therefore returns the quasiprobabilities as they come, and offers the projection onto the nearest true distribution.

One consequence is worth stating plainly. These are **quasiprobabilities, computed, not samples drawn**. A signed distribution cannot be sampled from, so cutting does not give back the ability to draw shots from the uncut circuit but gives an estimate of each value.

(prob-ref)=
## References

1. K. Mitarai and K. Fujii, *Constructing a virtual two-qubit gate by sampling single-qubit operations*, New J. Phys. **23**, 023021 (2021), [arXiv:1909.07534](https://arxiv.org/abs/1909.07534). Quasiprobability decomposition of a non-local gate into local operations.
2. C. Piveteau and D. Sutter, *Circuit knitting with classical communication*, IEEE Trans. Inf. Theory **70**, 2734 (2024), [arXiv:2205.00016](https://arxiv.org/abs/2205.00016). Quasiprobability cutting, and what classical communication between the sides buys.
3. N. M. P. Neumann, C. M. R. Rocha, J. Verbree and M. van Vliet, *Probability distribution reconstruction using circuit cutting applied to a variational classifier*, [arXiv:2510.03077](https://arxiv.org/abs/2510.03077) (2025). Theorem 1 reconstructs the distribution from a Mitarai–Fujii gate cut as a weighted sum of the subcircuits' empirical distributions.
