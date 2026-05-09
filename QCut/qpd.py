"""Define identity channel quasi probability decomposition."""

from QCut.qpd_gates import (
    i_minus_init,
    i_plus_init,
    idmeas,
    minus_init,
    one_init,
    plus_init,
    s,
    sdg,
    sdg_meas,
    xmeas,
    ymeas,
    z,
    zero_init,
    zmeas,
)

# Note that here all the coefficients are +-1 instead of +-1/2 since for
# now all elements of the qpds have same magnitude coefficient and hence
# only their sign matters

identity_qpd = [
    {"op_0": idmeas, "op_1": zero_init, "c": 1},
    {"op_0": idmeas, "op_1": one_init, "c": 1},
    {"op_0": xmeas, "op_1": plus_init, "c": 1},
    {"op_0": xmeas, "op_1": minus_init, "c": -1},
    {"op_0": ymeas, "op_1": i_plus_init, "c": 1},
    {"op_0": ymeas, "op_1": i_minus_init, "c": -1},
    {"op_0": zmeas, "op_1": zero_init, "c": 1},
    {"op_0": zmeas, "op_1": one_init, "c": -1},
]

cz_qpd = [
    {"op_0": sdg, "op_1": sdg, "c": 1},
    {"op_0": s, "op_1": s, "c": 1},
    {"op_0": sdg_meas, "op_1": idmeas, "c": 1},
    {"op_0": sdg_meas, "op_1": z, "c": -1},
    {"op_0": idmeas, "op_1": sdg_meas, "c": 1},
    {"op_0": z, "op_1": sdg_meas, "c": -1},
]

# swap_qpd and iswap_qpd are not yet implemented.
# Derive from Mitarai & Fujii, PRA 2021 (doi:10.1103/PhysRevA.104.062421).
# SWAP has Weyl coords (π/4, π/4, π/4), y=7 -> 14 terms each with |c|=1/2.
# iSWAP has Weyl coords (π/4, π/4, 0),  y=5 -> 10 terms each with |c|=1/2.
# Use the same +-1 coefficient convention as cz_qpd (true coefficients are +-1/2).
swap_qpd: list[dict] = []   # TODO: fill in from paper
iswap_qpd: list[dict] = []  # TODO: fill in from paper
