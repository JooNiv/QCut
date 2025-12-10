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