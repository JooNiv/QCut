"""Define identity channel quasi probability decomposition."""  # noqa: N999

from .qpd_gates import (
    i_minus_init,
    i_plus_init,
    idmeas,
    minus_init,
    one_init,
    plus_init,
    xmeas,
    ymeas,
    zero_init,
    zmeas,
)

identity_qpd = [
    {"op": idmeas, "init":zero_init, "c": 1},
    {"op": idmeas, "init":one_init, "c": 1},
    {"op": xmeas, "init":plus_init, "c": 1},
    {"op": xmeas, "init":minus_init, "c": -1},
    {"op": ymeas, "init":i_plus_init, "c": 1},
    {"op": ymeas, "init":i_minus_init, "c": -1},
    {"op": zmeas, "init":zero_init, "c": 1},
    {"op": zmeas, "init":one_init, "c": -1},
]
