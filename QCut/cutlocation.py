"""Storage class for storing cut locations."""

from __future__ import annotations

from qiskit import QuantumRegister


class CutLocation:
    """Storage class for storing cut locations."""

    def __init__(
        self, cut_location: tuple[tuple[tuple[QuantumRegister, int]], int]
    ) -> None:
        """Init."""
        self.qubits = cut_location[0]
        self.meas = cut_location[0][0][1]
        self.init = cut_location[0][1][1]
        self.index = cut_location[1]

    def __eq__(self, other: CutLocation) -> bool:
        """Equality."""
        if not isinstance(other, CutLocation):
            return NotImplemented

        return (
            self.meas == other.meas
            and self.init == other.init
            and self.index == other.index
        )

    def __str__(self) -> str:
        """Format string."""
        msg = (
            f"meas qubit: {self.meas}, init qubit: {self.init}, "
            f"cut index: {self.index}"
        )
        return msg

    def __repr__(self) -> str:
        """Represent as string."""
        return str(self)

class SingleQubitCutLocation:
    """Storage class for storing cut locations."""

    def __init__(self, cut_location: tuple[tuple[QuantumRegister, int], int]) -> None:
        """Init."""
        self.qubits = cut_location[0]
        self.meas = cut_location[0][1]
        self.init = cut_location[0][1]
        self.index = cut_location[1]

    def __eq__(self, other: SingleQubitCutLocation) -> bool:
        """Equality."""
        if not isinstance(other, SingleQubitCutLocation):
            return NotImplemented

        return (
            self.meas == other.meas
            and self.init == other.init
            and self.index == other.index
        )

    def __str__(self) -> str:
        """Format string."""
        msg = (
            f"meas qubit: {self.meas}, init qubit: {self.init}, "
            f"cut index: {self.index}"
        )
        return msg

    def __repr__(self) -> str:
        """Represent as string."""
        return str(self)
