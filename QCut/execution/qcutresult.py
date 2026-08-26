"""Helper classes for storing results."""


# Class for storing results from single sub-circuit run
class SubResult:
    """Storage class for easier storage/access to the results of a subcircuit."""

    def __init__(self, measurements: list, count: int) -> None:
        """Init."""
        self.measurements = measurements  # measurement results
        self.count = count  # counts for this specific measurement

    def __str__(self) -> str:
        """Format string."""
        return f"{self.measurements}, {self.count}"

    def __repr__(self) -> str:
        """Represent as string."""
        return str(self)


# Store total results of all sub-circuits for a single experiment run
class TotalResult:
    """Storage class for easier access to the results of a subcircuit group."""

    def __init__(self, *subcircuits: list[list[SubResult]]) -> None:
        """Init."""
        self.subcircuits = subcircuits

    def __str__(self) -> str:
        """Format string."""
        substr = ""
        for i in self.subcircuits:
            substr += f"{i}"
        return substr

    def __repr__(self) -> str:
        """Represent as string."""
        return str(self)


class RawResult:
    """
    Simple wrapper class for raw experiment results. Stores the raw results, the shot
    count they were taken at, and the experiment data needed to interpret them, so that
    :func:`QCut.postprocess.estimate_expectation_values` can be called on the result
    alone rather than having the caller carry a second object around.
    """

    def __init__(
        self,
        results: list[list[dict[int, dict[str, int]]]],
        shots: int,
        expv_data: dict | None = None,
    ):
        self._shots = shots
        self._expv_data = expv_data
        self.results = results

    @property
    def expv_data(self) -> dict | None:
        """The experiment data these results came from, if it was recorded."""
        return self._expv_data

    def result(self) -> list[list[dict[int, dict[str, int]]]]:
        """
        Get raw results for all experiments.

        Currently the format is not great. Might change in the future.
        """
        return self.results
