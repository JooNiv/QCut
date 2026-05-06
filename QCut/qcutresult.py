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


# Store total results of all sub-circuits (two for now)
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
    Simple wrapper class for raw experiment results. Stores the raw results and provides
    a method to return the results in the expected format for post-processing and also
    stores the number of samples and shots used for the experiment which are used for
    postprocessing.
    """

    def __init__(
        self,
        results: list[list[dict[int, dict[str, int]]]],
        samples: int,
        shots: int,
    ):
        self._samples = samples
        self._shots = shots
        self.results = results

    def result(self) -> list[list[dict[int, dict[str, int]]]]:
        """
        Get raw results for all experiments.

        Currently the format is not great. Might change in the future.
        """
        return self.results
