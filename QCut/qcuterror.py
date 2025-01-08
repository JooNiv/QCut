"Define QCutError class for custom error handling."

from typing import Optional


class QCutError(Exception):
    """Exception raised for custom error conditions.

    Attributes:
        message (str): Explanation of the error.
        code (int, optional): Error code representing the error type.

    """

    def __init__(
        self, message: str = "An error occurred", code: Optional[int] = None
    ) -> None:
        """Init.

        Args:
            message (str): Explanation of the error. Default is "An error occurred".
            code (int): Optional error code representing the error type.

        """
        self.message = message
        self.code = code
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return the string representation of the error.

        Returns
            str: A string describing the error, including the code if available.

        """
        if self.code:
            return f"[Error {self.code}] {self.message}"

        return self.message