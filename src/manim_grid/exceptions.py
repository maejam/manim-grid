class GridError(RuntimeError):
    """Base class for all grid-related errors."""


class GridKeyError(GridError, KeyError):
    """Raised when a supplied key cannot be interpreted for a given dimension."""


class GridValueError(GridError, ValueError):
    """Raised when a supplied value is of the wrong type or shape."""


class GridLabelError(GridKeyError):
    """Raised when a string label is not present in the label mapping."""


class GridShapeError(GridKeyError):
    """Raised when a Grid does not have the right shape."""
