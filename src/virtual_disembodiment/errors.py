class ContractError(ValueError):
    """Raised when a recording, configuration, or operation violates its contract."""


class FutureLeakageError(ContractError):
    """Raised when data after an observation cutoff is supplied as history."""


class FormalRunBlocked(RuntimeError):
    """Raised when a formal run is attempted with unresolved proposal inputs."""

