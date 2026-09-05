"""Tools package."""


class ToolError(Exception):
    """Raised by agent tools when an operation fails.

    Used for clean exception typing so the supervisor and worker subgraphs
    can distinguish tool failures from other exceptions (e.g. auth errors).
    """
    pass
