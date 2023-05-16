from enum import Enum


class RandomizationBoundType(Enum):
    """
    Enum describing the type of bound.
    """

    UPPER_BOUND = "upper_bound"
    LOWER_BOUND = "lower_bound"
    pass
