from dataclasses import dataclass

import numpy as np

from auto_dr.randomization.randomization_bound_type import RandomizationBoundType


@dataclass
class RandomizationBound:
    """
    Dataclass to keep track of information regarding randomization bounds.
    """

    type: RandomizationBoundType
    value: float
    min_value: float
    max_value: float

    def __post_init__(self):
        """
        Post-init for randomization bounds, adds validation for the values provided.

        Returns:
          None
        """
        assert self.min_value <= self.value <= self.max_value
        pass

    def increase(self, delta: float) -> None:
        """
        Increase the value of the bound by the delta provided.

        Args:
          delta (float): Amount that the bound should be increased by.

        Returns:
          None
        """
        if not np.isclose(self.max_value, self.value):
            self.value = min(self.value + delta, self.max_value)

    def decrease(self, delta: float) -> None:
        """
        Decrease the value of the bound by the delta provided.

        Args:
          delta (float): Amount that the bound should be decreased by.

        Returns:
          None
        """
        if not np.isclose(self.min_value, self.value):
            self.value = max(self.value - delta, self.min_value)
