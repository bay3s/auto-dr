from dataclasses import dataclass

import numpy as np

from auto_dr.randomization.randomization_bound import RandomizationBound


@dataclass
class RandomizationParameter:
    """
    Dataclass describing a randomization associated with a specific environment.
    """

    name: str
    lower_bound: RandomizationBound
    upper_bound: RandomizationBound
    delta: float

    def __post_init__(self):
        """
        Post-init for randomization parameters, adds validation for the values provided.

        Returns:
          None
        """
        assert self.lower_bound.value <= self.upper_bound.value
        assert self.lower_bound.max_value <= self.upper_bound.min_value

    @property
    def range(self) -> float:
        """
        Return the range of the parameter.

        Returns:
          float
        """
        return self.upper_bound.value - self.lower_bound.value

    def sample(self) -> float:
        """
        Sample a value for the randomized parameter.

        Returns:
          float
        """
        return np.random.uniform(self.lower_bound.value, self.upper_bound.value)

    def increase_upper_bound(self) -> None:
        """
        Increase the current upper bound for the randomized parameter.

        Returns:
          None
        """
        self.upper_bound.increase(self.delta)

    def decrease_upper_bound(self) -> None:
        """
        Decrease the current upper bound for the randomized parameter.

        Returns:
          None
        """
        self.upper_bound.decrease(self.delta)

    def decrease_lower_bound(self) -> None:
        """
        Decrease the lower bound by the delta value.

        Returns:
          None
        """
        self.lower_bound.decrease(self.delta)

    def increase_lower_bound(self) -> None:
        """
        Increase the lower bound by the delta value.

        Returns:
          None
        """
        self.lower_bound.increase(self.delta)
