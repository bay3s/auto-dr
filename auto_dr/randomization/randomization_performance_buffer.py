from collections import deque
from typing import List

from auto_dr.randomization.randomization_boundary import RandomizationBoundary
from auto_dr.randomization.randomization_parameter import RandomizationParameter


class RandomizationPerformanceBuffer:
    def __init__(
        self, randomized_parameters: List[RandomizationParameter], buffer_size: int
    ):
        """
        Buffer to keep track of returns associacted with boundary sampling.

        Args:
          randomized_parameters (List[RandomizationParameter]): List of randomized paramters for Auto-DR.
          buffer_size (int): Buffer size required - this should be the same as the number of returns required by Auto DR to
            evaluate performance for a specific attempt at boundary sampling.
        """
        self._randomized_parameters = randomized_parameters
        self._buffer_size = buffer_size
        self._buffer = self._init_buffer(randomized_parameters, buffer_size)
        pass

    @staticmethod
    def _init_buffer(
        randomization_parameters: List[RandomizationParameter], buffer_size: int
    ) -> dict:
        """
        Initialize the buffer required based on the list of randomized parameters and their respective bounds.

        Args:
          randomization_parameters (List[RandomizationParameter]): List of randomized parameters.
          buffer_size (int): Buffer size to instantiate.

        Returns:
          dict
        """
        buffer = dict()

        for param in randomization_parameters:
            buffer[param.name] = dict()

            lower_bound = param.lower_bound
            upper_bound = param.upper_bound

            buffer[param.name][lower_bound.type.value] = deque(maxlen=buffer_size)
            buffer[param.name][upper_bound.type.value] = deque(maxlen=buffer_size)

        return buffer

    def is_full(self, randomization_boundary: RandomizationBoundary) -> bool:
        """
        Returns true if the buffer for a specific sampled boundary is full.

        Args:
          randomization_boundary (RandomizationBoundary): Instance specifying which boundary was sampled.

        Returns:
          bool
        """
        param = randomization_boundary.parameter
        bound = randomization_boundary.bound

        return len(self._buffer[param.name][bound.type.value]) >= self._buffer_size

    def insert(
        self, randomization_boundary: RandomizationBoundary, episode_return: float
    ) -> None:
        """
        Update the buffer for a specific sampled boundary with return from an associated episode.

        Args:
          randomization_boundary (RandomizationBoundary): Sampled boundary.
          episode_return (float): Episode return associated with the sampled boundary.

        Returns:
          None
        """
        param = randomization_boundary.parameter
        bound = randomization_boundary.bound

        self._buffer[param.name][bound.type.value].append(episode_return)
        pass

    def truncate(self, randomization_boundary: RandomizationBoundary) -> None:
        """
        Truncate the buffer associated with a sampled boundary.

        Args:
          randomization_boundary (RandomizationBoundary): Parameter boundary for which to truncate the buffer.

        Returns:
          None
        """
        param = randomization_boundary.parameter
        bound = randomization_boundary.bound

        self._buffer[param.name][bound.type.value] = deque(maxlen=self._buffer_size)
        pass

    def truncate_all(self) -> None:
        """
        Truncate all buffers.

        Returns:
          None
        """
        self._buffer = self._init_buffer(self._randomized_parameters, self._buffer_size)

    def get(self, randomization_boundary: RandomizationBoundary) -> list:
        """
        Get episode returns associated with a specific parameter boundary.

        Args:
          randomization_boundary (RandomizationBoundary): Parameter boundary for which to retrieve episode returns.

        Returns:
          list
        """
        param = randomization_boundary.parameter
        bound = randomization_boundary.bound

        return self._buffer[param.name][bound.type.value]
