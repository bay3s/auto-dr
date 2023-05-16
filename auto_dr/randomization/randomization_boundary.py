from dataclasses import dataclass

from auto_dr.randomization.randomization_bound import RandomizationBound
from auto_dr.randomization.randomization_parameter import RandomizationParameter


@dataclass
class RandomizationBoundary:
    """
    Describes the boundary sampled during Auto DR by the parameter and bound .
    """

    parameter: RandomizationParameter
    bound: RandomizationBound
