from typing import List, Tuple

import random

import gym
import numpy as np

from auto_dr.randomization.randomization_performance_buffer import (
    RandomizationPerformanceBuffer,
)
from auto_dr.randomization.randomization_parameter import RandomizationParameter
from auto_dr.randomization.randomization_bound_type import RandomizationBoundType
from auto_dr.randomization.randomization_boundary import RandomizationBoundary
from auto_dr.envs.pytorch_vec_env_wrapper import PyTorchVecEnvWrapper


class Randomizer:
    def __init__(
        self,
        parallel_envs: PyTorchVecEnvWrapper,
        evaluation_probability: float,
        buffer_size: int,
        delta: float,
        performance_threshold_lower: float,
        performance_threshold_upper: float,
    ) -> None:
        """
        Automatic Domain Randomization (ADR) based on the Open AI paper.

        Args:
            parallel_envs (int): Number of environments being randomized in parallel.
            evaluation_probability (float): Probability of boundary sampling and subsequently increasing the difficulty.
            buffer_size (int): Minimum buffer size required for evaluating boundary sampling performance.
            delta (float): Delta parameter by which to increment or decrement parameter bounds.
            performance_threshold_upper (float): Lower threshold for performance on a specific environment, if this is
                not met then the parameter entropy is decreased.
            performance_threshold_lower (float): Lower threshold for performance on a specific environment, if this is
                met then the parameter entropy is increased.
        """
        self.parallel_envs = parallel_envs

        randomizable_params = self.parallel_envs.env_method(
            "randomizable_parameters", indices=0
        )[0]
        self.randomized_parameters = self._init_params(randomizable_params)
        self.buffer = RandomizationPerformanceBuffer(
            randomizable_params, buffer_size=buffer_size
        )

        self.evaluation_probability = evaluation_probability
        self.buffer_size = buffer_size
        self.delta = delta

        self.sampled_boundaries = [None] * parallel_envs.num_envs

        # performance
        self.lower_performance_threshold = performance_threshold_lower
        self.upper_performance_threshold = performance_threshold_upper

        # logging
        self.last_performances = []
        self.last_increments = []
        self.current_bound = 0
        pass

    @staticmethod
    def _init_params(params: List[RandomizationParameter]) -> dict:
        """
        Convert a list of parameters to dict.

        Args:
            params (List[RandomizationParameter]): A list of randomized parameters.

        Returns:
            dict
        """
        randomized = dict()

        for param in params:
            randomized[param.name] = param

        return randomized

    def entropy(self, eps: float = 1e-12) -> float:
        """
        Evaluate the current entropy of the parameters.

        Args:
          eps (float): Stub to avoid numerical issues.

        Returns:
          float
        """
        ranges = list()

        for param in self.randomized_parameters.values():
            ranges.append(param.range + eps)

        return np.log(ranges).mean()

    def re_evaluate(self, sampled_boundary: RandomizationBoundary) -> None:
        """
        Update ADR bounds based on the performance for a given boundary.

        Args:
          sampled_boundary (RandomizationBoundary): Sampled boundary to evaluate.

        Returns:
          None
        """
        if not self.buffer.is_full(sampled_boundary):
            return

        performance = np.mean(np.array(self.buffer.get(sampled_boundary)))
        self.buffer.truncate(sampled_boundary)

        param = sampled_boundary.parameter
        bound = sampled_boundary.bound

        # increase entropy
        if performance >= self.upper_performance_threshold:
            if bound.type == RandomizationBoundType.UPPER_BOUND:
                self.randomized_parameters[param.name].increase_upper_bound()
            elif bound.type == RandomizationBoundType.LOWER_BOUND:
                self.randomized_parameters[param.name].decrease_lower_bound()
            else:
                raise ValueError

        # decrease entropy
        if performance < self.lower_performance_threshold:
            if bound.type == RandomizationBoundType.UPPER_BOUND:
                self.randomized_parameters[param.name].decrease_upper_bound()
            elif bound.type == RandomizationBoundType.LOWER_BOUND:
                self.randomized_parameters[param.name].increase_lower_bound()
            else:
                raise ValueError

    def _get_task(self) -> Tuple:
        """
        Get randomized parameter values.

        Returns:
          Tuple
        """
        randomized_params = dict()

        # boundary
        sampled_boundary = None

        for param in self.randomized_parameters.values():
            lower_bound = param.lower_bound
            upper_bound = param.upper_bound

            randomized_params[param.name] = np.random.uniform(
                lower_bound.value, upper_bound.value
            )

        # adr
        if np.random.uniform(0, 1) <= self.evaluation_probability:
            sampled_param = random.choice(list(self.randomized_parameters.values()))
            sampled_bound = random.choice(
                list([sampled_param.lower_bound, sampled_param.upper_bound])
            )

            # boundary sampling
            if sampled_bound.type == RandomizationBoundType.UPPER_BOUND:
                randomized_params[sampled_param.name] = sampled_bound.value
            elif sampled_bound.type == RandomizationBoundType.LOWER_BOUND:
                randomized_params[sampled_param.name] = sampled_bound.value
            else:
                raise ValueError

            sampled_boundary = RandomizationBoundary(
                parameter=sampled_param, bound=sampled_bound
            )
            pass

        return randomized_params, sampled_boundary

    def randomize_all(self) -> None:
        """
        Sample tasks for each environment.

        Returns:
          None
        """
        zipped = zip(range(self.parallel_envs.num_envs), self.sampled_boundaries)
        new_tasks = list()

        for env_idx, boundary in zipped:
            randomized_params, boundary = self._get_task()
            self.sampled_boundaries[env_idx] = boundary
            new_tasks.append(randomized_params)

        self.parallel_envs.update_tasks_async(np.array(new_tasks))
        pass

    def update_buffer(
        self, sampled_boundary: RandomizationBoundary, episode_return: float
    ) -> None:
        """
        Update buffer with the sampled boundary and associated episode return.

        Args:
          sampled_boundary (RandomizationBoundary): Parameter boundary sampled for Auto DR.
          episode_return (float): Episode return for the sampled boundary.

        Returns:
          None
        """
        self.buffer.insert(sampled_boundary, episode_return)

    def start_meta_episode(self):
        """
        Resets the difficulty adjusted flags to `False` at the beginning of the meta-episode.

        Returns:
            None
        """
        self.difficulty_updated = [False] * self.parallel_envs.num_envs

    def _on_step(self, dones: List, infos: List) -> None:
        """
        Randomizer logic to be executed after each environment step.

        - Update the performance buffer with the episode returns.
        - Update randomization bounds and entropy.
        - Propagate updates to the environment.

        Args:
          dones (List): List of boolean values indicating whether an episode is done.
          infos (List): Info related to the current environment step.

        Returns:
          None
        """
        zipped = zip(
            dones, infos, range(self.parallel_envs.num_envs), self.sampled_boundaries
        )

        for done, info, env_idx, boundary in zipped:
            if not done and not self.difficulty_updated[env_idx]:
                continue

            if boundary is not None:
                self.update_buffer(boundary, info["episode"]["r"])
                self.re_evaluate(boundary)

            # sample
            randomized_params, boundary = self._get_task()
            self.sampled_boundaries[env_idx] = boundary
            self.parallel_envs.env_method(
                "update_task", randomized_params, indices=env_idx
            )

    @property
    def observation_space(self) -> gym.Space:
        """
        Return the observation space for the environment.

        Returns:
            gym.Space
        """
        return self.parallel_envs.observation_space

    @property
    def action_space(self) -> gym.Space:
        """
        Return the action space for the environment.

        Returns:
            gym.Space
        """
        return self.parallel_envs.action_space

    @property
    def num_envs(self) -> int:
        """
        Return the number of parallel environments that the Randomizer is controlling.

        Returns:
            int
        """
        return self.parallel_envs.num_envs

    def step(self, actions: np.ndarray) -> Tuple:
        """
        Take a step in the environments and return a tuple of observations, rewards, dones, infos.

        Args:
            actions (np.ndarray): Actions to take in the randomized environments.

        Returns:
            Tuple
        """
        obs, rewards, dones, infos = self.parallel_envs.step(actions)
        self._on_step(dones, infos)

        return obs, rewards, dones, infos

    @property
    def info(self) -> dict:
        """
        Returns info regarding a specific randomizer instance.

        Returns:
            dict
        """
        info = dict()

        for param in self.randomized_parameters.values():
            # boundaries
            lower_boundary = RandomizationBoundary(param, param.lower_bound)
            upper_boundary = RandomizationBoundary(param, param.upper_bound)

            # buffers
            lower_buffer = self.buffer.get(lower_boundary)
            upper_buffer = self.buffer.get(upper_boundary)

            # upper
            info[f"randomizer/{param.name}_upper"] = param.upper_bound.value
            info[f"randomizer/{param.name}_upper_buffer_size"] = len(upper_buffer)

            # lower
            info[f"randomizer/{param.name}_lower"] = param.lower_bound.value
            info[f"randomizer/{param.name}_lower_buffer_size"] = len(lower_buffer)

            # range
            info[f"randomizer/{param.name}_range"] = param.range
            continue

        info["randomizer/entropy"] = self.entropy()

        return info
