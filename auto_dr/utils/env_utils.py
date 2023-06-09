import os
from typing import Union, Callable

import torch
import gym

from stable_baselines3.common.monitor import Monitor
from gym.envs.registration import register

from auto_dr.envs.normalized_vec_env import NormalizedVecEnv as NormalizedVecEnv
from auto_dr.envs.multiprocessing_vec_env import MultiprocessingVecEnv
from auto_dr.envs.pytorch_vec_env_wrapper import PyTorchVecEnvWrapper
from auto_dr.envs.rl_squared_env import RLSquaredEnv


def get_render_func(venv: gym.Env):
    """
    Get render function for the environment.

    Args:
        venv (object): Environment in which

    Returns:
        Callable
    """
    if hasattr(venv, "envs"):
        return venv.envs[0].render
    elif hasattr(venv, "venv"):
        return get_render_func(venv.venv)
    elif hasattr(venv, "env"):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv: gym.Env) -> Union[NormalizedVecEnv, None]:
    """
    Given an environment, wraps it in a normalized environment wrapper.

    Args:
        venv (gym.Env): Gym environment to normalize.

    Returns:
        gym.Env
    """
    if isinstance(venv, NormalizedVecEnv):
        return venv

    elif hasattr(venv, "venv"):
        return get_vec_normalize(venv.venv)

    return None


def make_env_thunk(
    env_name: str,
    env_configs: dict,
    seed: int,
    rank: int,
    log_dir: str,
    allow_early_resets: bool,
) -> Callable:
    """
    Returns a callable to create environments based on the specs provided.

    Args:
        env_name (str): Environment to create.
        env_configs (dict): Key word arguments for making the environment.
        seed (int): Random seed for the experiments.
        rank (int): "Rank" of the environment that the callable would return.
        log_dir (str): Directory for logging.
        allow_early_resets (bool): Allows resetting the environment before it is done.

    Returns:
        Callable
    """

    def _thunk():
        env = gym.make(env_name, **env_configs)
        env.seed(seed + rank)

        env = RLSquaredEnv(env)

        env = Monitor(
            env, os.path.join(log_dir, str(rank)), allow_early_resets=allow_early_resets
        )

        # @todo requires convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            raise NotImplementedError

        return env

    return _thunk


def make_vec_envs(
    env_name: str,
    env_kwargs: dict,
    seed: int,
    num_processes: int,
    gamma: float,
    log_dir: str,
    device: torch.device,
    allow_early_resets: bool,
) -> PyTorchVecEnvWrapper:
    """
    Returns PyTorch compatible vectorized environments.

    Args:
        env_name (str): Name of the environment to be created.
        env_kwargs (dict): Key word arguments to create the environment.
        seed (int): Random seed for environments.
        num_processes (int): Number of parallel processes to be used for simulations.
        gamma (float): Discount factor for computing returns.
        log_dir (str): Directory for logging.
        device (torch.device): Device to use with PyTorch tensors.
        allow_early_resets (bool): Allows resetting the environment before it is done.

    Returns:
        PyTorchVecEnvWrapper
    """
    envs = [
        make_env_thunk(env_name, env_kwargs, seed, i, log_dir, allow_early_resets)
        for i in range(num_processes)
    ]

    envs = MultiprocessingVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        # @todo normalize when necessary
        pass
    else:
        raise NotImplementedError

    envs = PyTorchVecEnvWrapper(envs, device)

    return envs


def register_custom_envs() -> None:
    """
    Register custom environments for experiments.

    Returns:
        None
    """
    register(
        id="PointRobotNavigation-v1",
        entry_point="auto_dr.envs.point_robot.navigation_env:NavigationEnv",
    )
