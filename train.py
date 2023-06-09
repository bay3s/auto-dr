import os

import argparse

from auto_dr.training.trainer import ExperimentConfig
from auto_dr.training.trainer import Trainer
from auto_dr.utils.env_utils import register_custom_envs

register_custom_envs()


SUPPORTED_ENVIRONMENTS = [
    "point_robot_navigation",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Automatic Domain Randomization",
        description="Script to run experiments on domain randomization benchmarks.",
    )

    parser.add_argument(
        "--run-all",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to run all environments, if this is set then the environment parameter is ignored.",
    )

    parser.add_argument(
        "--env-name",
        choices=SUPPORTED_ENVIRONMENTS,
        default=None,
        help=f"Environment to run the experiment in, one of [{', '.join(SUPPORTED_ENVIRONMENTS)}].",
    )

    parser.add_argument(
        "--from-checkpoint",
        type=str,
        default=None,
        help="Checkpoint, if any, from which to restart the training run.",
    )

    parser.add_argument(
        "--disable-wandb",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help=f"Whether to log the experiment to `wandb`.",
    )

    args = parser.parse_args()

    if args.env_name is None and not args.run_all:
        raise ValueError(
            f"Unable to infer experiment environment from the inputs, either provide `--env-name` or "
            f"set `--run-all` to `True`"
        )

    environments = [args.env_name] if not args.run_all else SUPPORTED_ENVIRONMENTS

    for env_name in environments:
        # load config
        config_path = f"{os.path.dirname(__file__)}/configs/{env_name}.json"
        experiment_config = ExperimentConfig.from_json(config_path)

        # train
        trainer = Trainer(experiment_config, checkpoint_path=args.from_checkpoint)
        trainer.train(enable_wandb=not args.disable_wandb)
        pass
