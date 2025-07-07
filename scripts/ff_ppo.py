"""Script to train with agent with feedforward PPO"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import hydra
import random
import shutil

from colorama import Fore, Style
from omegaconf import OmegaConf, DictConfig
from external.stoix.systems.ppo import ff_ppo as algorithm


@hydra.main(
    config_path="../configs/main",
    config_name="main_ff_ppo",
    version_base=None,
)
def main(cfg: DictConfig) -> float:
    """Experiment entry point."""

    # print config
    # print(OmegaConf.to_yaml(cfg, resolve=True))

    # allow modifying attributes
    OmegaConf.set_struct(cfg, False)

    # set random seed if unset
    if cfg.run.seed is None:
        cfg.run.seed = random.randint(0, 2**15-1)

    # save resolved config (the original, unresolved config is automatically saved by hydra)
    filepath = os.path.join(cfg.run.output_dir, 'config.yaml')
    OmegaConf.save(config=cfg, f=filepath, resolve=True)

    # create trained_agent dir for potential Kaggle submission
    shutil.copytree("./player", cfg.run.trained_agent_dir)
    filepath = os.path.join(cfg.run.trained_agent_dir, 'config_agent.yaml')
    OmegaConf.save(config=cfg.env.agent, f=filepath, resolve=True)

    # run experiment
    eval_performance = algorithm.run_experiment(cfg)

    print(f"Experiment completed")
    print(f"{Fore.RED}{Style.BRIGHT}Final performance: {cfg.env.eval_metric} = {eval_performance}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()

