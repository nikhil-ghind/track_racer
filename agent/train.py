import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from env.racing_env import RacingEnv
from agent.callbacks import LapCounterCallback


def train(env_config_path: str = "configs/env_config.yaml", ppo_config_path: str = "configs/ppo_config.yaml"):
    with open(ppo_config_path) as f:
        cfg = yaml.safe_load(f)

    env = make_vec_env(lambda: RacingEnv(config_path=env_config_path), n_envs=1)

    model = PPO(
        cfg["policy"],
        env,
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
        ent_coef=cfg["ent_coef"],
        vf_coef=cfg["vf_coef"],
        learning_rate=cfg["learning_rate"],
        tensorboard_log=cfg["tensorboard_log"],
        verbose=1,
    )

    callbacks = [
        LapCounterCallback(),
        CheckpointCallback(save_freq=50_000, save_path="./models/", name_prefix="ppo_racing"),
    ]

    model.learn(total_timesteps=cfg["total_timesteps"], callback=callbacks, progress_bar=True)
    model.save(cfg["model_save_path"])
    print(f"Model saved to {cfg['model_save_path']}")
    env.close()
