from stable_baselines3 import PPO
from env.racing_env import RacingEnv


def evaluate(model_path: str, env_config_path: str = "configs/env_config.yaml",
             n_episodes: int = 10, render: bool = False) -> dict:
    model = PPO.load(model_path)
    render_mode = "human" if render else None
    env = RacingEnv(config_path=env_config_path, render_mode=render_mode)

    total_rewards, total_laps, total_lengths = [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward, ep_laps, ep_len = 0.0, 0, 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_laps = max(ep_laps, info.get("lap_count", 0))
            ep_len += 1
            done = terminated or truncated
        total_rewards.append(ep_reward)
        total_laps.append(ep_laps)
        total_lengths.append(ep_len)

    env.close()
    return {
        "mean_reward": sum(total_rewards) / n_episodes,
        "mean_laps": sum(total_laps) / n_episodes,
        "mean_episode_length": sum(total_lengths) / n_episodes,
    }
