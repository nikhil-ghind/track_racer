import argparse
from agent.evaluate import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="./models/ppo_racing")
    parser.add_argument("--n-episodes", type=int, default=5)
    args = parser.parse_args()
    results = evaluate(args.model_path, n_episodes=args.n_episodes, render=True)
    print(f"Mean reward:  {results['mean_reward']:.2f}")
    print(f"Mean laps:    {results['mean_laps']:.2f}")
    print(f"Mean ep len:  {results['mean_episode_length']:.0f}")
