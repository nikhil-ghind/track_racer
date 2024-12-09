import argparse
from agent.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/env_config.yaml")
    parser.add_argument("--ppo-config", default="configs/ppo_config.yaml")
    args = parser.parse_args()
    train(args.env_config, args.ppo_config)
