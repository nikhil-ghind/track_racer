# Track Racer

PPO reinforcement learning agent trained to autonomously drive a racing track using Stable-Baselines3 and a custom Gymnasium environment.

## Overview

- Custom `RacingEnv` (Gymnasium-compliant) with circular track, 9 ray sensors, bicycle-model car physics
- Shaped reward: progress along waypoints, speed bonus, heading alignment, off-track penalty, lap completion bonus
- PPO agent (MlpPolicy) with TensorBoard logging and checkpoint callbacks
- Pygame real-time renderer showing track boundaries, car, ray sensors, and HUD

## Tech Stack

Python 3.11 · Stable-Baselines3 · Gymnasium · PyTorch · Pygame · TensorBoard

## Quickstart

```bash
pip install -r requirements.txt

# Train
python scripts/train.py --env-config configs/env_config.yaml --ppo-config configs/ppo_config.yaml

# Watch a trained agent
python scripts/watch_agent.py --model-path ./models/ppo_racing --n-episodes 5

# Monitor training
tensorboard --logdir ./tensorboard_logs

# Tests
pytest tests/ -v
```

## Architecture

```mermaid
flowchart TB
    subgraph env["RacingEnv — Gymnasium"]
        track["Track<br/>50 waypoints on a circle,<br/>inner radius 200 / outer 280,<br/>get_progress, is_on_track"]
        car["Car<br/>bicycle-style update:<br/>steering_rate, acceleration,<br/>brake, friction, max_speed 10"]
        rays["cast_rays<br/>9 ray sensors, length 200"]
        obs["observation, shape 11<br/>dist_to_center / half width,<br/>heading_error / 180,<br/>9 ray distances"]
        rew["compute_reward<br/>progress x 10<br/>speed x 0.1 when on track<br/>heading alignment up to 0.5<br/>off track -10<br/>lap +100<br/>-0.01 per step"]
        done["terminated: off track<br/>truncated: 2000 steps"]
    end

    subgraph agent["PPO — Stable-Baselines3"]
        pol["MlpPolicy actor-critic"]
        roll["rollout buffer<br/>n_steps 2048, gamma 0.99,<br/>gae_lambda 0.95"]
        upd["update<br/>10 epochs, batch 64, clip 0.2,<br/>ent_coef 0.01, vf_coef 0.5, lr 3e-4"]
    end

    act["action, shape 2<br/>steering and throttle in [-1, 1]"]
    cb["LapCounterCallback<br/>logs lap_count from info"]
    tb["TensorBoard<br/>./tensorboard_logs"]
    save["models/ppo_racing"]
    watch["scripts/watch_agent.py<br/>Pygame renderer: boundaries,<br/>car, rays and HUD"]
    ev["agent/evaluate.py<br/>mean_reward, mean_laps, episode length"]

    car --> rays --> obs
    track --> rays
    track --> rew
    car --> rew
    obs --> pol --> act
    act --> car
    rew --> roll
    obs --> roll
    roll --> upd --> pol
    done --> roll
    cb --> tb
    upd --> tb
    upd --> save
    save --> watch
    save --> ev
    env --> cb
```

One environment step, in the order the code runs it:

```mermaid
sequenceDiagram
    participant P as PPO policy
    participant E as RacingEnv.step
    participant C as Car
    participant T as Track

    P->>E: action = [steering, throttle]
    E->>C: car.step(steering, throttle)
    E->>T: get_progress(x, y, prev_waypoint_idx)
    T-->>E: new waypoint index, progress_delta
    alt total progress reaches 50 waypoints
        E->>E: lap_count += 1, wrap the progress counter
    end
    E->>T: get_nearest_waypoint and is_on_track
    T-->>E: dist_to_center, on_track
    E->>E: heading_error against the next waypoint bearing
    E->>E: compute_reward(...)
    E->>C: cast_rays for the next observation
    E-->>P: obs, reward, terminated (off track),<br/>truncated (step limit), info with lap_count
```

## Evaluation

`agent/evaluate.py` rolls out a trained policy for `n_episodes` and reports:

| Metric | Description |
|--------|-------------|
| `mean_reward` | Average undiscounted episode return across rollouts |
| `mean_laps` | Average completed laps per episode (primary task-success signal) |
| Episode length | Mean steps survived; truncation indicates the agent stayed on track |

Run:

```bash
python -m agent.evaluate --model-path ./models/ppo_racing --n-episodes 50
```

Recommended additional offline diagnostics:

- **Lap completion rate**: fraction of episodes with `lap_count >= 1` (success rate).
- **Off-track termination rate**: fraction of episodes ending in the off-track penalty.
- **Mean lap time**: steps per completed lap, lower is better, indicates speed–stability trade-off.
- **Reward decomposition**: log progress / speed / alignment / penalty terms separately to
  TensorBoard to detect reward hacking.
- **Robustness sweep**: re-evaluate the same checkpoint with perturbed track radius / friction
  in `configs/env_config.yaml` to estimate generalization.

Training curves (`ep_rew_mean`, `ep_len_mean`, `policy_loss`, `value_loss`) are streamed to
`./tensorboard_logs/` and viewable via `tensorboard --logdir ./tensorboard_logs`.

## Reward Engineering

| Component | Value |
|---|---|
| Progress | +progress_delta × 10 |
| Speed | +speed × 0.1 (on-track only) |
| Heading alignment | +max(0, 1 - \|heading_err\| / 90) × 0.5 |
| Off-track penalty | −10 (terminal) |
| Lap completion | +100 |
| Time penalty | −0.01 per step |
