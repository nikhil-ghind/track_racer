# Track Racer

## Project Overview

An AI agent trained to autonomously drive and complete laps on a racing track using deep Reinforcement Learning with Proximal Policy Optimization (PPO). The agent learns from a reward function engineered for lap completion and track navigation, and the training progress is visualized in real time. The simulation environment is either a custom-built Gymnasium environment or an existing racing simulator.

**Key Goals:**
- Build or integrate a 2D/3D racing track simulation environment compatible with Gymnasium
- Engineer a reward function that incentivizes lap completion, speed, and staying on track
- Train a PPO agent with an appropriate actor-critic architecture
- Visualize the agent's driving behavior in real time during and after training
- Achieve consistent lap completion within N training steps

---

## Tech Stack

| Component | Technology | Version |
|---|---|---|
| Language | Python | 3.11+ |
| RL Library | Stable-Baselines3 | 2.3.x |
| Environment | Gymnasium | 0.29.x |
| Physics/Rendering | Pygame | 2.5.x |
| Neural Networks | PyTorch | 2.3.x |
| Monitoring | TensorBoard | 2.17.x |
| Numerical | NumPy | 1.x |
| Config | YAML + argparse | stdlib |
| Testing | pytest | 8.x |

---

## Architecture Overview

```
RacingEnv (Gymnasium)
  - Track geometry (waypoints, boundaries)
  - Car physics (velocity, heading, friction)
  - Observation: [dist_to_center, heading_error, speed, ray_sensor_readings x N]
  - Action: [steering, throttle/brake]
  - Reward function
       |
       v
PPO Agent (Stable-Baselines3)
  - MlpPolicy (Actor + Critic, shared MLP backbone)
  - Rollout buffer: N steps per update
  - Clipped surrogate objective + value loss + entropy bonus
       |
       v
TensorBoard Logger
  - ep_rew_mean, ep_len_mean, policy_loss, value_loss
       |
       v
Real-time Renderer (Pygame)
  - Draws track, car position, ray sensors
  - Shows current reward, lap count
```

---

## Phase 1: Project Scaffolding

**Goal:** Set up project layout and dependencies.

### Tasks

1. Create directory structure:
   ```
   intelligentCarRacingSimulation/
   ├── env/
   │   ├── __init__.py
   │   ├── racing_env.py
   │   ├── track.py
   │   ├── car.py
   │   └── renderer.py
   ├── agent/
   │   ├── __init__.py
   │   ├── train.py
   │   ├── evaluate.py
   │   └── callbacks.py
   ├── configs/
   │   ├── env_config.yaml
   │   └── ppo_config.yaml
   ├── tests/
   │   ├── test_env.py
   │   ├── test_car.py
   │   └── test_track.py
   ├── scripts/
   │   ├── train.py
   │   └── watch_agent.py
   ├── requirements.txt
   └── assets/
       └── track_waypoints.json
   ```

2. Create `requirements.txt`:
   ```
   gymnasium==0.29.*
   stable-baselines3==2.3.*
   torch==2.3.*
   pygame==2.5.*
   numpy==1.*
   tensorboard==2.17.*
   pyyaml==6.*
   pytest==8.*
   ```

3. Create `configs/env_config.yaml`:
   ```yaml
   track:
     num_waypoints: 50
     track_width: 80          # pixels
     inner_radius: 200
     outer_radius: 280
   car:
     max_speed: 10.0
     acceleration: 0.5
     brake_deceleration: 0.8
     steering_rate: 5.0       # degrees per step
     friction: 0.05
   observation:
     num_ray_sensors: 9
     ray_length: 200
   max_steps_per_episode: 2000
   ```

4. Create `configs/ppo_config.yaml`:
   ```yaml
   policy: "MlpPolicy"
   total_timesteps: 2000000
   n_steps: 2048
   batch_size: 64
   n_epochs: 10
   gamma: 0.99
   gae_lambda: 0.95
   clip_range: 0.2
   ent_coef: 0.01
   vf_coef: 0.5
   learning_rate: 3.0e-4
   tensorboard_log: "./tensorboard_logs"
   model_save_path: "./models/ppo_racing"
   ```

5. Run: `pip install -r requirements.txt`

---

## Phase 2: Track Definition

**Goal:** Define the race track as a series of waypoints with inner/outer boundaries.

### Tasks

1. **`env/track.py`**
   - Define `class Track`:
     - `__init__(self, config: dict)`: generates a circular or oval track from config parameters
     - `generate_waypoints(self) -> list[tuple[float, float]]`: creates `num_waypoints` center-line points evenly spaced around an ellipse defined by `inner_radius` and `outer_radius`
     - `get_boundaries(self) -> tuple[list, list]`: returns `(inner_boundary_points, outer_boundary_points)` as lists of `(x, y)` tuples, offset `track_width/2` inward and outward from center line
     - `get_nearest_waypoint(self, x: float, y: float) -> tuple[int, float]`: returns `(waypoint_index, distance_to_center_line)` for a given car position
     - `get_progress(self, x: float, y: float, prev_waypoint_idx: int) -> tuple[int, float]`: returns new nearest waypoint index and distance traveled along center line since last step
     - `is_on_track(self, x: float, y: float) -> bool`: returns True if `distance_to_center_line < track_width / 2`
   - Save default waypoints to `assets/track_waypoints.json` via a helper function `save_waypoints(track: Track, path: str)`

2. **`tests/test_track.py`**
   - Test `generate_waypoints()` returns exactly `num_waypoints` points
   - Test `is_on_track()` returns True for a point at track center, False for a point 500px from center
   - Test `get_progress()` returns increasing waypoint index as a car moves forward

---

## Phase 3: Car Physics

**Goal:** Implement a simple bicycle-model car physics simulation.

### Tasks

1. **`env/car.py`**
   - Define `class Car`:
     - State: `x: float`, `y: float`, `heading: float` (degrees), `speed: float`
     - `__init__(self, start_x: float, start_y: float, start_heading: float, config: dict)`
     - `def step(self, steering: float, throttle: float) -> None`:
       - Clamps `steering` to `[-1, 1]`, `throttle` to `[-1, 1]`
       - Updates heading: `self.heading += steering * config.steering_rate`
       - Updates speed: if `throttle > 0`, `speed += throttle * config.acceleration`; if `throttle < 0`, `speed += throttle * config.brake_deceleration`; apply friction: `speed *= (1 - config.friction)`; clamp speed to `[0, config.max_speed]`
       - Updates position: `x += speed * cos(heading_rad)`, `y += speed * sin(heading_rad)`
     - `def reset(self, x: float, y: float, heading: float) -> None`: resets state
     - `def cast_rays(self, track: Track, num_rays: int, ray_length: float) -> list[float]`: casts `num_rays` rays evenly distributed around the car's forward direction (`[-90°, +90°]`); for each ray, steps along the ray and returns the normalized distance to the nearest boundary intersection (returns `1.0` if no intersection within `ray_length`)

2. **`tests/test_car.py`**
   - Test that applying max throttle for 10 steps increases speed
   - Test that applying brake from max speed reduces speed to 0 eventually
   - Test that `cast_rays()` returns `num_rays` values, all in `[0.0, 1.0]`

---

## Phase 4: Gymnasium Environment

**Goal:** Implement a fully Gymnasium-compliant racing environment.

### Tasks

1. **`env/racing_env.py`**
   - Define `class RacingEnv(gymnasium.Env)`:
     - `metadata = {"render_modes": ["human", "rgb_array"]}`
     - `__init__(self, config_path: str, render_mode: str | None = None)`:
       - Loads `env_config.yaml`
       - Creates `Track` and `Car` instances
       - Defines `observation_space`: `gymnasium.spaces.Box(low=-inf, high=inf, shape=(2 + num_ray_sensors,))` where the 2 extra dims are `[distance_to_center_norm, heading_error_norm]`
       - Defines `action_space`: `gymnasium.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)` for `[steering, throttle]`
       - Initializes renderer if `render_mode == "human"`
     - `def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]`:
       - Resets car to start position (first waypoint), heading toward second waypoint
       - Resets `step_count = 0`, `prev_waypoint_idx = 0`, `lap_count = 0`, `total_progress = 0.0`
       - Returns initial observation and empty info dict
     - `def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]`:
       - Unpacks `action` into `[steering, throttle]`
       - Calls `car.step(steering, throttle)`
       - Calls `track.get_progress(car.x, car.y, prev_waypoint_idx)` to get new waypoint and progress
       - Computes reward (see Phase 5)
       - Checks termination: `is_on_track == False` (off-road) or `step_count >= max_steps`
       - Checks truncation: `step_count >= max_steps`
       - Updates `lap_count` if `total_progress >= num_waypoints`
       - Builds observation: `np.array([dist_to_center / (track_width/2), heading_error / 180.0, *ray_distances], dtype=np.float32)`
       - Returns `(obs, reward, terminated, truncated, info)`
     - `def render(self) -> None | np.ndarray`: delegates to `Renderer`
     - `def close(self) -> None`: calls `renderer.close()` if initialized
   - Register the environment:
     ```python
     gymnasium.register(
         id="RacingEnv-v0",
         entry_point="env.racing_env:RacingEnv",
         max_episode_steps=2000,
     )
     ```

2. **`tests/test_env.py`**
   - Test `env.reset()` returns observation with correct shape
   - Test `env.step(np.array([0.0, 1.0]))` returns 5-tuple with correct types
   - Test that going off track sets `terminated=True`
   - Use `gymnasium.utils.env_checker.check_env(env)` to validate the environment

---

## Phase 5: Reward Engineering

**Goal:** Design a shaped reward function that incentivizes correct driving behavior.

### Tasks

1. Implement `def compute_reward(self, dist_to_center, heading_error, speed, progress_delta, on_track, lap_completed) -> float` as a method of `RacingEnv`:

   | Reward Component | Value | Condition |
   |---|---|---|
   | Progress reward | `+progress_delta * 10.0` | Per step, based on waypoints advanced |
   | Speed reward | `+speed * 0.1` | Only when `on_track` |
   | Heading alignment | `+max(0, (1 - abs(heading_error) / 90)) * 0.5` | Reward for facing forward |
   | Off-track penalty | `-10.0` | When `not on_track` |
   | Lap completion bonus | `+100.0` | Each time a full lap is completed |
   | Time penalty | `-0.01` | Per step, to discourage dawdling |

2. Reward shaping notes (document in docstring):
   - Progress delta is capped at 5 waypoints per step to prevent reward hacking from teleportation bugs
   - Off-track terminates the episode, so penalty is a one-time terminal penalty
   - Heading error is computed as the angle between the car's heading and the vector to the next waypoint

---

## Phase 6: Renderer

**Goal:** Build a Pygame-based real-time renderer.

### Tasks

1. **`env/renderer.py`**
   - Define `class RacingRenderer`:
     - `__init__(self, track: Track, width: int = 800, height: int = 800)`: initializes Pygame display
     - `def render(self, car: Car, step: int, reward: float, lap_count: int, ray_distances: list[float]) -> None`:
       - Fills background (dark gray)
       - Draws outer track boundary (gray)
       - Draws inner track boundary (dark green)
       - Draws center line (dashed yellow, using waypoints)
       - Draws car as a colored rectangle (red) rotated to `car.heading`
       - Draws ray sensors as lines from car position, color-coded by distance (green=far, red=close)
       - Draws HUD text (top-left): `Step: {step}`, `Reward: {reward:.2f}`, `Laps: {lap_count}`
       - Calls `pygame.display.flip()`
       - Handles `pygame.QUIT` event to allow window close
     - `def close(self) -> None`: calls `pygame.quit()`

---

## Phase 7: Agent Training

**Goal:** Train the PPO agent and save checkpoints.

### Tasks

1. **`agent/callbacks.py`**
   - Define `class LapCounterCallback(BaseCallback)` (inherits `stable_baselines3.common.callbacks.BaseCallback`):
     - Tracks total laps completed across all episodes
     - Logs `custom/lap_count` to TensorBoard every 1000 steps
   - Define `class CheckpointCallback` — use SB3's built-in `CheckpointCallback` configured for every 50,000 steps

2. **`agent/train.py`**
   - Define `def train(env_config_path: str, ppo_config_path: str) -> None`:
     - Loads configs from YAML
     - Creates `RacingEnv(config_path=env_config_path)` wrapped in `Monitor` and `DummyVecEnv`
     - Creates `PPO("MlpPolicy", env, **ppo_params, tensorboard_log=cfg.tensorboard_log, verbose=1)`
     - Adds callbacks: `LapCounterCallback()`, `CheckpointCallback(save_freq=50000, save_path="./models/")`
     - Calls `model.learn(total_timesteps=cfg.total_timesteps, callback=callbacks, progress_bar=True)`
     - Saves final model: `model.save(cfg.model_save_path)`
     - Prints final stats

3. **`scripts/train.py`**
   - CLI: `python scripts/train.py --env-config configs/env_config.yaml --ppo-config configs/ppo_config.yaml`
   - Calls `agent.train.train()`
   - Run TensorBoard in parallel: `tensorboard --logdir ./tensorboard_logs`

4. **`agent/evaluate.py`**
   - Define `def evaluate(model_path: str, env_config_path: str, n_episodes: int = 10, render: bool = True) -> dict`:
     - Loads model: `PPO.load(model_path)`
     - Creates env with `render_mode="human"` if `render=True`
     - Runs `n_episodes` episodes, records total reward, lap count, episode length per episode
     - Returns `{"mean_reward": float, "mean_laps": float, "mean_episode_length": float}`

5. **`scripts/watch_agent.py`**
   - CLI: `python scripts/watch_agent.py --model-path ./models/ppo_racing --n-episodes 5`
   - Calls `agent.evaluate.evaluate(...)` with `render=True`
   - Prints summary statistics
