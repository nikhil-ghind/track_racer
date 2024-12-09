import math
import yaml
import numpy as np
import gymnasium
from gymnasium import spaces

from env.track import Track
from env.car import Car


class RacingEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config_path: str = "configs/env_config.yaml", render_mode=None):
        super().__init__()
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.track = Track(self.config)
        self.car = Car(0, 0, 0, self.config)
        self.render_mode = render_mode
        self.renderer = None

        num_rays = self.config["observation"]["num_ray_sensors"]
        obs_dim = 2 + num_rays
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        self.step_count = 0
        self.prev_waypoint_idx = 0
        self.lap_count = 0
        self.total_progress = 0.0
        self.last_reward = 0.0

        if render_mode == "human":
            from env.renderer import RacingRenderer
            self.renderer = RacingRenderer(self.track)

    def _get_obs(self):
        num_rays = self.config["observation"]["num_ray_sensors"]
        ray_length = self.config["observation"]["ray_length"]
        ray_distances = self.car.cast_rays(self.track, num_rays, ray_length)

        _, dist_to_center = self.track.get_nearest_waypoint(self.car.x, self.car.y)
        dist_norm = dist_to_center / (self.config["track"]["track_width"] / 2)

        wp = self.track.waypoints[self.prev_waypoint_idx]
        next_idx = (self.prev_waypoint_idx + 1) % self.track.num_waypoints
        nwp = self.track.waypoints[next_idx]
        wp_angle = math.degrees(math.atan2(nwp[1] - wp[1], nwp[0] - wp[0]))
        heading_error = (self.car.heading - wp_angle + 180) % 360 - 180
        heading_norm = heading_error / 180.0

        return np.array([dist_norm, heading_norm] + ray_distances, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        start = self.track.waypoints[0]
        next_wp = self.track.waypoints[1]
        heading = math.degrees(math.atan2(next_wp[1] - start[1], next_wp[0] - start[0]))
        self.car.reset(start[0], start[1], heading)
        self.step_count = 0
        self.prev_waypoint_idx = 0
        self.lap_count = 0
        self.total_progress = 0.0
        self.last_reward = 0.0
        return self._get_obs(), {}

    def compute_reward(self, dist_to_center, heading_error, speed, progress_delta, on_track, lap_completed) -> float:
        reward = 0.0
        reward += progress_delta * 10.0
        if on_track:
            reward += speed * 0.1
        reward += max(0.0, (1 - abs(heading_error) / 90)) * 0.5
        if not on_track:
            reward -= 10.0
        if lap_completed:
            reward += 100.0
        reward -= 0.01
        return reward

    def step(self, action):
        steering, throttle = float(action[0]), float(action[1])
        self.car.step(steering, throttle)
        self.step_count += 1

        new_idx, progress_delta = self.track.get_progress(self.car.x, self.car.y, self.prev_waypoint_idx)
        self.total_progress += progress_delta
        self.prev_waypoint_idx = new_idx

        lap_completed = False
        if self.total_progress >= self.track.num_waypoints:
            self.lap_count += 1
            self.total_progress -= self.track.num_waypoints
            lap_completed = True

        _, dist_to_center = self.track.get_nearest_waypoint(self.car.x, self.car.y)
        on_track = self.track.is_on_track(self.car.x, self.car.y)

        wp = self.track.waypoints[self.prev_waypoint_idx]
        next_idx = (self.prev_waypoint_idx + 1) % self.track.num_waypoints
        nwp = self.track.waypoints[next_idx]
        wp_angle = math.degrees(math.atan2(nwp[1] - wp[1], nwp[0] - wp[0]))
        heading_error = (self.car.heading - wp_angle + 180) % 360 - 180

        reward = self.compute_reward(
            dist_to_center, heading_error, self.car.speed, progress_delta, on_track, lap_completed
        )
        self.last_reward = reward

        terminated = not on_track
        truncated = self.step_count >= self.config["max_steps_per_episode"]

        if self.renderer and self.render_mode == "human":
            num_rays = self.config["observation"]["num_ray_sensors"]
            ray_length = self.config["observation"]["ray_length"]
            rays = self.car.cast_rays(self.track, num_rays, ray_length)
            self.renderer.render(self.car, self.step_count, reward, self.lap_count, rays)

        info = {"lap_count": self.lap_count, "progress": self.total_progress, "on_track": on_track}
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.renderer:
            return self.renderer.render(self.car, self.step_count, self.last_reward, self.lap_count, [])

    def close(self):
        if self.renderer:
            self.renderer.close()


gymnasium.register(
    id="RacingEnv-v0",
    entry_point="env.racing_env:RacingEnv",
    max_episode_steps=2000,
)
