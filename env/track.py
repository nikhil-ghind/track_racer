import json
import math
import numpy as np


class Track:
    def __init__(self, config: dict):
        self.num_waypoints = config["track"]["num_waypoints"]
        self.track_width = config["track"]["track_width"]
        self.inner_radius = config["track"]["inner_radius"]
        self.outer_radius = config["track"]["outer_radius"]
        self.center_x = 400
        self.center_y = 400
        self.waypoints = self.generate_waypoints()
        self.inner_boundary, self.outer_boundary = self.get_boundaries()

    def generate_waypoints(self) -> list:
        points = []
        for i in range(self.num_waypoints):
            angle = 2 * math.pi * i / self.num_waypoints
            r = (self.inner_radius + self.outer_radius) / 2
            x = self.center_x + r * math.cos(angle)
            y = self.center_y + r * math.sin(angle)
            points.append((x, y))
        return points

    def get_boundaries(self) -> tuple:
        inner, outer = [], []
        half_w = self.track_width / 2
        for i in range(self.num_waypoints):
            angle = 2 * math.pi * i / self.num_waypoints
            r_inner = (self.inner_radius + self.outer_radius) / 2 - half_w
            r_outer = (self.inner_radius + self.outer_radius) / 2 + half_w
            inner.append((self.center_x + r_inner * math.cos(angle), self.center_y + r_inner * math.sin(angle)))
            outer.append((self.center_x + r_outer * math.cos(angle), self.center_y + r_outer * math.sin(angle)))
        return inner, outer

    def get_nearest_waypoint(self, x: float, y: float) -> tuple:
        dists = [math.hypot(x - wx, y - wy) for wx, wy in self.waypoints]
        idx = int(np.argmin(dists))
        dist_to_center = math.hypot(x - self.center_x, y - self.center_y) - (self.inner_radius + self.outer_radius) / 2
        return idx, abs(dist_to_center)

    def get_progress(self, x: float, y: float, prev_idx: int) -> tuple:
        new_idx, dist = self.get_nearest_waypoint(x, y)
        delta = (new_idx - prev_idx) % self.num_waypoints
        if delta > self.num_waypoints // 2:
            delta = 0
        progress = min(delta, 5)
        return new_idx, progress

    def is_on_track(self, x: float, y: float) -> bool:
        dist = math.hypot(x - self.center_x, y - self.center_y)
        center_r = (self.inner_radius + self.outer_radius) / 2
        return abs(dist - center_r) < self.track_width / 2

    def save_waypoints(self, path: str):
        with open(path, "w") as f:
            json.dump(self.waypoints, f)
