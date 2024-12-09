import math
import numpy as np


class Car:
    def __init__(self, start_x: float, start_y: float, start_heading: float, config: dict):
        self.x = start_x
        self.y = start_y
        self.heading = start_heading  # degrees
        self.speed = 0.0
        self.config = config["car"]

    def step(self, steering: float, throttle: float) -> None:
        steering = float(np.clip(steering, -1.0, 1.0))
        throttle = float(np.clip(throttle, -1.0, 1.0))

        self.heading += steering * self.config["steering_rate"]
        self.heading = self.heading % 360

        if throttle > 0:
            self.speed += throttle * self.config["acceleration"]
        else:
            self.speed += throttle * self.config["brake_deceleration"]

        self.speed *= 1 - self.config["friction"]
        self.speed = float(np.clip(self.speed, 0.0, self.config["max_speed"]))

        heading_rad = math.radians(self.heading)
        self.x += self.speed * math.cos(heading_rad)
        self.y += self.speed * math.sin(heading_rad)

    def reset(self, x: float, y: float, heading: float) -> None:
        self.x = x
        self.y = y
        self.heading = heading
        self.speed = 0.0

    def cast_rays(self, track, num_rays: int, ray_length: float) -> list:
        distances = []
        for i in range(num_rays):
            angle_offset = -90 + 180 * i / max(num_rays - 1, 1)
            ray_angle = math.radians(self.heading + angle_offset)
            dist = 1.0
            for step in range(1, int(ray_length) + 1):
                rx = self.x + step * math.cos(ray_angle)
                ry = self.y + step * math.sin(ray_angle)
                if not track.is_on_track(rx, ry):
                    dist = step / ray_length
                    break
            distances.append(float(np.clip(dist, 0.0, 1.0)))
        return distances
