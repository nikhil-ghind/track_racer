import pytest
from env.car import Car
from env.track import Track

CONFIG = {
    "track": {"num_waypoints": 50, "track_width": 80, "inner_radius": 200, "outer_radius": 280},
    "car": {"max_speed": 10.0, "acceleration": 0.5, "brake_deceleration": 0.8, "steering_rate": 5.0, "friction": 0.05},
    "observation": {"num_ray_sensors": 9, "ray_length": 200},
    "max_steps_per_episode": 2000,
}


def test_throttle_increases_speed():
    car = Car(400, 640, 0, CONFIG)
    for _ in range(10):
        car.step(0.0, 1.0)
    assert car.speed > 0


def test_brake_reduces_speed():
    car = Car(400, 640, 0, CONFIG)
    for _ in range(20):
        car.step(0.0, 1.0)
    initial_speed = car.speed
    for _ in range(50):
        car.step(0.0, -1.0)
    assert car.speed <= initial_speed


def test_ray_count_and_range():
    track = Track(CONFIG)
    car = Car(track.waypoints[0][0], track.waypoints[0][1], 0, CONFIG)
    rays = car.cast_rays(track, 9, 200)
    assert len(rays) == 9
    for r in rays:
        assert 0.0 <= r <= 1.0
