import pytest
from env.track import Track

CONFIG = {
    "track": {"num_waypoints": 50, "track_width": 80, "inner_radius": 200, "outer_radius": 280},
    "car": {"max_speed": 10.0, "acceleration": 0.5, "brake_deceleration": 0.8, "steering_rate": 5.0, "friction": 0.05},
    "observation": {"num_ray_sensors": 9, "ray_length": 200},
    "max_steps_per_episode": 2000,
}


def test_waypoint_count():
    track = Track(CONFIG)
    assert len(track.waypoints) == 50


def test_on_track_center():
    track = Track(CONFIG)
    cx, cy = track.center_x, track.center_y
    r = (CONFIG["track"]["inner_radius"] + CONFIG["track"]["outer_radius"]) / 2
    import math
    x = cx + r
    y = cy
    assert track.is_on_track(x, y)


def test_off_track_far():
    track = Track(CONFIG)
    assert not track.is_on_track(0, 0)


def test_progress_increases():
    track = Track(CONFIG)
    idx0 = 0
    wp0 = track.waypoints[0]
    wp5 = track.waypoints[5]
    new_idx, prog = track.get_progress(wp5[0], wp5[1], idx0)
    assert prog >= 0
