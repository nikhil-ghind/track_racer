import math
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class RacingRenderer:
    def __init__(self, track, width: int = 800, height: int = 800):
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame is required for rendering")
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Track Racer")
        self.track = track
        self.font = pygame.font.SysFont("monospace", 18)
        self.clock = pygame.time.Clock()

    def render(self, car, step: int, reward: float, lap_count: int, ray_distances: list) -> None:
        self.screen.fill((40, 40, 40))

        # Draw outer boundary
        if len(self.track.outer_boundary) > 1:
            pygame.draw.polygon(self.screen, (100, 100, 100), [(int(x), int(y)) for x, y in self.track.outer_boundary], 2)

        # Draw inner boundary
        if len(self.track.inner_boundary) > 1:
            pygame.draw.polygon(self.screen, (30, 80, 30), [(int(x), int(y)) for x, y in self.track.inner_boundary], 2)

        # Draw center line (dashed)
        for i in range(0, len(self.track.waypoints), 2):
            x, y = self.track.waypoints[i]
            pygame.draw.circle(self.screen, (200, 200, 0), (int(x), int(y)), 2)

        # Draw ray sensors
        num_rays = len(ray_distances)
        for i, d in enumerate(ray_distances):
            angle_offset = -90 + 180 * i / max(num_rays - 1, 1)
            ray_angle = math.radians(car.heading + angle_offset)
            ray_len = d * 200
            ex = car.x + ray_len * math.cos(ray_angle)
            ey = car.y + ray_len * math.sin(ray_angle)
            g = int(255 * d)
            r = int(255 * (1 - d))
            pygame.draw.line(self.screen, (r, g, 0), (int(car.x), int(car.y)), (int(ex), int(ey)), 1)

        # Draw car
        pygame.draw.circle(self.screen, (220, 50, 50), (int(car.x), int(car.y)), 8)

        # HUD
        texts = [f"Step: {step}", f"Reward: {reward:.2f}", f"Laps: {lap_count}", f"Speed: {car.speed:.1f}"]
        for i, t in enumerate(texts):
            surf = self.font.render(t, True, (255, 255, 255))
            self.screen.blit(surf, (10, 10 + i * 22))

        pygame.display.flip()
        self.clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def close(self) -> None:
        pygame.quit()
