import pygame
import sys
import numpy as np
from typing import Optional, Dict, Any


class AgriScanRenderer:
    """
    Simple 2D visualization using pygame.
    Shows:
    - A window with a "plant" whose color depends on severity
    - Text information about issue type, severity, soil moisture, etc.
    - Last action and reward
    """

    def __init__(self, width: int = 600, height: int = 400):
        pygame.init()
        self.width = width
        self.height = height
        self.window = pygame.display.set_mode((width, height))
        pygame.display.set_caption("AgriScan RL Environment")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)

    def render(
        self,
        obs: np.ndarray,
        action: Optional[int],
        reward: float,
        done: bool,
        info: Dict[str, Any],
    ):
        # Handle close events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.window.fill((255, 255, 255))  # white background

        issue_type = int(obs[0])
        severity = int(obs[1])
        soil_moisture = float(obs[2])
        forecast_rain = int(obs[3])
        previous_action = int(obs[4])

        # Choose plant color based on severity
        if severity == 0:
            plant_color = (0, 200, 0)  # green
        elif severity == 1:
            plant_color = (200, 200, 0)  # yellow
        elif severity == 2:
            plant_color = (255, 165, 0)  # orange
        else:
            plant_color = (200, 0, 0)  # red

        # Draw plant as a rectangle in the center
        plant_rect = pygame.Rect(
            self.width // 2 - 50,
            self.height // 2 - 60,
            100,
            120,
        )
        pygame.draw.rect(self.window, plant_color, plant_rect)

        issue_names = ["Healthy", "Fungal disease", "Pest attack", "Nutrient deficiency"]
        action_names = [
            "Do nothing",
            "Fungicide",
            "Pesticide",
            "Fertilizer",
            "Call agronomist",
        ]

        lines = [
            f"Issue type: {issue_names[issue_type]} ({issue_type})",
            f"Severity: {severity} (0=healthy, 3=dead)",
            f"Soil moisture: {soil_moisture:.2f}",
            f"Forecast rain: {'Yes' if forecast_rain == 1 else 'No'}",
            f"Previous action: {action_names[previous_action]} ({previous_action})",
            f"Step: {info.get('step_count', 0)}",
            f"Last reward: {reward:.2f}",
            f"Episode done: {done}",
        ]

        if action is not None:
            lines.insert(5, f"Chosen action: {action_names[action]} ({action})")

        # Render text lines on the left
        y = 20
        for line in lines:
            text_surf = self.font.render(line, True, (0, 0, 0))
            self.window.blit(text_surf, (20, y))
            y += 24

        pygame.display.flip()
        self.clock.tick(4)  # ~4 FPS

    def close(self):
        pygame.quit()
