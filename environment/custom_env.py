import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from .rendering import AgriScanRenderer


class AgriScanEnv(gym.Env):
    """
    Custom RL environment for the AgriScan mission.
    The agent recommends treatment actions for a crop plot.
    """
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None, max_steps: int = 5):
        super().__init__()

        # === Action space ===
        # 0 = do nothing
        # 1 = apply fungicide
        # 2 = apply pesticide
        # 3 = apply fertilizer
        # 4 = call agronomist
        self.action_space = spaces.Discrete(5)

        # === Observation space ===
        # [issue_type, severity, soil_moisture, forecast_rain, previous_action]
        # issue_type: 0-3
        # severity: 0-3
        # soil_moisture: 0.0-1.0
        # forecast_rain: 0 or 1
        # previous_action: 0-4
        low = np.array([0, 0, 0.0, 0, 0], dtype=np.float32)
        high = np.array([3, 3, 1.0, 1, 4], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.max_steps = max_steps

        # Internal state variables
        self.issue_type: int = 0
        self.severity: int = 0
        self.soil_moisture: float = 0.0
        self.forecast_rain: int = 0
        self.previous_action: int = 0
        self.step_count: int = 0

        # Renderer instance
        self.renderer: Optional[AgriScanRenderer] = None
        if self.render_mode == "human":
            self.renderer = AgriScanRenderer()

    def _get_obs(self) -> np.ndarray:
        """Return current state as observation vector."""
        return np.array(
            [
                float(self.issue_type),
                float(self.severity),
                float(self.soil_moisture),
                float(self.forecast_rain),
                float(self.previous_action),
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> Dict[str, Any]:
        """Optional extra info for debugging."""
        return {
            "issue_type": self.issue_type,
            "severity": self.severity,
            "step_count": self.step_count,
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Randomly initialize a new plant episode
        self.issue_type = self.np_random.integers(0, 4)      # 0-3
        self.severity = int(self.np_random.integers(1, 3))   # 1 or 2
        self.soil_moisture = float(self.np_random.uniform(0.2, 0.8))
        self.forecast_rain = int(self.np_random.integers(0, 2))  # 0 or 1
        self.previous_action = 0
        self.step_count = 0

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human" and self.renderer is not None:
            self.renderer.render(
                obs=obs,
                action=None,
                reward=0.0,
                done=False,
                info=info,
            )

        return obs, info

    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid action {action}"
        self.step_count += 1

        # === Compute reward based on current issue & chosen action ===
        reward = 0.0

        if self.issue_type == 0:
            # Healthy plant
            if action == 0:
                reward += 2.0
            elif action in [1, 2, 3]:
                reward -= 1.0  # unnecessary treatment
            elif action == 4:
                reward += 0.0  # neutral
        else:
            # There is a real problem
            if self.issue_type == 1:  # fungal
                correct_action = 1
            elif self.issue_type == 2:  # pest
                correct_action = 2
            else:  # nutrient deficiency
                correct_action = 3

            if action == correct_action:
                reward += 3.0
            elif action == 0:
                reward -= 3.0  # ignored problem
            elif action == 4:
                reward += 1.0  # safe but not efficient
            else:
                reward -= 2.0  # wrong chemical

        # Small step penalty to encourage faster resolution
        reward -= 0.1

        # === State transition: update severity ===
        if self.issue_type == 0:
            # Healthy: maybe worsen if moisture is extreme
            if self.soil_moisture < 0.2 or self.soil_moisture > 0.9:
                self.severity = min(3, self.severity + 1)
        else:
            if (self.issue_type == 1 and action == 1) or \
               (self.issue_type == 2 and action == 2) or \
               (self.issue_type == 3 and action == 3):
                # Correct treatment -> severity decreases
                self.severity = max(0, self.severity - 1)
            elif action == 0:
                # Doing nothing tends to worsen the situation
                self.severity = min(3, self.severity + 1)
            else:
                # Wrong treatment also worsens
                self.severity = min(3, self.severity + 1)

        # Simple soil moisture dynamics
        delta_moist = self.np_random.uniform(-0.1, 0.1)
        if self.forecast_rain == 1:
            delta_moist += 0.05
        self.soil_moisture = float(
            np.clip(self.soil_moisture + delta_moist, 0.0, 1.0)
        )

        # Future weather
        self.forecast_rain = int(self.np_random.integers(0, 2))
        self.previous_action = int(action)

        # === Check terminal conditions ===
        terminated = False

        if self.severity >= 3:
            # Plant considered dead
            reward -= 5.0
            terminated = True
        elif self.severity <= 0:
            # Plant fully healthy
            reward += 5.0
            terminated = True
        elif self.step_count >= self.max_steps:
            terminated = True

        truncated = False  # not using truncation

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human" and self.renderer is not None:
            self.renderer.render(
                obs=obs,
                action=action,
                reward=reward,
                done=terminated,
                info=info,
            )

        return obs, reward, terminated, truncated, info

    def render(self):
        # Rendering is handled by the renderer in step/reset
        pass

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
