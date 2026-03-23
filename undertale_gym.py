"""
Undertale Battle Gymnasium
A custom Gymnasium environment that simulates the Undertale battle box.
Designed for training DQN agents at uncapped speed.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import math
import random

# ─── Arena Constants (matching the real game's pixel-perfect masks) ───
WINDOW_W, WINDOW_H = 640, 480
MASK_TOP = (WINDOW_H // 2) + 26       # 266
MASK_BOTTOM = (WINDOW_H - (WINDOW_H // 6)) - 14  # 386
MASK_LEFT = (WINDOW_W // 3) + 29      # 242
MASK_RIGHT = (WINDOW_W - (WINDOW_W // 3)) - 31    # 395

ARENA_W = MASK_RIGHT - MASK_LEFT  # 153
ARENA_H = MASK_BOTTOM - MASK_TOP  # 120

# Heart constants
HEART_SIZE = 8
HEART_SPEED = 3

# Bullet constants
BULLET_RADIUS = 3
BULLET_SPEED_MIN = 2
BULLET_SPEED_MAX = 5
MAX_BULLETS = 40
SPAWN_RATE = 0.3  # Probability of spawning a bullet each step


class Bullet:
    """A single bullet projectile."""
    def __init__(self, x, y, vx, vy, radius=BULLET_RADIUS):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius

    def update(self):
        self.x += self.vx
        self.y += self.vy

    def is_offscreen(self):
        """Check if bullet is far outside the arena (with margin)."""
        margin = 20
        return (self.x < MASK_LEFT - margin or self.x > MASK_RIGHT + margin or
                self.y < MASK_TOP - margin or self.y > MASK_BOTTOM + margin)

    def collides_with_heart(self, hx, hy, heart_half):
        """Simple AABB vs circle collision."""
        # Find nearest point on heart rect to bullet center
        closest_x = max(hx - heart_half, min(self.x, hx + heart_half))
        closest_y = max(hy - heart_half, min(self.y, hy + heart_half))
        dist_sq = (self.x - closest_x) ** 2 + (self.y - closest_y) ** 2
        return dist_sq <= self.radius ** 2


class UndertaleEnv(gym.Env):
    """
    Custom Gymnasium environment simulating the Undertale battle box.
    
    Action Space: Discrete(9)
        0: stay, 1: up, 2: down, 3: left, 4: right,
        5: up_left, 6: up_right, 7: down_left, 8: down_right
    
    Observation Space: Box(0, 255, (480, 640, 3), uint8) — BGR image
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    ACTION_MAP = {
        0: (0, 0),       # stay
        1: (0, -1),      # up
        2: (0, 1),       # down
        3: (-1, 0),      # left
        4: (1, 0),       # right
        5: (-0.707, -0.707),  # up_left
        6: (0.707, -0.707),   # up_right
        7: (-0.707, 0.707),   # down_left
        8: (0.707, 0.707),    # down_right
    }

    def __init__(self, render_mode=None, pattern="random", max_steps=3000, difficulty=50):
        super().__init__()
        self.render_mode = render_mode
        self.pattern = pattern
        self.max_steps = max_steps
        
        # Difficulty 0-100 scales spawn rate, bullet speed, and max bullets
        d = max(0, min(100, difficulty)) / 100.0  # Normalize to 0.0-1.0
        self.difficulty = difficulty
        self.spawn_rate = 0.05 + d * 0.55       # 0.05 at diff=0, 0.60 at diff=100
        self.bullet_speed_min = 1 + d * 2       # 1 at diff=0, 3 at diff=100
        self.bullet_speed_max = 2 + d * 6       # 2 at diff=0, 8 at diff=100
        self.max_bullets = int(10 + d * 50)     # 10 at diff=0, 60 at diff=100

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(WINDOW_H, WINDOW_W, 3),
            dtype=np.uint8
        )

        # State
        self.heart_x = 0.0
        self.heart_y = 0.0
        self.bullets = []
        self.step_count = 0
        self.total_reward = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Place heart at center of arena
        self.heart_x = float(MASK_LEFT + ARENA_W // 2)
        self.heart_y = float(MASK_TOP + ARENA_H // 2)
        self.bullets = []
        self.step_count = 0
        self.total_reward = 0.0

        if options and "pattern" in options:
            self.pattern = options["pattern"]

        obs = self._render_frame()
        return obs, {"pattern": self.pattern}

    def step(self, action):
        self.step_count += 1

        # ── Move heart ──
        dx, dy = self.ACTION_MAP[action]
        self.heart_x += dx * HEART_SPEED
        self.heart_y += dy * HEART_SPEED

        # Clamp to arena bounds
        half = HEART_SIZE / 2
        self.heart_x = max(MASK_LEFT + half, min(MASK_RIGHT - half, self.heart_x))
        self.heart_y = max(MASK_TOP + half, min(MASK_BOTTOM - half, self.heart_y))

        # ── Spawn bullets ──
        self._spawn_bullets()

        # ── Update bullets ──
        for b in self.bullets:
            b.update()

        # ── Remove offscreen bullets ──
        self.bullets = [b for b in self.bullets if not b.is_offscreen()]

        # ── Collision detection ──
        hit = False
        for b in self.bullets:
            if b.collides_with_heart(self.heart_x, self.heart_y, half):
                hit = True
                break

        # ── Rewards ──
        reward = 1.0  # Survived this step

        # Small center bonus (encourages staying central)
        arena_cx = MASK_LEFT + ARENA_W / 2
        arena_cy = MASK_TOP + ARENA_H / 2
        dist_to_center = math.sqrt((self.heart_x - arena_cx)**2 + (self.heart_y - arena_cy)**2)
        max_dist = math.sqrt((ARENA_W/2)**2 + (ARENA_H/2)**2)
        center_bonus = 0.1 * (1.0 - dist_to_center / max_dist)
        reward += center_bonus

        terminated = False
        if hit:
            reward = -100.0
            terminated = True

        truncated = self.step_count >= self.max_steps
        self.total_reward += reward

        obs = self._render_frame()
        info = {
            "step": self.step_count,
            "total_reward": self.total_reward,
            "num_bullets": len(self.bullets),
            "hit": hit
        }

        if self.render_mode == "human":
            self._show_window(obs)

        return obs, reward, terminated, truncated, info

    def _spawn_bullets(self):
        """Spawn bullets based on the current pattern."""
        if len(self.bullets) >= self.max_bullets:
            return

        if self.pattern == "rain_down":
            self._spawn_rain_down()
        elif self.pattern == "rain_sides":
            self._spawn_rain_sides()
        elif self.pattern == "aimed":
            self._spawn_aimed()
        elif self.pattern == "mixed":
            choice = random.choice(["rain_down", "rain_sides", "aimed", "random"])
            self._spawn_pattern(choice)
        else:  # "random"
            self._spawn_random()

    def _spawn_pattern(self, p):
        """Helper to dispatch to a specific pattern."""
        if p == "rain_down": self._spawn_rain_down()
        elif p == "rain_sides": self._spawn_rain_sides()
        elif p == "aimed": self._spawn_aimed()
        else: self._spawn_random()

    def _spawn_rain_down(self):
        if random.random() < self.spawn_rate:
            x = random.uniform(MASK_LEFT, MASK_RIGHT)
            speed = random.uniform(self.bullet_speed_min, self.bullet_speed_max)
            self.bullets.append(Bullet(x, MASK_TOP - 10, 0, speed))

    def _spawn_rain_sides(self):
        if random.random() < self.spawn_rate * 0.5:
            y = random.uniform(MASK_TOP, MASK_BOTTOM)
            speed = random.uniform(self.bullet_speed_min, self.bullet_speed_max)
            self.bullets.append(Bullet(MASK_LEFT - 10, y, speed, 0))
        if random.random() < self.spawn_rate * 0.5:
            y = random.uniform(MASK_TOP, MASK_BOTTOM)
            speed = random.uniform(self.bullet_speed_min, self.bullet_speed_max)
            self.bullets.append(Bullet(MASK_RIGHT + 10, y, -speed, 0))

    def _spawn_aimed(self):
        if random.random() < self.spawn_rate * 0.4:
            side = random.randint(0, 3)
            if side == 0:
                x = random.uniform(MASK_LEFT, MASK_RIGHT)
                y = MASK_TOP - 10
            elif side == 1:
                x = random.uniform(MASK_LEFT, MASK_RIGHT)
                y = MASK_BOTTOM + 10
            elif side == 2:
                x = MASK_LEFT - 10
                y = random.uniform(MASK_TOP, MASK_BOTTOM)
            else:
                x = MASK_RIGHT + 10
                y = random.uniform(MASK_TOP, MASK_BOTTOM)

            dx = self.heart_x - x
            dy = self.heart_y - y
            dist = max(1, math.sqrt(dx*dx + dy*dy))
            speed = random.uniform(self.bullet_speed_min, self.bullet_speed_max)
            self.bullets.append(Bullet(x, y, dx/dist * speed, dy/dist * speed))

    def _spawn_random(self):
        if random.random() < self.spawn_rate:
            side = random.randint(0, 3)
            if side == 0:
                x = random.uniform(MASK_LEFT, MASK_RIGHT)
                y = MASK_TOP - 10
                angle = random.uniform(math.pi * 0.2, math.pi * 0.8)
            elif side == 1:
                x = random.uniform(MASK_LEFT, MASK_RIGHT)
                y = MASK_BOTTOM + 10
                angle = random.uniform(-math.pi * 0.8, -math.pi * 0.2)
            elif side == 2:
                x = MASK_LEFT - 10
                y = random.uniform(MASK_TOP, MASK_BOTTOM)
                angle = random.uniform(-math.pi * 0.3, math.pi * 0.3)
            else:
                x = MASK_RIGHT + 10
                y = random.uniform(MASK_TOP, MASK_BOTTOM)
                angle = random.uniform(math.pi * 0.7, math.pi * 1.3)

            speed = random.uniform(self.bullet_speed_min, self.bullet_speed_max)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            self.bullets.append(Bullet(x, y, vx, vy))

    def _render_frame(self):
        """Render the current state as a 640x480 BGR image."""
        # Black background
        frame = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)

        # Purple masks (matching the procedural dodger's masks)
        purple = (128, 0, 128)
        cv2.rectangle(frame, (0, 0), (WINDOW_W, MASK_TOP), purple, -1)
        cv2.rectangle(frame, (0, MASK_BOTTOM), (WINDOW_W, WINDOW_H), purple, -1)
        cv2.rectangle(frame, (0, MASK_TOP), (MASK_LEFT, MASK_BOTTOM), purple, -1)
        cv2.rectangle(frame, (MASK_RIGHT, MASK_TOP), (WINDOW_W, MASK_BOTTOM), purple, -1)

        # White arena border (the battle box walls)
        cv2.rectangle(frame, (MASK_LEFT, MASK_TOP), (MASK_RIGHT, MASK_BOTTOM), (255, 255, 255), 2)

        # Draw bullets (white circles)
        for b in self.bullets:
            bx, by = int(b.x), int(b.y)
            if 0 <= bx < WINDOW_W and 0 <= by < WINDOW_H:
                cv2.circle(frame, (bx, by), b.radius, (255, 255, 255), -1)

        # Draw heart (red, filled)
        hx, hy = int(self.heart_x), int(self.heart_y)
        half = HEART_SIZE // 2
        cv2.rectangle(frame, (hx - half, hy - half), (hx + half, hy + half), (0, 0, 255), -1)

        return frame

    def _show_window(self, frame):
        """Display the frame in an OpenCV window."""
        display = frame.copy()
        cv2.putText(display, f"Step: {self.step_count}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(display, f"Bullets: {len(self.bullets)}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(display, f"Reward: {self.total_reward:.1f}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(display, f"Pattern: {self.pattern}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
        cv2.putText(display, f"Difficulty: {self.difficulty}", (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
        cv2.imshow("Undertale Gym", display)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()


# ─── Interactive Demo ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    import argparse
    parser = argparse.ArgumentParser(description="Undertale Gym Interactive Demo")
    parser.add_argument("pattern", nargs="?", default="random", help="Bullet pattern")
    parser.add_argument("-d", "--difficulty", type=int, default=50, help="Difficulty 0-100")
    args = parser.parse_args()

    print(f"Starting Undertale Gym demo | Pattern: {args.pattern} | Difficulty: {args.difficulty}")
    print("Controls: WASD to move, Q to quit")

    env = UndertaleEnv(render_mode="human", pattern=args.pattern, difficulty=args.difficulty)
    obs, info = env.reset()

    # Key → action mapping
    key_action = {
        ord('w'): 1, ord('s'): 2, ord('a'): 3, ord('d'): 4,
        82: 1,  # Up arrow
        84: 2,  # Down arrow
        81: 3,  # Left arrow
        83: 4,  # Right arrow
    }

    total_steps = 0
    episodes = 0

    while True:
        # Get key press from OpenCV window
        key = cv2.waitKey(33) & 0xFF  # ~30 FPS for human play

        if key == ord('q') or key == 27:  # q or ESC
            break

        action = key_action.get(key, 0)  # Default: stay

        obs, reward, terminated, truncated, info = env.step(action)
        total_steps += 1

        if terminated:
            episodes += 1
            print(f"HIT! Episode {episodes} ended at step {info['step']} | Total reward: {info['total_reward']:.1f}")
            obs, info = env.reset()

        if truncated:
            episodes += 1
            print(f"SURVIVED! Episode {episodes} completed! | Total reward: {info['total_reward']:.1f}")
            obs, info = env.reset()

    env.close()
    print(f"\nDemo ended. {episodes} episodes, {total_steps} total steps.")
