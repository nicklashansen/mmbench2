import gymnasium as gym
import numpy as np
import pygame
import math


class RocketCollectEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, max_episode_steps=500):
        super().__init__()
        self.max_episode_steps = max_episode_steps

        # obs = [agent_x,y,vx,vy, coin_x,y, dx,dy, collision_flag] + asteroid states
        self.max_obstacles = 3
        self.obs_dim = 9 + self.max_obstacles * 4
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-np.ones(2, dtype=np.float32), high=np.ones(2, dtype=np.float32)
        )

        # Screen
        self.width, self.height = 224, 224

        # Physics
        self.thrust = 0.75
        self.max_speed = 4.0
        self.damping = 0.98

        # Entities
        self.agent_r = 10
        self.coin_r = int(6 * 1.5)
        self.obs_r = 14

        # State
        self.agent_x = self.agent_y = 0.0
        self.agent_vx = self.agent_vy = 0.0
        self.coin_x = self.coin_y = 0.0
        self.obstacles = []

        # Bookkeeping
        self.steps = 0
        self.score = 0
        self.collision_timer = 0
        self._surface = None
        self.clock = None
        self.font = None
        self.effects = []       # coin sparkles
        self.text_effects = []  # floating text

        # Background stars
        self.stars = [(np.random.randint(0, self.width),
                       np.random.randint(0, self.height),
                       np.random.randint(1, 3)) for _ in range(40)]

    # --------------------- Utils ----------------------

    def _spawn_entities(self):
        self.agent_x = self.width / 2
        self.agent_y = self.height - 40
        self.agent_vx = self.agent_vy = 0.0
        self._spawn_coin()
        self.obstacles = []
        self._update_obstacle_count()

    def _spawn_coin(self):
        self.coin_x = self.np_random.integers(30, self.width - 30)
        self.coin_y = self.np_random.integers(40, self.height // 2)

    def _spawn_obstacle(self):
        # Choose spawn side (off-screen)
        side = self.np_random.integers(0, 4)
        if side == 0:      # top
            x = self.np_random.integers(20, self.width - 20); y = -self.obs_r * 2
        elif side == 1:    # bottom
            x = self.np_random.integers(20, self.width - 20); y = self.height + self.obs_r * 2
        elif side == 2:    # left
            x = -self.obs_r * 2; y = self.np_random.integers(20, self.height - 20)
        else:              # right
            x = self.width + self.obs_r * 2; y = self.np_random.integers(20, self.height - 20)

        # Aim toward an interior target (guarantees entry)
        tx = float(self.np_random.uniform(self.width * 0.25,  self.width * 0.75))
        ty = float(self.np_random.uniform(self.height * 0.25, self.height * 0.75))
        dx, dy = tx - x, ty - y
        mag = math.hypot(dx, dy) + 1e-6
        speed = float(self.np_random.uniform(1.0, 2.2))
        vx, vy = speed * dx / mag, speed * dy / mag

        # Persistent jagged shape
        num_pts = 8
        angles = np.linspace(0, 2 * np.pi, num_pts, endpoint=False)
        shape = [((self.obs_r + self.np_random.integers(-3, 4)) * np.cos(a),
                (self.obs_r + self.np_random.integers(-3, 4)) * np.sin(a))
                for a in angles]

        return {
            "x": x, "y": y, "vx": vx, "vy": vy,
            "angle": 0.0,
            "rot_speed": float(self.np_random.uniform(-0.02, 0.02)),
            "shape": shape,
            "entered": False,
        }

    def _update_obstacle_count(self):
        target_n = min(1 + self.steps // 150, self.max_obstacles)
        while len(self.obstacles) < target_n:
            self.obstacles.append(self._spawn_obstacle())

    def _circle_collision(self, ax, ay, ar, bx, by, br):
        return np.hypot(ax - bx, ay - by) < (ar + br)

    def _get_obs(self):
        dx = (self.coin_x - self.agent_x) / self.width
        dy = (self.coin_y - self.agent_y) / self.height
        collision_flag = 1.0 if self.collision_timer > 0 else 0.0

        obs = [
            self.agent_x / self.width,
            self.agent_y / self.height,
            self.agent_vx / self.max_speed,
            self.agent_vy / self.max_speed,
            self.coin_x / self.width,
            self.coin_y / self.height,
            dx, dy,
            collision_flag,
        ]

        for i in range(self.max_obstacles):
            if i < len(self.obstacles):
                o = self.obstacles[i]
                obs += [o["x"] / self.width, o["y"] / self.height,
                        o["vx"] / self.max_speed, o["vy"] / self.max_speed]
            else:
                obs += [0.0, 0.0, 0.0, 0.0]

        return np.array(obs, dtype=np.float32)

    # --------------------- Gym API ----------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0
        self._spawn_entities()
        self.collision_timer = 0
        self.effects.clear()
        self.text_effects.clear()
        
        # Background stars
        self.stars = [(np.random.randint(0, self.width),
                       np.random.randint(0, self.height),
                       np.random.randint(1, 3)) for _ in range(40)]
        
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # Rocket physics
        self.agent_vx += float(action[0]) * self.thrust
        self.agent_vy += float(action[1]) * self.thrust
        self.agent_vx *= self.damping
        self.agent_vy *= self.damping
        speed = np.hypot(self.agent_vx, self.agent_vy)
        if speed > self.max_speed:
            self.agent_vx *= self.max_speed / speed
            self.agent_vy *= self.max_speed / speed

        # Clamp rocket inside 5px boundary
        margin = 5
        self.agent_x = np.clip(self.agent_x + self.agent_vx,
                               margin + self.agent_r, self.width - margin - self.agent_r)
        self.agent_y = np.clip(self.agent_y + self.agent_vy,
                               margin + self.agent_r, self.height - margin - self.agent_r)

        # Update obstacles: bounce inside screen
        for o in self.obstacles:
            o["x"] += o["vx"]
            o["y"] += o["vy"]
            o["angle"] += o["rot_speed"]

            # Mark as 'entered' once fully inside; before that, no bounce
            if not o["entered"] and self._is_inside(o["x"], o["y"], self.obs_r):
                o["entered"] = True

            # Bounce only after entering the playfield
            if o["entered"]:
                if o["x"] < self.obs_r:
                    o["x"] = self.obs_r; o["vx"] *= -1
                elif o["x"] > self.width - self.obs_r:
                    o["x"] = self.width - self.obs_r; o["vx"] *= -1
                if o["y"] < self.obs_r:
                    o["y"] = self.obs_r; o["vy"] *= -1
                elif o["y"] > self.height - self.obs_r:
                    o["y"] = self.height - self.obs_r; o["vy"] *= -1

        reward = 0.0

        # Proximity reward to coin
        dist = np.hypot(self.agent_x - self.coin_x, self.agent_y - self.coin_y)
        max_dist = np.hypot(self.width, self.height)
        reward += 0.002 * (1.0 - np.clip(dist / max_dist, 0, 1))

        # Coin collection
        if self._circle_collision(self.agent_x, self.agent_y, self.agent_r,
                          self.coin_x, self.coin_y, self.coin_r):
            reward += 1.0
            self.score += 1
            # Sparkle effect
            self.effects.append({"x": self.coin_x, "y": self.coin_y, "timer": 8})
            # Floating +1 text effect
            self.text_effects.append({
                "x": self.agent_x,
                "y": self.agent_y - 15,
                "value": "+1",
                "timer": 20
            })
            self._spawn_coin()

        # Obstacle collisions
        for o in self.obstacles:
            if self._circle_collision(self.agent_x, self.agent_y, self.agent_r, o["x"], o["y"], self.obs_r):
                reward -= 0.2
                self.collision_timer = 6  # flash & shake for ~6 frames

        self.steps += 1
        self._update_obstacle_count()
        if self.collision_timer > 0:
            self.collision_timer -= 1
        truncated = self.steps >= self.max_episode_steps
        return self._get_obs(), reward, False, truncated, {}

    # --------------------- Render ----------------------

    def _is_inside(self, x, y, r):
        return (r <= x <= self.width - r) and (r <= y <= self.height - r)

    def _draw_background(self):
        for y in range(self.height):
            c = int(20 + 40 * (y / self.height))
            self._surface.fill((c // 2, 0, c), (0, y, self.width, 1))
        for (sx, sy, r) in self.stars:
            pygame.draw.circle(self._surface, (255, 255, 255), (sx, sy), r)

    def _draw_rocket(self, thrust):
        x, y = int(self.agent_x), int(self.agent_y)
        pts = [(x, y - 14), (x - 8, y + 10), (x + 8, y + 10)]
        pygame.draw.polygon(self._surface, (180, 180, 255), pts)
        pygame.draw.polygon(self._surface, (150, 150, 200),
                            [(x - 8, y + 10), (x - 12, y + 16), (x - 4, y + 10)])
        pygame.draw.polygon(self._surface, (150, 150, 200),
                            [(x + 8, y + 10), (x + 12, y + 16), (x + 4, y + 10)])
        if np.linalg.norm(thrust) > 0.05:
            flame_len = int(3 + 4 * np.linalg.norm(thrust) + np.random.randint(-1, 2))
            flame_col = (255, np.random.randint(150, 200), 80)
            pygame.draw.polygon(self._surface, flame_col,
                                [(x - 3, y + 10), (x + 3, y + 10), (x, y + 10 + flame_len)])

    def _draw_asteroid(self, o):
        x, y, angle = o["x"], o["y"], o["angle"]
        pts = []
        for (px, py) in o["shape"]:
            rx = px * math.cos(angle) - py * math.sin(angle)
            ry = px * math.sin(angle) + py * math.cos(angle)
            pts.append((x + rx, y + ry))
        pygame.draw.polygon(self._surface, (120, 120, 120), pts)

    def _draw_coin(self):
        x, y = int(self.coin_x), int(self.coin_y)
        pygame.draw.circle(self._surface, (255, 215, 0), (x, y), self.coin_r)
        pygame.draw.circle(self._surface, (200, 120, 0), (x, y), self.coin_r, 2)

    def render(self):
        if self._surface is None:
            pygame.init()
            self._surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            self.clock = pygame.time.Clock()
        if self.font is None:
            self.font = pygame.font.SysFont("Arial", 16, bold=True)

        # Base rendering
        self._draw_background()
        self._draw_rocket([self.agent_vx, self.agent_vy])
        for o in self.obstacles:
            self._draw_asteroid(o)
        self._draw_coin()

        # Draw coin pickup sparkles
        new_effects = []
        for eff in self.effects:
            t = eff["timer"]
            radius = int(self.coin_r + (8 - t) * 2)
            alpha = int(255 * (t / 8))
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.circle(overlay, (255, 220, 100, alpha),
                            (int(eff["x"]), int(eff["y"])), radius, 2)
            self._surface.blit(overlay, (0, 0))
            eff["timer"] -= 1
            if eff["timer"] > 0:
                new_effects.append(eff)
        self.effects = new_effects

        # Score text
        text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self._surface.blit(text, (5, 5))

        # Floating text effects (+1 on coin pickup)
        new_texts = []
        for eff in self.text_effects:
            t = eff["timer"]
            alpha = int(255 * (t / 20))  # fade out
            text_surface = self.font.render(eff["value"], True, (255, 255, 0))
            text_surface.set_alpha(alpha)
            self._surface.blit(text_surface, (int(eff["x"]), int(eff["y"] - (20 - t) * 0.5)))
            eff["timer"] -= 1
            if eff["timer"] > 0:
                new_texts.append(eff)
        self.text_effects = new_texts

        # Collision visual: red overlay + shake
        shake_x, shake_y = 0, 0
        if self.collision_timer > 0:
            shake_x = np.random.randint(-3, 4)
            shake_y = np.random.randint(-3, 4)
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((255, 0, 0, 80))
            self._surface.blit(overlay, (0, 0))

        # Apply shake by drawing onto a larger canvas
        canvas = pygame.Surface((self.width + 6, self.height + 6))
        canvas.fill((0, 0, 0))
        canvas.blit(self._surface, (3 + shake_x, 3 + shake_y))

        arr = np.transpose(np.array(pygame.surfarray.pixels3d(canvas))[3:-3, 3:-3], (1, 0, 2)).copy()
        return arr

    def close(self):
        if self._surface is not None:
            pygame.quit()
            self._surface = None
