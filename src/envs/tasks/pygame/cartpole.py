import gymnasium as gym
import numpy as np
import pygame
import math


class CartpoleEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, max_episode_steps=500,
                 task="balance"):        # "balance" or "swingup"
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.task = task

        # Screen setup
        self.width, self.height = 224, 224
        self.area_size = 184
        self.area_x = (self.width - self.area_size) // 2
        self.area_y = (self.height - self.area_size) // 2

        # --- Physics parameters (Gym style) ---
        self.gravity = 9.81
        self.masscart = 0.9
        self.masspole = 0.11
        self.total_mass = self.masspole + self.masscart
        self.length = 0.6
        self.polemass_length = self.masspole * self.length
        self.force_mag = 11.0
        self.tau = 0.025

        self.x_threshold = 2.4
        self.theta_threshold_radians = 12 * 2 * math.pi / 360

        self.state = None

        # Observation / Action
        high = np.array([self.x_threshold*2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians*2,
                         np.finfo(np.float32).max], dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Rendering params
        self.cart_w, self.cart_h = 28, 14
        self.cart_y = self.area_y + self.area_size - 50  # floor is raised
        self.ceiling_points = []
        self.rocks = []  # precomputed shapes (no per-frame randomness)
        self.lantern_phases = [0.0, 0.0]

        self.steps = 0
        self._surface = None
        self.clock = None
        self.font = None

    # ---------- World→pixel mapping ----------
    def _px_per_meter(self):
        return (self.area_size - self.cart_w) / (2 * self.x_threshold)

    def _cart_px_x(self):
        cx = self.area_x + self.area_size / 2
        return int(round(cx + self.state[0] * self._px_per_meter()))

    # ---------- Reset ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.task == "balance":
            self.state = self.np_random.uniform(low=-0.065, high=0.065, size=(4,))
        else:  # swingup
            x = self.np_random.uniform(-0.065, 0.065)
            x_dot = self.np_random.uniform(-0.065, 0.065)
            theta = math.pi + self.np_random.uniform(-0.065, 0.065)
            theta_dot = self.np_random.uniform(-0.065, 0.065)
            self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        self.steps = 0

        # Ceiling
        ceiling_y = self.area_y + 25
        self.ceiling_points = [(0, ceiling_y)]
        for i in range(12):
            x = int(i * (self.width / 11))
            y = ceiling_y + int(self.np_random.integers(0, 12))
            self.ceiling_points.append((x, y))
        self.ceiling_points.append((self.width, ceiling_y))

        # Rocks (precompute geometry)
        self.rocks = []
        for _ in range(self.np_random.integers(5, 8)):
            cx = int(self.np_random.integers(10, self.width - 10))
            cy = self.height - int(self.np_random.integers(26, 34))
            base = int(self.np_random.integers(4, 8))
            shade = int(self.np_random.integers(70, 110))
            color = (shade, max(0, shade - 25), max(0, shade - 35))
            shape_type = self.np_random.choice(["circle", "ellipse"])
            if shape_type == "circle":
                self.rocks.append({"type": "circle", "c": (cx, cy), "r": base, "color": color})
            elif shape_type == "ellipse":
                rx = int(base * self.np_random.uniform(1.2, 1.6))
                ry = int(base * self.np_random.uniform(0.8, 1.2))
                rect = (cx - rx, cy - ry, rx * 2, ry * 2)
                self.rocks.append({"type": "ellipse", "rect": rect, "color": color})

        self.lantern_phases = [float(self.np_random.uniform(0, 2 * math.pi)),
                               float(self.np_random.uniform(0, 2 * math.pi))]

        return self._get_obs(), {}

    # ---------- Step ----------
    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = float(np.clip(action[0], -1.0, 1.0)) * self.force_mag

        costheta, sintheta = math.cos(theta), math.sin(theta)
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Euler integration
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot)

        # Reward
        cart_in_bounds = abs(x) <= self.x_threshold
        angle = min(abs(theta), math.pi / 2)
        reward = (1.0 - angle / (math.pi / 2)) * float(cart_in_bounds)

        self.steps += 1
        truncated = self.steps >= self.max_episode_steps
        info = {
            'cart_in_bounds': cart_in_bounds,
            'angle': angle,
        }
        return self._get_obs(), float(reward), False, truncated, info

    def _get_obs(self):
        return np.array(self.state, dtype=np.float32)

    # ---------- Render ----------
    def render(self):
        if self._surface is None:
            pygame.init()
            self._surface = pygame.Surface((self.width, self.height))
            self.clock = pygame.time.Clock()
        if self.font is None:
            self.font = pygame.font.SysFont("Arial", 18, bold=True)

        surf = self._surface

        # Vertical gradient background (top lighter → bottom darker)
        for y in range(self.height):
            t = y / self.height
            r = int(120 - 30 * t)
            g = int(90 - 30 * t)
            b = int(50 - 15 * t)
            pygame.draw.line(surf, (r, g, b), (0, y), (self.width, y))

        # Lanterns (fixed positions; stronger flicker + noise; bracket above)
        ceiling_top = self.ceiling_points[0][1]
        lantern_positions = [(50, ceiling_top + 14), (self.width - 70, ceiling_top + 14)]
        for idx, (lx, ly) in enumerate(lantern_positions):
            pygame.draw.rect(surf, (230, 190, 90), (lx, ly, 12, 16))
            pygame.draw.rect(surf, (120, 100, 60), (lx, ly, 12, 16), 2)
            pygame.draw.rect(surf, (160, 160, 160), (lx + 4, ly - 14, 4, 14))  # bracket above

            phase = self.lantern_phases[idx]
            base = math.sin(2 * math.pi * (self.steps / 60.0) + phase)
            noise = np.random.normal(0, 0.08)  # tiny randomness each frame
            flicker = 0.9 + 0.35 * base + noise
            flicker = max(0.4, min(1.3, flicker))

            max_r = 55
            overlay = pygame.Surface((max_r * 2, max_r * 2), pygame.SRCALPHA)
            cx, cy = max_r, max_r
            base_alpha = int(90 * flicker)
            for r in range(max_r, 0, -3):
                falloff = (1.0 - r / max_r)
                alpha = int(base_alpha * (falloff ** 2))
                if alpha > 0:
                    pygame.draw.circle(overlay, (255, 210, 120, alpha), (cx, cy), r)
            surf.blit(overlay, (lx + 6 - max_r, ly + 8 - max_r))

        # Ceiling
        ceiling_extended = [(0, 0), (self.width, 0), (self.width, ceiling_top)] + list(reversed(self.ceiling_points))
        pygame.draw.polygon(surf, (85, 65, 45), ceiling_extended)
        pygame.draw.polygon(surf, (85, 65, 45), self.ceiling_points)

        # Floor (raised, clearer)
        floor_h = 32
        pygame.draw.rect(surf, (100, 75, 45), (0, self.height - floor_h, self.width, floor_h))

        # Rocks (precomputed; no frame-to-frame randomness → no flicker)
        for rock in self.rocks:
            c = rock["color"]
            if rock["type"] == "circle":
                pygame.draw.circle(surf, c, rock["c"], rock["r"])
            elif rock["type"] == "ellipse":
                pygame.draw.ellipse(surf, c, pygame.Rect(*rock["rect"]))

        # Rail (full width)
        rail_y = self.cart_y + self.cart_h // 2
        pygame.draw.line(surf, (125, 120, 120), (0, rail_y), (self.width, rail_y), 6)

        # Cart
        cart_px_x = self._cart_px_x()
        cart_rect = pygame.Rect(cart_px_x - self.cart_w // 2,
                                self.cart_y - self.cart_h // 2,
                                self.cart_w, self.cart_h)
        pygame.draw.rect(surf, (180, 110, 60), cart_rect)
        pygame.draw.rect(surf, (190, 185, 185), cart_rect, 2)

        # Wheels
        wheel_r = 5
        for offset in (-self.cart_w // 3, self.cart_w // 3):
            wx = cart_px_x + offset
            wy = self.cart_y + self.cart_h // 2
            pygame.draw.circle(surf, (200, 195, 195), (wx, wy), wheel_r)

        # Pole (bright wood)
        px_per_m = self._px_per_meter()
        pole_px_len = int(round((2 * self.length) * px_per_m))
        base_x, base_y = cart_px_x, self.cart_y - self.cart_h // 2
        tip_x = base_x + int(round(pole_px_len * math.sin(self.state[2])))
        tip_y = base_y - int(round(pole_px_len * math.cos(self.state[2])))
        pygame.draw.line(surf, (200, 140, 80), (base_x, base_y), (tip_x, tip_y), 4)

        return np.transpose(np.array(pygame.surfarray.pixels3d(surf)), (1, 0, 2)).copy()

    def close(self):
        if self._surface is not None:
            pygame.quit()
            self._surface = None


class CartpoleBalanceEnv(CartpoleEnv):
    """
    A cartpole environment where the goal is to balance the pole upright.
    """

    def __init__(self, max_episode_steps=500):
        super().__init__(max_episode_steps=max_episode_steps, task="balance")


class CartpoleSwingupEnv(CartpoleEnv):
    """
    A cartpole environment where the goal is to swing up the pole and balance it.
    """

    def __init__(self, max_episode_steps=500):
        super().__init__(max_episode_steps=max_episode_steps, task="swingup")


def sparse_reward(angle, cart_in_bounds, angle_threshold=0.15):
    # Sparse variant: reward 1 only when the pole is within angle_threshold of upright.
    reward = 1.0 if abs(angle) < angle_threshold else 0.0
    return reward * float(cart_in_bounds)


class CartpoleBalanceSparseEnv(CartpoleBalanceEnv):
    """
    A cartpole environment where the goal is to balance the pole upright with sparse rewards.
    """

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        return obs, sparse_reward(self.state[2], info['cart_in_bounds']), terminated, truncated, info


class CartpoleSwingupSparseEnv(CartpoleSwingupEnv):
    """
    A cartpole environment where the goal is to swing up the pole and balance it with sparse rewards.
    """

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        return obs, sparse_reward(self.state[2], info['cart_in_bounds']), terminated, truncated, info


class CartpoleTremorEnv(CartpoleEnv):
    """
    A cartpole environment with random 'mine tremors'. At random intervals,
    the scene shakes and a destabilizing force is applied to the cart & pole.
    """

    def __init__(self, max_episode_steps=500, task="balance"):
        super().__init__(max_episode_steps=max_episode_steps, task=task)

        # Shake parameters
        self.shake_timer = 0       # frames remaining in current shake
        self.shake_force = 0.0     # force applied during shake
        self.next_shake = 0        # countdown to next shake
        self.dust_particles = []   # active dust particles

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.shake_timer = 0
        self.shake_force = 0.0
        self.next_shake = int(self.np_random.integers(40, 81))
        self.dust_particles = []
        return obs, info

    def step(self, action):
        # Handle shake scheduling
        if self.shake_timer > 0:
            self.shake_timer -= 1
            pole_shake = self.shake_force

            # Spawn dust particles
            if self.np_random.random() < 0.3:  # spawn a few per frame
                x = int(self.np_random.integers(20, self.width - 20))
                self.dust_particles.append({
                    "x": x,
                    "y": self.ceiling_points[0][1] + 5,
                    "vy": self.np_random.uniform(1.5, 4.0),
                    "life": int(self.np_random.integers(18, 40)),
                    "radius": int(self.np_random.integers(2, 4))
                })

        else:
            pole_shake = 0.0
            self.next_shake -= 1
            if self.next_shake <= 0:
                # Start new shake
                self.shake_timer = int(self.np_random.integers(5, 11))
                self.shake_force_dir = self.np_random.choice([-1, 1])
                self.shake_force = float(self.np_random.uniform(0.35, 1.0) * 5.0 * self.shake_force_dir)
                self.next_shake = int(self.np_random.integers(50, 91))

        # Update dust particles
        for p in self.dust_particles:
            p["y"] += p["vy"]
            p["life"] -= 1
        self.dust_particles = [p for p in self.dust_particles if p["life"] > 0 and p["y"] < self.height-10]

        # Physics (same as base, but shake only in thetaacc)
        x, x_dot, theta, theta_dot = self.state
        agent_force = float(np.clip(action[0], -1.0, 1.0)) * self.force_mag
        total_force = agent_force

        costheta, sintheta = math.cos(theta), math.sin(theta)
        temp = (total_force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )

        # Inject shake directly into pole angular acceleration
        thetaacc += pole_shake

        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Euler integration
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot)

        # Reward
        cart_in_bounds = abs(x) <= self.x_threshold
        angle = min(abs(theta), math.pi / 2)
        reward = (1.0 - angle / (math.pi / 2)) * float(cart_in_bounds)

        self.steps += 1
        truncated = self.steps >= self.max_episode_steps
        info = {
            'cart_in_bounds': cart_in_bounds,
            'angle': angle,
        }
        return self._get_obs(), float(reward), False, truncated, info

    def render(self):
        # Render the base scene
        frame = super().render()

        # Convert numpy frame to pygame surface with alpha support
        surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.surfarray.blit_array(surf, np.transpose(frame, (1, 0, 2)))

        # Draw dust particles
        if self.dust_particles:
            for p in self.dust_particles:
                alpha = max(30, int(120 * (p["life"] / 35.0)))
                color = (150, 120, 90, alpha)
                pygame.draw.circle(surf, color, (int(p["x"]), int(p["y"])), p["radius"])

        frame = np.transpose(np.array(pygame.surfarray.pixels3d(surf)), (1, 0, 2))

        # Apply camera shake with guaranteed coverage
        if self.shake_timer > 0:
            max_offset = 5   # maximum pixel jitter
            pad = max_offset * 4
            h, w, _ = frame.shape

            # Fill canvas with background brown instead of black
            bg_color = np.array([100, 75, 45], dtype=frame.dtype)  # same as floor base color
            canvas = np.tile(bg_color, (h + 2*pad, w + 2*pad, 1))

            # Paste the rendered frame into the center
            canvas[pad:pad+h, pad:pad+w] = frame

            # Pick jitter offset
            magnitude = min(max_offset, int(abs(self.shake_force) / self.force_mag * max_offset))
            dx = int(self.np_random.integers(-magnitude, magnitude+1))
            dy = int(self.np_random.integers(-magnitude, magnitude+1))

            # Crop shifted region
            cx, cy = pad + w//2 + dx, pad + h//2 + dy
            x0, y0 = cx - w//2, cy - h//2
            x1, y1 = x0 + w, y0 + h

            cropped = canvas[y0:y1, x0:x1]
            return cropped.copy()

        return frame
