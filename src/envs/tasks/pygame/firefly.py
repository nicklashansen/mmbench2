import gymnasium as gym
import numpy as np
import pygame
import math


class FireflyEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, max_episode_steps=500, n_fireflies=6):
        super().__init__()
        self.max_episode_steps = max_episode_steps

        # canvas / play area
        self.width, self.height = 224, 224
        self.area_margin = 16
        self.area_x = self.area_margin
        self.area_y = self.area_margin
        self.area_w = self.width - 2 * self.area_margin
        self.area_h = self.height - 2 * self.area_margin

        self._scale = 1.3  # global size multiplier for agent + fireflies

        # agent dynamics
        self.agent_r = int(round(5 * self._scale))
        self.accel_scale = 0.9
        self.friction = 0.10
        self.max_speed = 3.8

        # firefly dynamics
        self.n_fireflies = n_fireflies
        self.firefly_r = float(2.0 * self._scale)  # used for drawing + boundary clamp
        self.firefly_max_speed = 2.4
        self.firefly_noise = 0.7
        self.firefly_repel_radius = 40.0
        self.firefly_repel_strength = 0.09

        # capture / reward
        self.capture_radius = float(8.0 * self._scale)
        self.step_cost = -0.001
        self.capture_reward = 1.0
        self.dist_shaping_scale = 0.02

        # runtime state
        self.ax = self.ay = 0.0
        self.vx = self.vy = 0.0
        self.fireflies_pos = np.zeros((self.n_fireflies, 2), dtype=np.float32)
        self.fireflies_vel = np.zeros((self.n_fireflies, 2), dtype=np.float32)
        self.steps = 0
        self.total_captures = 0
        self.steps_since_last_capture = 0

        # sparkle bursts on catches
        self._sparkles = []  # list of dicts

        # observation: 4 + n*(4) + 2
        obs_dim = 4 + self.n_fireflies * 4 + 2
        assert obs_dim <= 128
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
            low=-np.ones(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )

        # visuals: moonlit forest clearing
        self._bg_color = (6, 8, 20)
        self._bg_star_color = (25, 30, 55)
        self._ground_color = (12, 18, 35)
        self._ground_ring_color = (20, 26, 48)

        # agent visuals
        self._agent_core = (210, 230, 255)
        self._agent_outline = (120, 170, 255)
        self._agent_trail = (140, 180, 255)

        # firefly / sparkle colors
        self._firefly_core = (255, 246, 210)     # warm core
        self._firefly_glow = (255, 210, 120)     # amber glow
        self._sparkle_color = (255, 215, 140)    # sparkle

        # pygame
        self._surface = None
        self._font = None

        # pre-sampled background "stars"
        self._stars = [
            (
                np.random.randint(0, self.width),
                np.random.randint(0, self.height),
                np.random.randint(1, 3),
            )
            for _ in range(40)
        ]

    # ---------- helpers ----------

    def _random_pos_in_area(self):
        x = self.np_random.uniform(
            self.area_x + self.agent_r, self.area_x + self.area_w - self.agent_r
        )
        y = self.np_random.uniform(
            self.area_y + self.agent_r, self.area_y + self.area_h - self.agent_r
        )
        return x, y

    def _clamp_to_area(self, x, y, radius):
        x = float(np.clip(x, self.area_x + radius, self.area_x + self.area_w - radius))
        y = float(np.clip(y, self.area_y + radius, self.area_y + self.area_h - radius))
        return x, y

    def _get_obs(self):
        ax_n = (self.ax - self.area_x) / self.area_w
        ay_n = (self.ay - self.area_y) / self.area_h
        vx_n = np.clip(self.vx / self.max_speed, -1.0, 1.0)
        vy_n = np.clip(self.vy / self.max_speed, -1.0, 1.0)

        ff_feats = []
        for i in range(self.n_fireflies):
            fx, fy = self.fireflies_pos[i]
            fvx, fvy = self.fireflies_vel[i]

            dx = np.clip((fx - self.ax) / self.area_w, -1.0, 1.0)
            dy = np.clip((fy - self.ay) / self.area_h, -1.0, 1.0)
            fvx_n = np.clip(fvx / self.firefly_max_speed, -1.0, 1.0)
            fvy_n = np.clip(fvy / self.firefly_max_speed, -1.0, 1.0)
            ff_feats.extend([float(dx), float(dy), float(fvx_n), float(fvy_n)])

        step_frac = float(self.steps) / float(self.max_episode_steps)
        last_cap_norm = float(
            np.clip(self.steps_since_last_capture / float(self.max_episode_steps), 0.0, 1.0)
        )

        return np.array([ax_n, ay_n, vx_n, vy_n] + ff_feats + [step_frac, last_cap_norm], dtype=np.float32)

    def _respawn_firefly(self, idx):
        x, y = self._random_pos_in_area()
        self.fireflies_pos[idx, 0] = x
        self.fireflies_pos[idx, 1] = y
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(0.5, self.firefly_max_speed)
        self.fireflies_vel[idx, 0] = math.cos(angle) * speed * 0.5
        self.fireflies_vel[idx, 1] = math.sin(angle) * speed * 0.5

    # -----------------------------
    # Sparkle burst on catch
    # -----------------------------
    def _spawn_catch_sparkles(self, x, y):
        # small explosion of warm sparkles
        n = 14
        life = 18
        for i in range(n):
            ang = float(self.np_random.uniform(0, 2 * math.pi))
            spd = float(self.np_random.uniform(0.6, 2.2))
            self._sparkles.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "vx": math.cos(ang) * spd,
                    "vy": math.sin(ang) * spd,
                    "start": int(self.steps),
                    "life": int(life),
                    "r0": float(self.np_random.uniform(1.5, 2.8) * self._scale),
                }
            )

    # ---------- gym API ----------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.ax = self.area_x + self.area_w / 2.0
        self.ay = self.area_y + self.area_h / 2.0
        self.vx = 0.0
        self.vy = 0.0

        self.steps = 0
        self.total_captures = 0
        self.steps_since_last_capture = 0
        self._sparkles.clear()

        for i in range(self.n_fireflies):
            self._respawn_firefly(i)

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # --- update agent ---
        vx = self.vx + float(action[0]) * self.accel_scale
        vy = self.vy + float(action[1]) * self.accel_scale

        vx *= (1.0 - self.friction)
        vy *= (1.0 - self.friction)

        speed = math.hypot(vx, vy)
        if speed > self.max_speed:
            s = self.max_speed / (speed + 1e-8)
            vx *= s
            vy *= s

        ax = self.ax + vx
        ay = self.ay + vy
        ax, ay = self._clamp_to_area(ax, ay, self.agent_r)

        self.ax, self.ay = ax, ay
        self.vx, self.vy = vx, vy

        # --- update fireflies ---
        pos = self.fireflies_pos
        vel = self.fireflies_vel

        for i in range(self.n_fireflies):
            fx, fy = pos[i]
            fvx, fvy = vel[i]

            noise_angle = self.np_random.uniform(0, 2 * math.pi)
            fvx += math.cos(noise_angle) * self.firefly_noise
            fvy += math.sin(noise_angle) * self.firefly_noise

            dx = fx - ax
            dy = fy - ay
            dist = math.hypot(dx, dy) + 1e-8
            if dist < self.firefly_repel_radius:
                repel = self.firefly_repel_strength * (1.0 - dist / self.firefly_repel_radius)
                fvx += (dx / dist) * repel * self.firefly_repel_radius
                fvy += (dy / dist) * repel * self.firefly_repel_radius

            fs = math.hypot(fvx, fvy)
            if fs > self.firefly_max_speed:
                s = self.firefly_max_speed / (fs + 1e-8)
                fvx *= s
                fvy *= s

            fx += fvx
            fy += fvy
            fx, fy = self._clamp_to_area(fx, fy, radius=self.firefly_r)

            pos[i, 0] = fx
            pos[i, 1] = fy
            vel[i, 0] = fvx
            vel[i, 1] = fvy

        # --- reward + captures ---
        reward = self.step_cost
        n_caught = 0
        min_dist = float("inf")

        for i in range(self.n_fireflies):
            dx = pos[i, 0] - ax
            dy = pos[i, 1] - ay
            d = math.hypot(dx, dy)
            min_dist = min(min_dist, d)

            if d <= self.capture_radius:
                # sparkle at catch position BEFORE respawn
                self._spawn_catch_sparkles(pos[i, 0], pos[i, 1])

                reward += self.capture_reward
                self.total_captures += 1
                self.steps_since_last_capture = 0
                n_caught += 1
                self._respawn_firefly(i)

        if min_dist == float("inf"):
            min_dist = 0.0

        max_d = math.hypot(self.area_w, self.area_h)
        dist_norm = float(np.clip(min_dist / (max_d / 2), 0.0, 1.0))
        reward += self.dist_shaping_scale * (1.0 - dist_norm)

        if n_caught == 0:
            self.steps_since_last_capture += 1

        self.steps += 1
        terminated = False
        truncated = self.steps >= self.max_episode_steps

        info = {
            "n_caught": n_caught,
            "total_captures": self.total_captures,
            "min_dist_to_firefly": min_dist,
            "dist_shaping": self.dist_shaping_scale * (1.0 - dist_norm),
            "steps_since_last_capture": self.steps_since_last_capture,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    # ---------- rendering ----------

    def _hud_alpha_for_rect(self, rect):
        """
        Make HUD more transparent if agent or any firefly overlaps the HUD region.
        This keeps objects behind readable.
        """
        pad = 6
        r = rect.inflate(pad * 2, pad * 2)

        # agent overlap
        if r.collidepoint(int(self.ax), int(self.ay)):
            return 100

        # firefly overlap
        for i in range(self.n_fireflies):
            fx, fy = self.fireflies_pos[i]
            if r.collidepoint(int(fx), int(fy)):
                return 100

        return 200

    def render(self):
        if self._surface is None:
            pygame.init()
            self._surface = pygame.Surface((self.width, self.height))
        if self._font is None:
            pygame.font.init()
            self._font = pygame.font.SysFont("arial", 16, bold=True)

        surf = self._surface
        surf.fill(self._bg_color)

        for (sx, sy, rad) in self._stars:
            pygame.draw.circle(surf, self._bg_star_color, (sx, sy), rad)

        ground_rect = pygame.Rect(self.area_x, self.area_y, self.area_w, self.area_h)
        pygame.draw.rect(surf, self._ground_color, ground_rect, border_radius=18)

        ring_margin = 6
        inner_rect = ground_rect.inflate(-ring_margin, -ring_margin)
        pygame.draw.rect(surf, self._ground_ring_color, inner_rect, border_radius=16, width=2)

        phase1 = math.sin(0.12 * self.steps)
        phase2 = math.sin(0.37 * self.steps + 1.1)
        flicker = 1.0 + 0.2 * phase1 + 0.1 * phase2
        flicker = max(0.7, min(1.3, flicker))

        # --- sparkles (draw first so fireflies sit on top) ---
        if self._sparkles:
            new_s = []
            for sp in self._sparkles:
                age = self.steps - sp["start"]
                if age < 0 or age > sp["life"]:
                    continue

                t = age / float(sp["life"])  # 0..1
                # integrate (simple)
                x = sp["x"] + sp["vx"] * age
                y = sp["y"] + sp["vy"] * age

                alpha = int(220 * (1.0 - t))
                if alpha <= 0:
                    continue

                rr = max(1, int(sp["r0"] * (1.0 - 0.6 * t)))
                s_surf = pygame.Surface((2 * rr + 2, 2 * rr + 2), pygame.SRCALPHA)
                pygame.draw.circle(s_surf, (*self._sparkle_color, alpha), (rr + 1, rr + 1), rr)
                surf.blit(s_surf, (int(x) - rr - 1, int(y) - rr - 1))
                new_s.append(sp)
            self._sparkles = new_s

        # --- fireflies ---
        for i in range(self.n_fireflies):
            fx, fy = self.fireflies_pos[i]

            glow_radius = int(round(10 * self._scale))
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)

            base_alpha = 95
            for yy in range(glow_radius * 2):
                dy = yy - glow_radius
                for xx in range(glow_radius * 2):
                    dx = xx - glow_radius
                    dist = math.hypot(dx, dy) / glow_radius
                    if dist > 1.0:
                        continue
                    falloff = (1.0 - dist) ** 2
                    alpha = int(base_alpha * falloff * flicker)
                    if alpha <= 0:
                        continue
                    glow_surf.set_at((xx, yy), (*self._firefly_glow, alpha))

            surf.blit(glow_surf, (int(fx) - glow_radius, int(fy) - glow_radius))

            core_r = max(2, int(round(2 * self._scale)))
            pygame.draw.circle(surf, self._firefly_core, (int(fx), int(fy)), core_r)

        # --- agent trail + agent ---
        trail_len = 4
        speed = math.hypot(self.vx, self.vy)
        if speed > 0.05:
            dir_x = -self.vx / (speed + 1e-8)
            dir_y = -self.vy / (speed + 1e-8)
            for k in range(1, trail_len + 1):
                t = k / float(trail_len + 1)
                tx = self.ax + dir_x * 4 * k
                ty = self.ay + dir_y * 4 * k
                alpha = int(140 * (1.0 - t))
                rr = max(1, int(self.agent_r * (1.0 - 0.4 * t)))
                trail_surf = pygame.Surface((2 * rr, 2 * rr), pygame.SRCALPHA)
                pygame.draw.circle(trail_surf, (*self._agent_trail, alpha), (rr, rr), rr)
                surf.blit(trail_surf, (int(tx) - rr, int(ty) - rr))

        ax_i, ay_i = int(self.ax), int(self.ay)
        pygame.draw.circle(surf, self._agent_core, (ax_i, ay_i), self.agent_r)
        pygame.draw.circle(surf, self._agent_outline, (ax_i, ay_i), self.agent_r, 1)

        glow_radius_agent = 16
        glow_surf_agent = pygame.Surface((glow_radius_agent * 2, glow_radius_agent * 2), pygame.SRCALPHA)
        base_alpha_agent = 60
        for yy in range(glow_radius_agent * 2):
            dy = yy - glow_radius_agent
            for xx in range(glow_radius_agent * 2):
                dx = xx - glow_radius_agent
                dist = math.hypot(dx, dy) / glow_radius_agent
                if dist > 1.0:
                    continue
                falloff = (1.0 - dist) ** 2
                alpha = int(base_alpha_agent * falloff * flicker)
                if alpha <= 0:
                    continue
                glow_surf_agent.set_at((xx, yy), (*self._agent_outline, alpha))
        surf.blit(glow_surf_agent, (ax_i - glow_radius_agent, ay_i - glow_radius_agent))

        # --- HUD with "auto transparency" if something is behind it ---
        text = f"Score: {self.total_captures}"
        text_surf = self._font.render(text, True, (220, 230, 255))
        shadow_surf = self._font.render(text, True, (10, 10, 20))
        rect = text_surf.get_rect(topleft=(8, 6))

        alpha = self._hud_alpha_for_rect(rect)

        hud = pygame.Surface((rect.width + 6, rect.height + 4), pygame.SRCALPHA)
        # soft backing (also alpha-adjusted)
        backing_alpha = int(0.35 * alpha)
        pygame.draw.rect(hud, (10, 12, 24, backing_alpha), hud.get_rect(), border_radius=6)

        # shadow + text
        shadow_tmp = shadow_surf.copy()
        text_tmp = text_surf.copy()
        shadow_tmp.set_alpha(alpha)
        text_tmp.set_alpha(alpha)
        hud.blit(shadow_tmp, (4, 3))
        hud.blit(text_tmp, (3, 2))

        surf.blit(hud, (rect.x - 3, rect.y - 2))

        arr = np.transpose(pygame.surfarray.pixels3d(surf), (1, 0, 2)).copy()
        return arr

    def close(self):
        if self._surface is not None:
            pygame.quit()
            self._surface = None
