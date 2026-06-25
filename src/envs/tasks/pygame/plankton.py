import gymnasium as gym
import numpy as np
import pygame
import math


class PlanktonEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, max_episode_steps=500, n_motes=6):
        super().__init__()
        self.max_episode_steps = int(max_episode_steps)

        # canvas / play area
        self.width, self.height = 224, 224
        self.area_margin = 14
        self.area_x = self.area_margin
        self.area_y = self.area_margin
        self.area_w = self.width - 2 * self.area_margin
        self.area_h = self.height - 2 * self.area_margin

        # fast-paced base (global +50% speed multiplier)
        self._speed_mult = 1.5

        # ---------- agent (mass growth) ----------
        self.agent_r0 = 6.0
        self.agent_r = float(self.agent_r0)

        self.max_speed0 = 3.0 * self._speed_mult
        self.max_speed_min = 1.6 * self._speed_mult
        self.max_speed = float(self.max_speed0)

        self.accel_scale0 = 0.90 * self._speed_mult
        self.accel_scale = float(self.accel_scale0)
        self.friction = 0.11

        self.r_gain = 1.25
        self.speed_decay = 0.92
        self.accel_decay = 0.965

        # ---------- small plankton ----------
        self.n_motes = int(n_motes)
        self.mote_r = 3.0
        self.mote_max_speed = 2.2 * self._speed_mult
        self.mote_noise = 0.75 * self._speed_mult
        self.mote_repel_radius = 46.0
        self.mote_repel_strength = 0.085

        # ---------- big plankton ----------
        self.big_r0 = 11.0
        self.big_r = float(self.big_r0)

        # big plankton deliberately move at half speed
        self.big_max_speed = 2.3 * self._speed_mult * 0.5
        self.big_noise = 0.75 * self._speed_mult * 0.5
        self.big_repel_radius = 54.0
        self.big_repel_strength = 0.090

        # stun
        self.stun_steps = 18
        self._stun = 0
        self.stun_step_penalty = 0.02  # reward penalty per step while stunned
        self._shake_phase = 0.0

        # score/reward
        self.step_cost = -0.001
        self.score = 0
        self.small_score = 1
        self.big_score = 5

        # distance shaping (reduced)
        self.dist_shaping_scale = 0.02

        # capture radii
        self.capture_radius_base = 10.0
        self.big_contact_extra = 1.0

        # runtime state
        self.ax = self.ay = 0.0
        self.vx = self.vy = 0.0
        self.motes_pos = np.zeros((self.n_motes, 2), dtype=np.float32)
        self.motes_vel = np.zeros((self.n_motes, 2), dtype=np.float32)
        self.big_pos = np.zeros((2,), dtype=np.float32)
        self.big_vel = np.zeros((2,), dtype=np.float32)
        self.steps = 0
        self.steps_since_last_score = 0

        # visuals state
        self._sparkles = []
        self._bubbles = []
        self._bg_specks = []

        # observation
        # agent: 6
        # small: 4n
        # big: 4
        # extras: step_frac + last_score + stun_norm = 3
        obs_dim = 6 + 4 * self.n_motes + 4 + 3
        assert obs_dim <= 128
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-np.ones(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )

        # ---------- visuals ----------
        self._bg_top = (3, 12, 24)
        self._bg_bottom = (2, 28, 40)
        self._ground_ring = (8, 60, 70)
        self._water_haze = (60, 160, 170)

        self._agent_core = (195, 230, 255)
        self._agent_outline = (110, 190, 255)
        self._agent_trail = (140, 210, 255)

        self._mote_core = (235, 255, 250)
        self._mote_glow = (110, 255, 220)
        self._sparkle_color = (160, 255, 235)

        self._big_core = (235, 255, 235)
        self._big_glow = (80, 235, 200)
        self._big_sparkle = (140, 255, 220)

        self._hud_text = (210, 245, 255)
        self._hud_shadow = (0, 10, 20)

        self._surface = None
        self._font = None

    # ---------- helpers ----------

    def _clamp_to_area(self, x, y, radius):
        x = float(np.clip(x, self.area_x + radius, self.area_x + self.area_w - radius))
        y = float(np.clip(y, self.area_y + radius, self.area_y + self.area_h - radius))
        return x, y

    def _random_pos(self, radius):
        x = self.np_random.uniform(self.area_x + radius, self.area_x + self.area_w - radius)
        y = self.np_random.uniform(self.area_y + radius, self.area_y + self.area_h - radius)
        return float(x), float(y)

    def _respawn_mote(self, idx):
        x, y = self._random_pos(self.mote_r)
        self.motes_pos[idx, 0] = x
        self.motes_pos[idx, 1] = y
        ang = float(self.np_random.uniform(0, 2 * math.pi))
        spd = float(self.np_random.uniform(0.5, self.mote_max_speed))
        self.motes_vel[idx, 0] = math.cos(ang) * spd * 0.55
        self.motes_vel[idx, 1] = math.sin(ang) * spd * 0.55

    def _corner_spawn(self, radius):
        corners = [
            (self.area_x + radius, self.area_y + radius),
            (self.area_x + self.area_w - radius, self.area_y + radius),
            (self.area_x + radius, self.area_y + self.area_h - radius),
            (self.area_x + self.area_w - radius, self.area_y + self.area_h - radius),
        ]
        return corners[int(self.np_random.integers(0, 4))]

    def _respawn_big_in_corner(self):
        x, y = self._corner_spawn(self.big_r)
        self.big_pos[0], self.big_pos[1] = float(x), float(y)
        ang = float(self.np_random.uniform(0, 2 * math.pi))
        spd = float(self.np_random.uniform(0.6, self.big_max_speed))
        self.big_vel[0] = math.cos(ang) * spd * 0.55
        self.big_vel[1] = math.sin(ang) * spd * 0.55

    def _spawn_sparkles(self, x, y, color, n=16, life=20, r0_lo=1.8, r0_hi=3.2, spd_lo=0.7, spd_hi=2.6):
        for _ in range(n):
            ang = float(self.np_random.uniform(0, 2 * math.pi))
            spd = float(self.np_random.uniform(spd_lo, spd_hi))
            self._sparkles.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "vx": math.cos(ang) * spd,
                    "vy": math.sin(ang) * spd,
                    "start": int(self.steps),
                    "life": int(life),
                    "r0": float(self.np_random.uniform(r0_lo, r0_hi)),
                    "color": color,
                }
            )

    def _init_background_particles(self):
        self._bg_specks = []
        for _ in range(70):
            self._bg_specks.append(
                {
                    "x": float(self.np_random.uniform(0, self.width)),
                    "y": float(self.np_random.uniform(0, self.height)),
                    "r": float(self.np_random.uniform(0.8, 1.8)),
                    "vy": float(self.np_random.uniform(0.10, 0.32)),
                }
            )

        self._bubbles = []
        for _ in range(10):
            self._bubbles.append(
                {
                    "x": float(self.np_random.uniform(self.area_x, self.area_x + self.area_w)),
                    "y": float(self.np_random.uniform(self.area_y, self.area_y + self.area_h)),
                    "r": float(self.np_random.uniform(2.0, 5.5)),
                    "vy": float(self.np_random.uniform(0.20, 0.60)),
                    "phase": float(self.np_random.uniform(0, 2 * math.pi)),
                }
            )

    def _apply_mass_growth(self):
        self.agent_r = float(self.agent_r + self.r_gain)
        self.max_speed = float(max(self.max_speed_min, self.max_speed * self.speed_decay))
        self.accel_scale = float(max(0.35, self.accel_scale * self.accel_decay))
        self.ax, self.ay = self._clamp_to_area(self.ax, self.ay, self.agent_r)

        s = math.hypot(self.vx, self.vy)
        if s > self.max_speed:
            k = self.max_speed / (s + 1e-8)
            self.vx *= k
            self.vy *= k

    def _get_obs(self):
        ax_n = (self.ax - self.area_x) / self.area_w
        ay_n = (self.ay - self.area_y) / self.area_h

        vx_n = float(np.clip(self.vx / max(1e-6, self.max_speed0), -1.0, 1.0))
        vy_n = float(np.clip(self.vy / max(1e-6, self.max_speed0), -1.0, 1.0))

        size_norm = float(np.clip((self.agent_r - self.agent_r0) / 18.0, 0.0, 1.0))
        speed_norm = float(
            np.clip(
                (self.max_speed - self.max_speed_min) / (self.max_speed0 - self.max_speed_min + 1e-8),
                0.0,
                1.0,
            )
        )

        feats = [float(ax_n), float(ay_n), vx_n, vy_n, size_norm, speed_norm]

        for i in range(self.n_motes):
            mx, my = self.motes_pos[i]
            mvx, mvy = self.motes_vel[i]
            dx = float(np.clip((mx - self.ax) / self.area_w, -1.0, 1.0))
            dy = float(np.clip((my - self.ay) / self.area_h, -1.0, 1.0))
            mvx_n = float(np.clip(mvx / self.mote_max_speed, -1.0, 1.0))
            mvy_n = float(np.clip(mvy / self.mote_max_speed, -1.0, 1.0))
            feats.extend([dx, dy, mvx_n, mvy_n])

        bx, by = float(self.big_pos[0]), float(self.big_pos[1])
        bvx, bvy = float(self.big_vel[0]), float(self.big_vel[1])
        bdx = float(np.clip((bx - self.ax) / self.area_w, -1.0, 1.0))
        bdy = float(np.clip((by - self.ay) / self.area_h, -1.0, 1.0))
        bvx_n = float(np.clip(bvx / self.big_max_speed, -1.0, 1.0))
        bvy_n = float(np.clip(bvy / self.big_max_speed, -1.0, 1.0))
        feats.extend([bdx, bdy, bvx_n, bvy_n])

        step_frac = float(self.steps) / float(self.max_episode_steps)
        last_score = float(np.clip(self.steps_since_last_score / float(self.max_episode_steps), 0.0, 1.0))
        stun_norm = float(np.clip(self._stun / float(self.stun_steps), 0.0, 1.0))
        feats.extend([step_frac, last_score, stun_norm])

        return np.array(feats, dtype=np.float32)

    def _hud_alpha_for_rect(self, rect):
        pad = 6
        r = rect.inflate(pad * 2, pad * 2)
        if r.collidepoint(int(self.ax), int(self.ay)):
            return 100
        for i in range(self.n_motes):
            mx, my = self.motes_pos[i]
            if r.collidepoint(int(mx), int(my)):
                return 100
        if r.collidepoint(int(self.big_pos[0]), int(self.big_pos[1])):
            return 100
        return 200

    # ---------- gym API ----------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.steps_since_last_score = 0
        self._sparkles.clear()
        self._stun = 0
        self._shake_phase = 0.0

        self.agent_r = float(self.agent_r0)
        self.max_speed = float(self.max_speed0)
        self.accel_scale = float(self.accel_scale0)

        self.ax = self.area_x + self.area_w / 2.0
        self.ay = self.area_y + self.area_h / 2.0
        self.vx = 0.0
        self.vy = 0.0

        for i in range(self.n_motes):
            self._respawn_mote(i)

        self._respawn_big_in_corner()
        self._init_background_particles()

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        stunned_this_step = (self._stun > 0)

        # --- update agent (or stunned) ---
        if self._stun > 0:
            self._stun -= 1
            self.vx *= 0.80
            self.vy *= 0.80
        else:
            self.vx = self.vx + float(action[0]) * self.accel_scale
            self.vy = self.vy + float(action[1]) * self.accel_scale
            self.vx *= (1.0 - self.friction)
            self.vy *= (1.0 - self.friction)

        sp = math.hypot(self.vx, self.vy)
        if sp > self.max_speed:
            k = self.max_speed / (sp + 1e-8)
            self.vx *= k
            self.vy *= k

        self.ax = self.ax + self.vx
        self.ay = self.ay + self.vy
        self.ax, self.ay = self._clamp_to_area(self.ax, self.ay, self.agent_r)

        # --- update small motes ---
        for i in range(self.n_motes):
            mx, my = self.motes_pos[i]
            mvx, mvy = self.motes_vel[i]

            ang = float(self.np_random.uniform(0, 2 * math.pi))
            mvx += math.cos(ang) * self.mote_noise
            mvy += math.sin(ang) * self.mote_noise

            dx = mx - self.ax
            dy = my - self.ay
            d = math.hypot(dx, dy) + 1e-8
            if d < self.mote_repel_radius:
                repel = self.mote_repel_strength * (1.0 - d / self.mote_repel_radius)
                mvx += (dx / d) * repel * self.mote_repel_radius
                mvy += (dy / d) * repel * self.mote_repel_radius

            ms = math.hypot(mvx, mvy)
            if ms > self.mote_max_speed:
                k = self.mote_max_speed / (ms + 1e-8)
                mvx *= k
                mvy *= k

            mx += mvx
            my += mvy
            mx, my = self._clamp_to_area(mx, my, self.mote_r)

            self.motes_pos[i, 0] = mx
            self.motes_pos[i, 1] = my
            self.motes_vel[i, 0] = mvx
            self.motes_vel[i, 1] = mvy

        # --- update big mote ---
        bx, by = float(self.big_pos[0]), float(self.big_pos[1])
        bvx, bvy = float(self.big_vel[0]), float(self.big_vel[1])

        ang = float(self.np_random.uniform(0, 2 * math.pi))
        bvx += math.cos(ang) * self.big_noise
        bvy += math.sin(ang) * self.big_noise

        dx = bx - self.ax
        dy = by - self.ay
        d = math.hypot(dx, dy) + 1e-8
        if d < self.big_repel_radius:
            repel = self.big_repel_strength * (1.0 - d / self.big_repel_radius)
            bvx += (dx / d) * repel * self.big_repel_radius
            bvy += (dy / d) * repel * self.big_repel_radius

        bs = math.hypot(bvx, bvy)
        if bs > self.big_max_speed:
            k = self.big_max_speed / (bs + 1e-8)
            bvx *= k
            bvy *= k

        bx += bvx
        by += bvy
        bx, by = self._clamp_to_area(bx, by, self.big_r)
        self.big_pos[0], self.big_pos[1] = float(bx), float(by)
        self.big_vel[0], self.big_vel[1] = float(bvx), float(bvy)

        # --- reward/score ---
        reward = float(self.step_cost)
        if stunned_this_step:
            reward -= float(self.stun_step_penalty)

        scored_this_step = 0

        capture_radius = float(self.capture_radius_base + 0.35 * (self.agent_r - self.agent_r0))

        # distance shaping to nearest small mote
        min_dist = float("inf")
        for i in range(self.n_motes):
            mx, my = self.motes_pos[i]
            dd = math.hypot(mx - self.ax, my - self.ay)
            min_dist = min(min_dist, dd)

        # IMPORTANT: no small catches while stunned
        if not stunned_this_step:
            for i in range(self.n_motes):
                mx, my = self.motes_pos[i]
                dd = math.hypot(mx - self.ax, my - self.ay)
                if dd <= capture_radius:
                    self._spawn_sparkles(mx, my, self._sparkle_color, n=16, life=20)
                    self.score += self.small_score
                    scored_this_step += self.small_score
                    reward += 1.0
                    self.steps_since_last_score = 0
                    self._apply_mass_growth()
                    self._respawn_mote(i)

        # big interaction
        big_contact_r = float(self.agent_r + self.big_r + self.big_contact_extra)
        big_d = math.hypot(float(self.big_pos[0]) - self.ax, float(self.big_pos[1]) - self.ay)

        if big_d <= big_contact_r:
            if self.agent_r > self.big_r:
                self._spawn_sparkles(
                    float(self.big_pos[0]),
                    float(self.big_pos[1]),
                    self._big_sparkle,
                    n=26,
                    life=26,
                    r0_lo=2.4,
                    r0_hi=4.5,
                )
                self.score += self.big_score
                scored_this_step += self.big_score
                reward += 5.0
                self.steps_since_last_score = 0
                self._apply_mass_growth()
                self._respawn_big_in_corner()
            else:
                self._stun = max(self._stun, self.stun_steps)

        if min_dist == float("inf"):
            min_dist = 0.0
        max_d = math.hypot(self.area_w, self.area_h)
        dist_norm = float(np.clip(min_dist / (max_d / 2), 0.0, 1.0))
        reward += float(self.dist_shaping_scale) * (1.0 - dist_norm)

        if scored_this_step == 0:
            self.steps_since_last_score += 1

        self.steps += 1
        terminated = False
        truncated = self.steps >= self.max_episode_steps

        info = {
            "score": int(self.score),
            "scored_this_step": int(scored_this_step),
            "stun": int(self._stun),
            "agent_r": float(self.agent_r),
            "max_speed": float(self.max_speed),
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    # ---------- rendering ----------

    def _draw_vertical_gradient(self, surf, top_rgb, bottom_rgb):
        for y in range(self.height):
            t = y / float(self.height - 1)
            r = int(top_rgb[0] * (1.0 - t) + bottom_rgb[0] * t)
            g = int(top_rgb[1] * (1.0 - t) + bottom_rgb[1] * t)
            b = int(top_rgb[2] * (1.0 - t) + bottom_rgb[2] * t)
            pygame.draw.line(surf, (r, g, b), (0, y), (self.width, y))

    def render(self):
        if self._surface is None:
            pygame.init()
            self._surface = pygame.Surface((self.width, self.height))
        if self._font is None:
            pygame.font.init()
            self._font = pygame.font.SysFont("arial", 16, bold=True)

        surf = self._surface

        # gentle shake when stunned
        shake_x = 0
        shake_y = 0
        if self._stun > 0:
            self._shake_phase += 0.55
            amp = 2.0 * (self._stun / float(self.stun_steps))
            shake_x = int(round(math.sin(self._shake_phase) * amp))
            shake_y = int(round(math.cos(self._shake_phase * 0.9) * amp))
        else:
            self._shake_phase *= 0.9

        world = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        self._draw_vertical_gradient(world, self._bg_top, self._bg_bottom)

        for sp in self._bg_specks:
            sp["y"] += sp["vy"]
            if sp["y"] > self.height + 2:
                sp["y"] = -2
                sp["x"] = float(self.np_random.uniform(0, self.width))
            a = 35
            speck = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(speck, (120, 220, 230, a), (2, 2), int(sp["r"]))
            world.blit(speck, (int(sp["x"]), int(sp["y"])))

        area_rect = pygame.Rect(self.area_x, self.area_y, self.area_w, self.area_h)
        pygame.draw.rect(world, (2, 20, 35), area_rect, border_radius=18)
        pygame.draw.rect(world, self._ground_ring, area_rect, width=2, border_radius=18)

        phase1 = math.sin(0.12 * self.steps)
        phase2 = math.sin(0.41 * self.steps + 0.8)
        flicker = 1.0 + 0.22 * phase1 + 0.12 * phase2
        flicker = max(0.7, min(1.35, flicker))

        for b in self._bubbles:
            b["y"] -= b["vy"]
            b["phase"] += 0.02
            if b["y"] < self.area_y - 10:
                b["y"] = self.area_y + self.area_h + 10
                b["x"] = float(self.np_random.uniform(self.area_x, self.area_x + self.area_w))
                b["r"] = float(self.np_random.uniform(2.0, 5.5))
                b["vy"] = float(self.np_random.uniform(0.20, 0.60))
            wobble = math.sin(b["phase"]) * 0.6
            rr = int(b["r"])
            bub = pygame.Surface((rr * 2 + 2, rr * 2 + 2), pygame.SRCALPHA)
            pygame.draw.circle(bub, (180, 240, 255, 26), (rr + 1, rr + 1), rr, width=1)
            world.blit(bub, (int(b["x"] + wobble) - rr, int(b["y"]) - rr))

        # sparkles
        if self._sparkles:
            new_s = []
            for sp in self._sparkles:
                age = self.steps - sp["start"]
                if age < 0 or age > sp["life"]:
                    continue
                t = age / float(sp["life"])
                x = sp["x"] + sp["vx"] * age
                y = sp["y"] + sp["vy"] * age
                alpha = int(220 * (1.0 - t))
                if alpha <= 0:
                    continue
                rr = max(1, int(sp["r0"] * (1.0 - 0.6 * t)))
                s_surf = pygame.Surface((2 * rr + 2, 2 * rr + 2), pygame.SRCALPHA)
                pygame.draw.circle(s_surf, (*sp["color"], alpha), (rr + 1, rr + 1), rr)
                world.blit(s_surf, (int(x) - rr - 1, int(y) - rr - 1))
                new_s.append(sp)
            self._sparkles = new_s

        # small motes
        for i in range(self.n_motes):
            mx, my = self.motes_pos[i]
            glow_radius = 13
            glow = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
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
                    glow.set_at((xx, yy), (*self._mote_glow, alpha))
            world.blit(glow, (int(mx) - glow_radius, int(my) - glow_radius))
            pygame.draw.circle(world, self._mote_core, (int(mx), int(my)), int(self.mote_r))

        # big mote: glow with outer halo
        bx, by = float(self.big_pos[0]), float(self.big_pos[1])

        big_glow_r = int(self.big_r * 1.85)
        glow = pygame.Surface((big_glow_r * 2, big_glow_r * 2), pygame.SRCALPHA)
        base_alpha = 150
        for yy in range(big_glow_r * 2):
            dy = yy - big_glow_r
            for xx in range(big_glow_r * 2):
                dx = xx - big_glow_r
                dist = math.hypot(dx, dy) / big_glow_r
                if dist > 1.0:
                    continue
                falloff = (1.0 - dist) ** 2
                alpha = int(base_alpha * falloff * flicker)
                if alpha <= 0:
                    continue
                glow.set_at((xx, yy), (*self._big_glow, alpha))
        world.blit(glow, (int(bx) - big_glow_r, int(by) - big_glow_r))

        # outer halo ring
        halo_r = int(self.big_r * 1.15)
        halo = pygame.Surface((halo_r * 2 + 2, halo_r * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(halo, (*self._big_glow, int(70 * flicker)), (halo_r + 1, halo_r + 1), halo_r, width=2)
        world.blit(halo, (int(bx) - halo_r - 1, int(by) - halo_r - 1))

        pygame.draw.circle(world, self._big_core, (int(bx), int(by)), int(self.big_r))

        # agent trail
        speed = math.hypot(self.vx, self.vy)
        if speed > 0.05:
            dir_x = -self.vx / (speed + 1e-8)
            dir_y = -self.vy / (speed + 1e-8)
            trail_len = 4
            for k in range(1, trail_len + 1):
                t = k / float(trail_len + 1)
                tx = self.ax + dir_x * 4 * k
                ty = self.ay + dir_y * 4 * k
                alpha = int(140 * (1.0 - t))
                rr = max(1, int(self.agent_r * (1.0 - 0.45 * t)))
                trail = pygame.Surface((2 * rr, 2 * rr), pygame.SRCALPHA)
                pygame.draw.circle(trail, (*self._agent_trail, alpha), (rr, rr), rr)
                world.blit(trail, (int(tx) - rr, int(ty) - rr))

        ax_i, ay_i = int(self.ax), int(self.ay)

        # tint agent while stunned (darker + slightly desaturated)
        if self._stun > 0:
            t = min(1.0, self._stun / float(self.stun_steps))
            # blend toward a muted deep-teal
            muted = (40, 80, 95)
            core = tuple(int((1.0 - 0.55 * t) * c + (0.55 * t) * m) for c, m in zip(self._agent_core, muted))
            outline = tuple(int((1.0 - 0.65 * t) * c + (0.65 * t) * m) for c, m in zip(self._agent_outline, muted))
        else:
            core = self._agent_core
            outline = self._agent_outline

        pygame.draw.circle(world, core, (ax_i, ay_i), int(self.agent_r))
        pygame.draw.circle(world, outline, (ax_i, ay_i), int(self.agent_r), 1)

        haze = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        haze_alpha = int(12 + 10 * (flicker - 1.0))
        haze_alpha = max(8, min(26, haze_alpha))
        haze.fill((*self._water_haze, haze_alpha))
        world.blit(haze, (0, 0))

        surf.fill((0, 0, 0))
        surf.blit(world, (shake_x, shake_y))

        # HUD: Score only, alpha 200/100
        text = f"Score: {self.score}"
        text_surf = self._font.render(text, True, self._hud_text)
        shadow_surf = self._font.render(text, True, self._hud_shadow)
        rect = text_surf.get_rect(topleft=(8, 6))
        alpha = self._hud_alpha_for_rect(rect)

        hud = pygame.Surface((rect.width + 8, rect.height + 6), pygame.SRCALPHA)
        backing_alpha = int(0.28 * alpha)
        pygame.draw.rect(hud, (0, 15, 25, backing_alpha), hud.get_rect(), border_radius=7)

        ts = text_surf.copy()
        ss = shadow_surf.copy()
        ts.set_alpha(alpha)
        ss.set_alpha(alpha)
        hud.blit(ss, (5, 4))
        hud.blit(ts, (4, 3))
        surf.blit(hud, (rect.x - 4, rect.y - 3))

        arr = np.transpose(pygame.surfarray.pixels3d(surf), (1, 0, 2)).copy()
        return arr

    def close(self):
        if self._surface is not None:
            pygame.quit()
            self._surface = None
