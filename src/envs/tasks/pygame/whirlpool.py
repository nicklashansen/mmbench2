import gymnasium as gym
import numpy as np
import pygame
import math


class WhirlpoolEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, max_episode_steps=500, n_fish=7):
        super().__init__()
        self.max_episode_steps = int(max_episode_steps)

        # canvas
        self.width, self.height = 224, 224
        self.margin = 14
        self.area_x = self.margin
        self.area_y = self.margin
        self.area_w = self.width - 2 * self.margin
        self.area_h = self.height - 2 * self.margin

        # whirlpool center
        self.cx = self.area_x + self.area_w / 2
        self.cy = self.area_y + self.area_h / 2
        self.whirl_radius = 14.0

        # ----------------------------
        # Spiral field params
        # ----------------------------
        # agent: gentle swirl
        self.agent_spin = 0.035
        self.agent_pull = 0.010
        self.agent_max_whirl = 0.10

        # fish: strong tangential orbit, slow inward drift (after 1 lap)
        self.fish_orbit_speed = 2.00      # tangential component (px/step)
        self.fish_inward_speed = 0.20     # radial inward drift (px/step), AFTER half a lap
        self.fish_center_sink_radius = self.whirl_radius + 1.0

        # agent dynamics
        self.agent_r = 7.0
        self.agent_max_speed = 3.4
        self.agent_accel = 0.95
        self.agent_friction = 0.10

        # fish
        self.n_fish = int(n_fish)
        self.fish_r = 3.2

        # reward
        self.step_cost = -0.001
        self.catch_reward = 1.0
        self.whirlpool_penalty = -1.0
        self.whirl_push_strength = 2.7

        # state
        self.ax = self.ay = 0.0
        self.vx = self.vy = 0.0

        self.fish_pos = np.zeros((self.n_fish, 2), dtype=np.float32)
        self.fish_vel = np.zeros((self.n_fish, 2), dtype=np.float32)
        self.fish_color = np.zeros((self.n_fish,), dtype=np.int32)

        # accumulate angular travel per fish to enforce >=1 lap
        self.fish_lap = np.zeros((self.n_fish,), dtype=np.float32)
        self.fish_prev_ang = np.zeros((self.n_fish,), dtype=np.float32)

        self.steps = 0
        self.score = 0

        # streaks near center
        self.streak_count = 16
        self.streak_r_min = self.whirl_radius + 6.0
        self.streak_r_max = self.whirl_radius + 44.0
        self.streak_len = 14.0
        self.streak_width = 2

        # visuals: LIGHTER water
        self._bg_top = (18, 55, 95)
        self._bg_bottom = (8, 80, 115)
        self._arena_fill = (10, 65, 100)
        self._arena_ring = (80, 180, 205)

        self._agent_core = (210, 245, 255)
        self._agent_outline = (120, 210, 255)

        self._fish_palette = [
            (255, 235, 140),  # yellow
            (255, 175, 95),   # orange
            (255, 155, 145),  # salmon
        ]

        # water ripple splashes
        self._splashes = []  # expanding ripple rings
        self.splash_life = 22
        self.splash_ring_width = 2

        self._surface = None
        self._font = None

        # observation:
        # agent pos(2), vel(2)
        # fish: n * (relpos(2), vel(2), type(1))
        # step_frac(1)
        obs_dim = 4 + self.n_fish * 5 + 1
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-np.ones(2, dtype=np.float32), high=np.ones(2, dtype=np.float32), dtype=np.float32)

    # ---------- small math helpers ----------

    def _clamp(self, x, y, r):
        x = float(np.clip(x, self.area_x + r, self.area_x + self.area_w - r))
        y = float(np.clip(y, self.area_y + r, self.area_y + self.area_h - r))
        return x, y

    def _wrap_pi(self, a):
        # wrap to (-pi, pi]
        while a <= -math.pi:
            a += 2 * math.pi
        while a > math.pi:
            a -= 2 * math.pi
        return a

    def _random_spawn_away_from_center(self, r, min_dist=48.0):
        for _ in range(256):
            x = float(self.np_random.uniform(self.area_x + r, self.area_x + self.area_w - r))
            y = float(self.np_random.uniform(self.area_y + r, self.area_y + self.area_h - r))
            if math.hypot(x - self.cx, y - self.cy) >= min_dist:
                return x, y
        # fallback
        return self.area_x + r, self.area_y + r

    # ---------- forces ----------

    def _whirl_force_agent(self, x, y):
        dx = x - self.cx
        dy = y - self.cy
        dist = math.hypot(dx, dy) + 1e-6

        # inward (radial) pull + tangential spin
        rx, ry = dx / dist, dy / dist
        tx, ty = -ry, rx  # clockwise tangential

        fx = (-self.agent_pull * dist) * rx + (self.agent_spin) * tx
        fy = (-self.agent_pull * dist) * ry + (self.agent_spin) * ty

        mag = math.hypot(fx, fy)
        if mag > self.agent_max_whirl:
            s = self.agent_max_whirl / (mag + 1e-6)
            fx *= s
            fy *= s
        return fx, fy

    def _fish_step_spiral(self, i):
        """
        Deterministic spiral:
          - Always apply tangential velocity (clockwise)
          - Only apply inward radial drift AFTER accumulating >= half a lap (pi of angular travel)
        """
        x = float(self.fish_pos[i, 0])
        y = float(self.fish_pos[i, 1])

        dx = x - self.cx
        dy = y - self.cy
        dist = math.hypot(dx, dy) + 1e-6

        ang = math.atan2(dy, dx)

        # update lap accumulator from previous angle
        dtheta = self._wrap_pi(ang - float(self.fish_prev_ang[i]))
        self.fish_lap[i] += abs(dtheta)
        self.fish_prev_ang[i] = ang

        # unit vectors
        rx, ry = dx / dist, dy / dist
        tx, ty = -ry, rx  # clockwise tangential

        # tangential motion (dominant)
        vx = self.fish_orbit_speed * tx
        vy = self.fish_orbit_speed * ty

        # inward drift only after half a lap
        if float(self.fish_lap[i]) >= 1 * math.pi:
            vx += (-self.fish_inward_speed) * rx
            vy += (-self.fish_inward_speed) * ry

        x += vx
        y += vy

        x, y = self._clamp(x, y, self.fish_r)

        self.fish_pos[i, 0] = x
        self.fish_pos[i, 1] = y
        self.fish_vel[i, 0] = vx
        self.fish_vel[i, 1] = vy

    def _respawn_fish(self, i):
        # randomized spawn ring so they do laps
        # pick radius comfortably outside the sink region
        rmin = self.whirl_radius + 42.0
        rmax = min(self.area_w, self.area_h) * 0.48
        rr = float(self.np_random.uniform(rmin, rmax))
        ang = float(self.np_random.uniform(0, 2 * math.pi))
        x = self.cx + math.cos(ang) * rr
        y = self.cy + math.sin(ang) * rr
        x, y = self._clamp(x, y, self.fish_r)

        self.fish_pos[i, 0] = x
        self.fish_pos[i, 1] = y
        self.fish_color[i] = int(self.np_random.integers(0, 3))

        # reset lap tracking
        self.fish_lap[i] = 0.0
        self.fish_prev_ang[i] = math.atan2(y - self.cy, x - self.cx)

        # initialize vel consistent with spiral
        dx = x - self.cx
        dy = y - self.cy
        dist = math.hypot(dx, dy) + 1e-6
        rx, ry = dx / dist, dy / dist
        tx, ty = -ry, rx
        self.fish_vel[i, 0] = self.fish_orbit_speed * tx
        self.fish_vel[i, 1] = self.fish_orbit_speed * ty

    # ---------- obs ----------

    def _get_obs(self):
        ax_n = (self.ax - self.area_x) / self.area_w
        ay_n = (self.ay - self.area_y) / self.area_h
        vx_n = float(np.clip(self.vx / self.agent_max_speed, -1.0, 1.0))
        vy_n = float(np.clip(self.vy / self.agent_max_speed, -1.0, 1.0))
        feats = [ax_n, ay_n, vx_n, vy_n]

        for i in range(self.n_fish):
            fx, fy = float(self.fish_pos[i, 0]), float(self.fish_pos[i, 1])
            fvx, fvy = float(self.fish_vel[i, 0]), float(self.fish_vel[i, 1])
            feats.extend(
                [
                    float(np.clip((fx - self.ax) / self.area_w, -1.0, 1.0)),
                    float(np.clip((fy - self.ay) / self.area_h, -1.0, 1.0)),
                    float(np.clip(fvx / max(1e-6, self.fish_orbit_speed), -1.0, 1.0)),
                    float(np.clip(fvy / max(1e-6, self.fish_orbit_speed), -1.0, 1.0)),
                    float(self.fish_color[i]) / 2.0,
                ]
            )

        feats.append(float(self.steps) / float(self.max_episode_steps))
        return np.asarray(feats, dtype=np.float32)

    # ---------- gym API ----------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0

        # agent spawns away from center
        self.ax, self.ay = self._random_spawn_away_from_center(self.agent_r, min_dist=60.0)
        self.vx = self.vy = 0.0

        for i in range(self.n_fish):
            self._respawn_fish(i)

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # agent dynamics
        self.vx += float(action[0]) * self.agent_accel
        self.vy += float(action[1]) * self.agent_accel

        wx, wy = self._whirl_force_agent(self.ax, self.ay)
        self.vx += wx
        self.vy += wy

        self.vx *= (1.0 - self.agent_friction)
        self.vy *= (1.0 - self.agent_friction)

        sp = math.hypot(self.vx, self.vy)
        if sp > self.agent_max_speed:
            s = self.agent_max_speed / (sp + 1e-6)
            self.vx *= s
            self.vy *= s

        self.ax += self.vx
        self.ay += self.vy
        self.ax, self.ay = self._clamp(self.ax, self.ay, self.agent_r)

        reward = float(self.step_cost)

        # agent hits whirlpool -> penalty + push outward
        d_center = math.hypot(self.ax - self.cx, self.ay - self.cy)
        if d_center < self.whirl_radius + self.agent_r:
            reward += float(self.whirlpool_penalty)
            self._spawn_splash(self.ax, self.ay, color=(200, 245, 255), n_rings=3, life=24, r0=6.0, r1=28.0)
            nx = (self.ax - self.cx) / (d_center + 1e-6)
            ny = (self.ay - self.cy) / (d_center + 1e-6)
            self.vx += nx * self.whirl_push_strength
            self.vy += ny * self.whirl_push_strength
            # push position slightly outward too (prevents sticking)
            self.ax += nx * 2.0
            self.ay += ny * 2.0
            self.ax, self.ay = self._clamp(self.ax, self.ay, self.agent_r)

        # fish spiral update + respawn at center + catch
        for i in range(self.n_fish):
            self._fish_step_spiral(i)

            fx, fy = float(self.fish_pos[i, 0]), float(self.fish_pos[i, 1])

            # fish reaches whirlpool center -> respawn
            if math.hypot(fx - self.cx, fy - self.cy) <= self.fish_center_sink_radius:
                self._spawn_splash(fx, fy, color=(210, 250, 255), n_rings=2, life=20, r0=4.0, r1=20.0)
                self._respawn_fish(i)
                fx, fy = float(self.fish_pos[i, 0]), float(self.fish_pos[i, 1])

            # agent catches fish
            if math.hypot(fx - self.ax, fy - self.ay) <= (self.agent_r + self.fish_r):
                self.score += 1
                reward += float(self.catch_reward)
                self._respawn_fish(i)

        self.steps += 1
        truncated = self.steps >= self.max_episode_steps
        info = {"score": int(self.score)}
        return self._get_obs(), float(reward), False, truncated, info

    # ---------- rendering ----------

    def _draw_vertical_gradient(self, surf, top_rgb, bottom_rgb):
        for y in range(self.height):
            t = y / float(self.height - 1)
            r = int(top_rgb[0] * (1.0 - t) + bottom_rgb[0] * t)
            g = int(top_rgb[1] * (1.0 - t) + bottom_rgb[1] * t)
            b = int(top_rgb[2] * (1.0 - t) + bottom_rgb[2] * t)
            pygame.draw.line(surf, (r, g, b), (0, y), (self.width, y))

    def _rot2(self, x, y, ang):
        ca = math.cos(ang)
        sa = math.sin(ang)
        return x * ca - y * sa, x * sa + y * ca
    
    def _spawn_splash(self, x, y, color=(210, 245, 255), n_rings=2, life=22, r0=4.0, r1=22.0):
        # store multiple rings for a richer splash
        for k in range(n_rings):
            self._splashes.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "start": int(self.steps),
                    "life": int(life),
                    "r0": float(r0 + 2.0 * k),
                    "r1": float(r1 + 6.0 * k),
                    "w": int(max(1, self.splash_ring_width - (k > 0))),  # inner ring slightly thicker
                    "color": color,
                }
            )

    def _draw_splashes(self, surf):
        if not self._splashes:
            return
        keep = []
        for sp in self._splashes:
            age = self.steps - sp["start"]
            if age < 0 or age > sp["life"]:
                continue
            t = age / float(sp["life"])
            # smooth ease-out expansion
            tt = 1.0 - (1.0 - t) * (1.0 - t)
            r = sp["r0"] + (sp["r1"] - sp["r0"]) * tt

            # fade out
            alpha = int(180 * (1.0 - t))
            if alpha <= 0:
                continue

            ring = pygame.Surface((int(r * 2 + 4), int(r * 2 + 4)), pygame.SRCALPHA)
            pygame.draw.circle(
                ring,
                (*sp["color"], alpha),
                (ring.get_width() // 2, ring.get_height() // 2),
                int(r),
                width=sp["w"],
            )
            surf.blit(ring, (int(sp["x"] - r - 2), int(sp["y"] - r - 2)))
            keep.append(sp)
        self._splashes = keep

    def _draw_fish(self, surf, x, y, vx, vy, color, r):
        sp = math.hypot(vx, vy)
        if sp > 1e-3:
            ang = math.atan2(vy, vx)
        else:
            dx = x - self.cx
            dy = y - self.cy
            ang = math.atan2(dx, -dy)  # tangent

        body_len = r * 2.2
        body_w = r * 1.3
        tail_len = r * 1.2

        pts = [
            (body_len * 0.55, 0.0),
            (body_len * 0.10, -body_w * 0.55),
            (-body_len * 0.45, -body_w * 0.20),
            (-body_len * 0.45 - tail_len, -body_w * 0.55),
            (-body_len * 0.55, 0.0),
            (-body_len * 0.45 - tail_len, body_w * 0.55),
            (-body_len * 0.45, body_w * 0.20),
            (body_len * 0.10, body_w * 0.55),
        ]

        spts = []
        for px, py in pts:
            rx, ry = self._rot2(px, py, ang)
            spts.append((int(x + rx), int(y + ry)))

        # glow
        glow_r = int(max(5, r * 2.6))
        glow = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
        base_alpha = 30
        for yy in range(glow_r * 2):
            dy2 = yy - glow_r
            for xx in range(glow_r * 2):
                dx2 = xx - glow_r
                dist = math.hypot(dx2, dy2) / (glow_r + 1e-6)
                if dist > 1.0:
                    continue
                falloff = (1.0 - dist) ** 2
                a = int(base_alpha * falloff)
                if a > 0:
                    glow.set_at((xx, yy), (*color, a))
        surf.blit(glow, (int(x) - glow_r, int(y) - glow_r))

        pygame.draw.polygon(surf, color, spts)

        # highlight
        hx0, hy0 = self._rot2(-body_len * 0.05, -body_w * 0.15, ang)
        hx1, hy1 = self._rot2(body_len * 0.35, -body_w * 0.10, ang)
        pygame.draw.line(surf, (255, 255, 255), (int(x + hx0), int(y + hy0)), (int(x + hx1), int(y + hy1)), 1)

        # eye
        ex, ey = self._rot2(body_len * 0.30, -body_w * 0.12, ang)
        pygame.draw.circle(surf, (10, 10, 20), (int(x + ex), int(y + ey)), 1)

    def render(self):
        if self._surface is None:
            pygame.init()
            self._surface = pygame.Surface((self.width, self.height))
        if self._font is None:
            pygame.font.init()
            self._font = pygame.font.SysFont("arial", 16, bold=True)

        surf = self._surface

        # brighter watery gradient
        self._draw_vertical_gradient(surf, self._bg_top, self._bg_bottom)

        # arena
        area_rect = pygame.Rect(self.area_x, self.area_y, self.area_w, self.area_h)
        pygame.draw.rect(surf, self._arena_fill, area_rect, border_radius=18)
        pygame.draw.rect(surf, self._arena_ring, area_rect, width=2, border_radius=18)

        # animated whirl rings (light)
        for k in range(7):
            rad = self.whirl_radius + k * 6
            alpha = max(10, 70 - k * 8)
            ring = pygame.Surface((rad * 2 + 2, rad * 2 + 2), pygame.SRCALPHA)
            pygame.draw.circle(ring, (160, 230, 255, alpha), (rad + 1, rad + 1), rad, width=2)
            surf.blit(ring, (int(self.cx - rad), int(self.cy - rad)))

        # directional streaks near center
        streak_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        t0 = 0.040 * self.steps
        for k in range(self.streak_count):
            ang = (2 * math.pi * k / self.streak_count) + t0
            rr = self.streak_r_min + (self.streak_r_max - self.streak_r_min) * (0.5 + 0.5 * math.sin(ang * 1.7 + 1.3))

            sx = self.cx + math.cos(ang) * rr
            sy = self.cy + math.sin(ang) * rr

            # tangent + slight inward
            tx = -math.sin(ang)
            ty = math.cos(ang)
            tx = 0.85 * tx + 0.15 * (-math.cos(ang))
            ty = 0.85 * ty + 0.15 * (-math.sin(ang))
            norm = math.hypot(tx, ty) + 1e-6
            tx /= norm
            ty /= norm

            x0 = sx - tx * (0.35 * self.streak_len)
            y0 = sy - ty * (0.35 * self.streak_len)
            x1 = sx + tx * self.streak_len
            y1 = sy + ty * self.streak_len

            alpha = int(55 + 85 * (1.0 - (rr - self.streak_r_min) / (self.streak_r_max - self.streak_r_min + 1e-6)))
            alpha = max(25, min(140, alpha))
            pygame.draw.line(streak_surf, (170, 235, 255, alpha), (int(x0), int(y0)), (int(x1), int(y1)), self.streak_width)

        surf.blit(streak_surf, (0, 0))

        # caustic/haze overlay (lightens “water” feel)
        phase1 = math.sin(0.10 * self.steps)
        phase2 = math.sin(0.33 * self.steps + 1.1)
        haze_alpha = int(22 + 10 * phase1 + 6 * phase2)
        haze_alpha = max(16, min(36, haze_alpha))
        haze = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        haze.fill((180, 240, 255, haze_alpha))
        surf.blit(haze, (0, 0))

        self._draw_splashes(surf)

        # fish sprites
        for i in range(self.n_fish):
            fx, fy = float(self.fish_pos[i, 0]), float(self.fish_pos[i, 1])
            fvx, fvy = float(self.fish_vel[i, 0]), float(self.fish_vel[i, 1])
            col = self._fish_palette[int(self.fish_color[i])]
            self._draw_fish(surf, fx, fy, fvx, fvy, col, self.fish_r)

        # agent
        pygame.draw.circle(surf, self._agent_core, (int(self.ax), int(self.ay)), int(self.agent_r))
        pygame.draw.circle(surf, self._agent_outline, (int(self.ax), int(self.ay)), int(self.agent_r), 1)

        # HUD
        text = f"Score: {self.score}"
        t_surf = self._font.render(text, True, (245, 250, 255))
        shadow = self._font.render(text, True, (10, 25, 40))
        surf.blit(shadow, (9, 7))
        surf.blit(t_surf, (8, 6))

        arr = np.transpose(pygame.surfarray.pixels3d(surf), (1, 0, 2)).copy()
        return arr

    def close(self):
        if self._surface is not None:
            pygame.quit()
            self._surface = None
