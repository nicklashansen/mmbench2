import gymnasium as gym
import numpy as np
import pygame
import math


class CoconutDodgeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, max_episode_steps=500):
        super().__init__()
        self.max_episode_steps = max_episode_steps

        # Screen & centered 200x200 play area
        self.width, self.height = 224, 224
        self.area_size = 200
        self.area_x = (self.width - self.area_size) // 2
        self.area_y = (self.height - self.area_size) // 2

        # Horizon (sand/sky border)
        self.horizon_y = self.area_y + 92

        # Physics (1D lateral thrust)
        self.thrust = 0.85
        self.max_speed = 5.0
        self.damping = 0.97

        # Entities
        self.agent_r = 9
        self.base_ball_r = 7
        self.agent_y = self.area_y + self.area_size - 12  # fixed near bottom

        # Difficulty / counts
        self.max_balls = 6

        # Obs: [agent_x, agent_vx, nearest_dx, nearest_dy, collision_flag] + balls (x,y,vx,vy)*max_balls
        self.obs_dim = 5 + self.max_balls * 4
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-np.ones(1, dtype=np.float32), high=np.ones(1, dtype=np.float32)
        )

        # State
        self.agent_x = 0.0
        self.agent_vx = 0.0
        self.last_ax = 0.0
        self.walk_phase = 0.0
        self.foot_timer = 0
        self.balls = []
        self.particles = []    # dust puff particles
        self.footprints = []   # fading footprints

        # Visuals / bookkeeping
        self.steps = 0
        self.collision_timer = 0
        self._surface = None
        self.clock = None
        self.font = None  # reserved
        self.stars = []   # (x,y,r) only in sky band

        # Palms (fixed positions; scale jitter set in reset)
        self.palms = []
        # Dune randomness (set in reset)
        self._dune1 = (0.0, 0.0)  # (amp, phase)
        self._dune2 = (0.0, 0.0)

    # --------------------- Utils ----------------------

    def _difficulty(self):
        return min(1.0, max(0.0, self.steps / max(1, self.max_episode_steps)))

    def _spawn_entities(self):
        self.agent_x = self.area_x + self.area_size / 2
        self.agent_vx = 0.0
        self.last_ax = 0.0
        self.walk_phase = 0.0
        self.foot_timer = 0
        self.balls = []
        self.particles = []
        self.footprints = []
        self._update_ball_count()

    def _sample_speckles(self, r, k):
        pts, rad, tries = [], max(1, r - 2), 0
        while len(pts) < k and tries < 200:
            tries += 1
            dx = int(self.np_random.integers(-rad, rad + 1))
            dy = int(self.np_random.integers(-rad, rad + 1))
            if dx * dx + dy * dy <= rad * rad:
                pts.append((dx, dy, int(self.np_random.integers(1, 3))))
        return pts

    def _spawn_ball(self):
        d = self._difficulty()
        # Visual variant first (so we know the radius for spawn position)
        r = int(self.base_ball_r + int(self.np_random.integers(-1, 2)))  # -1,0,+1
        tone = int(self.np_random.integers(0, 2))  # 0=light, 1=dark
        stripe = bool(self.np_random.integers(0, 2))
        speckles = self._sample_speckles(r, int(self.np_random.integers(2, 5)))

        # Spawn INSIDE the playfield, right at the top border (emerging from it)
        x = float(self.np_random.uniform(self.area_x + r, self.area_x + self.area_size - r))
        y = float(self.area_y + r + 1)  # 1px inside

        vx = float(self.np_random.uniform(-0.5, 0.5) * (1.0 + 0.6 * d))
        vy = float(self.np_random.uniform(1.4, 2.6) * (1.0 + 0.8 * d))

        return {
            "x": x, "y": y, "vx": vx, "vy": vy,
            "state": "fall",
            "r": r,
            "crack_t": 0,
            "split_dir": 0,
            "variant": {"tone": tone, "stripe": stripe, "speckles": speckles},
        }

    def _update_ball_count(self):
        target = 1 + int(self._difficulty() * (self.max_balls - 1))
        while len(self.balls) < target:
            self.balls.append(self._spawn_ball())
        if len(self.balls) > target:
            self.balls = self.balls[:target]

    def _circle_dist(self, ax, ay, bx, by):
        return math.hypot(ax - bx, ay - by)

    def _get_obs(self):
        fallers = [b for b in self.balls if b["state"] == "fall"]
        if fallers:
            nb = min(fallers, key=lambda b: self._circle_dist(self.agent_x, self.agent_y, b["x"], b["y"]))
            ndx = (nb["x"] - self.agent_x) / self.area_size
            ndy = (nb["y"] - self.agent_y) / self.area_size
        else:
            ndx, ndy = 0.0, -1.0

        collision_flag = 1.0 if self.collision_timer > 0 else 0.0

        obs = [
            (self.agent_x - self.area_x) / self.area_size,
            self.agent_vx / self.max_speed,
            ndx, ndy,
            collision_flag,
        ]
        balls_sorted = sorted(self.balls,
                              key=lambda b: self._circle_dist(self.agent_x, self.agent_y, b["x"], b["y"]))
        for i in range(self.max_balls):
            if i < len(balls_sorted):
                b = balls_sorted[i]
                obs += [
                    (b["x"] - self.area_x) / self.area_size,
                    (b["y"] - self.area_y) / self.area_size,
                    b["vx"] / self.max_speed,
                    b["vy"] / self.max_speed,
                ]
            else:
                obs += [0.0, 0.0, 0.0, 0.0]

        return np.array(obs, dtype=np.float32)

    # --------------------- Gym API ----------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.collision_timer = 0
        self.particles.clear()
        self.footprints.clear()

        # Stars only in sky band (semi-transparent)
        n_stars = 22
        sky_top, sky_bot = self.area_y, self.horizon_y
        self.stars = [(int(self.np_random.integers(self.area_x + 6, self.area_x + self.area_size - 6)),
                       int(self.np_random.integers(sky_top + 6, max(sky_top + 7, sky_bot - 6))),
                       int(self.np_random.integers(1, 3)))
                      for _ in range(n_stars)]

        # Two fixed palms with ±10% scale jitter
        s1 = 1.1 * float(self.np_random.uniform(0.9, 1.1))
        s2 = 0.55 * float(self.np_random.uniform(0.9, 1.1))
        self.palms = [
            {"x": self.area_x + int(self.area_size * 0.28), "base_y": self.horizon_y + 16, "s": s1, "mirror": False},
            {"x": self.area_x + int(self.area_size * 0.76), "base_y": self.horizon_y + 6,  "s": s2, "mirror": True},
        ]

        # Dune amplitude/phase jitter
        self._dune1 = (float(self.np_random.uniform(3.0, 5.0)), float(self.np_random.uniform(0, 2*np.pi)))
        self._dune2 = (float(self.np_random.uniform(4.5, 6.0)), float(self.np_random.uniform(0, 2*np.pi)))

        self._spawn_entities()
        return self._get_obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # 1D thrust + damping; track recent acceleration for walk anim
        prev_vx = self.agent_vx
        self.agent_vx += a[0] * self.thrust
        self.agent_vx *= self.damping
        self.agent_vx = float(np.clip(self.agent_vx, -self.max_speed, self.max_speed))
        self.last_ax = self.agent_vx - prev_vx
        self.walk_phase += abs(self.agent_vx) * 0.12

        # Move & clamp inside play area
        left = self.area_x + self.agent_r
        right = self.area_x + self.area_size - self.agent_r
        self.agent_x = float(np.clip(self.agent_x + self.agent_vx, left, right))

        ground_y = self.area_y + self.area_size

        # Update coconuts
        for b in self.balls:
            if b["state"] == "fall":
                b["x"] += b["vx"]; b["y"] += b["vy"]
                if b["x"] < self.area_x + b["r"]:
                    b["x"] = self.area_x + b["r"]; b["vx"] *= -1
                elif b["x"] > self.area_x + self.area_size - b["r"]:
                    b["x"] = self.area_x + self.area_size - b["r"]; b["vx"] *= -1
                if b["y"] + b["r"] >= ground_y:
                    b["y"] = ground_y - b["r"]
                    b["state"] = "crack"
                    b["crack_t"] = 10
                    b["split_dir"] = 1 if (b["vx"] >= 0) else -1
                    self._spawn_dust_puff(b["x"], ground_y - 2, b["split_dir"])
            else:
                b["crack_t"] -= 1
                if b["crack_t"] <= 0:
                    b.update(self._spawn_ball())

        # Update dust particles
        self._update_particles()

        # Footprints: drop occasionally when moving
        self.foot_timer -= 1
        moving = abs(self.agent_vx) > 0.25
        if moving and self.foot_timer <= 0:
            self._drop_footprints()
            self.foot_timer = 4
        self._update_footprints()

        # Reward: -1 on hit, 0 otherwise
        hit = False
        for b in self.balls:
            if b["state"] != "fall":
                continue
            if self._circle_dist(self.agent_x, self.agent_y, b["x"], b["y"]) < (self.agent_r + b["r"]):
                hit = True
                self.collision_timer = 6
                break

        reward = -1.0 if hit else 0.0

        self.steps += 1
        self._update_ball_count()
        if self.collision_timer > 0:
            self.collision_timer -= 1

        truncated = self.steps >= self.max_episode_steps
        return self._get_obs(), float(reward), False, truncated, {}

    # --------------------- Dust particles ----------------------

    def _spawn_dust_puff(self, x, y, dir_sign):
        # Visible dust puff
        n = 16
        for _ in range(n):
            ang = self.np_random.uniform(0.9, 2.2)  # mostly upward
            speed = self.np_random.uniform(0.8, 2.2)
            vx = speed * math.cos(ang) * (0.7 + 0.3 * dir_sign)
            vy = -abs(speed * math.sin(ang)) * self.np_random.uniform(0.6, 1.1)
            self.particles.append({
                "x": float(x + self.np_random.uniform(-3, 3)),
                "y": float(y + self.np_random.uniform(-1, 1)),
                "vx": float(vx),
                "vy": float(vy),
                "life": 16,
                "r": int(self.np_random.integers(2, 4)),
            })

    def _update_particles(self):
        newp = []
        for p in self.particles:
            p["x"] += p["vx"]; p["y"] += p["vy"]
            p["vy"] += 0.15  # gravity
            p["life"] -= 1
            if (p["life"] > 0 and
                self.area_x <= p["x"] <= self.area_x + self.area_size and
                self.area_y <= p["y"] <= self.area_y + self.area_size):
                newp.append(p)
        self.particles = newp

    # --------------------- Footprints ----------------------

    def _drop_footprints(self):
        # Two prints (L/R) near the body bottom; fade over 6 frames
        body_w, body_h = 14, 18
        body_cx, body_cy = int(self.agent_x), int(self.agent_y - 6)
        body_rect = pygame.Rect(body_cx - body_w // 2, body_cy - body_h // 2, body_w, body_h)
        foot_y = body_rect.bottom + 0  # closer to body
        offset = 5
        vx_sign = 1 if self.agent_vx >= 0 else -1
        bias = -vx_sign * 1
        self.footprints.append({"x": body_cx - offset + bias, "y": foot_y + 2, "life": 6})
        self.footprints.append({"x": body_cx + offset + bias, "y": foot_y + 2, "life": 6})

    def _update_footprints(self):
        newprints = []
        for f in self.footprints:
            f["life"] -= 1
            if f["life"] > 0:
                newprints.append(f)
        self.footprints = newprints

    # --------------------- Render ----------------------

    def _draw_background(self):
        # Outside black
        self._surface.fill((0, 0, 0))

        # Sky band (sunset gradient)
        sky_h = self.horizon_y - self.area_y
        sky = pygame.Surface((self.area_size, max(1, sky_h)), pygame.SRCALPHA)
        for i in range(sky_h):
            t = i / max(1, sky_h - 1)
            r = int((1 - t) * 80 + t * 252)
            g = int((1 - t) * 40 + t * 165)
            b = int((1 - t) * 120 + t * 3)
            pygame.draw.line(sky, (r, g, b, 235), (0, i), (self.area_size, i))
        self._surface.blit(sky, (self.area_x, self.area_y))

        # Stars (only in sky band)
        star_ov = pygame.Surface((self.area_size, max(1, sky_h)), pygame.SRCALPHA)
        for (sx, sy, r) in self.stars:
            pygame.draw.circle(star_ov, (255, 255, 255, 110), (sx - self.area_x, sy - self.area_y), r)
        self._surface.blit(star_ov, (self.area_x, self.area_y))

        # Sand band
        sand_h = self.area_y + self.area_size - self.horizon_y
        sand = pygame.Surface((self.area_size, sand_h), pygame.SRCALPHA)
        sand.fill((196, 178, 128, 235))
        self._surface.blit(sand, (self.area_x, self.horizon_y))

        # Dunes with per-episode jitter (crest anchored at horizon)
        amp1, ph1 = self._dune1
        crest = [(self.area_x, self.horizon_y)]
        for i in range(self.area_size + 1):
            x = self.area_x + i
            y = self.horizon_y + int(amp1 + amp1 * math.sin(i / 23.0 + ph1))
            crest.append((x, y))
        crest += [(self.area_x + self.area_size, self.horizon_y + sand_h), (self.area_x, self.horizon_y + sand_h)]
        pygame.draw.polygon(self._surface, (208, 190, 140), crest)

        amp2, ph2 = self._dune2
        crest2 = [(self.area_x, self.horizon_y + 10)]
        for i in range(self.area_size + 1):
            x = self.area_x + i
            y = self.horizon_y + int(12 + amp2 * math.sin((i + 40) / 19.0 + ph2))
            crest2.append((x, y))
        crest2 += [(self.area_x + self.area_size, self.horizon_y + sand_h), (self.area_x, self.horizon_y + sand_h)]
        pygame.draw.polygon(self._surface, (192, 175, 132), crest2)

        # Two fixed palm silhouettes (with jittered scales)
        for palm in self.palms:
            self._draw_palm(palm)

        # Playfield border
        pygame.draw.rect(self._surface, (230, 230, 230),
                         pygame.Rect(self.area_x, self.area_y, self.area_size, self.area_size), 2)

    def _draw_palm(self, palm):
        cx, by, s = int(palm["x"]), int(palm["base_y"]), palm["s"]
        mirror = -1 if palm["mirror"] else 1
        color = (50, 35, 20)

        # Trunk
        pts = []
        for i in range(12):
            t = i / 11.0
            x = cx + mirror * int(5 * math.sin(t * 1.2))
            y = by - int(44 * s * t)
            pts.append((x, y))
        pygame.draw.lines(self._surface, color, False, pts, 3)

        # Fronds
        top = pts[-1]
        base_angles = [-1.05, -0.6, -0.2, 0.2, 0.6, 1.05]
        for ang in base_angles:
            length = 26 * s * (0.9 + 0.2 * (1 - abs(ang)))
            self._draw_frond(top, ang, length, 5.0 * s, mirror, color)

    def _draw_frond(self, base, angle, length, width, mirror, color):
        bx, by = base
        tip = (bx + int(mirror * length * math.cos(angle)),
               by - int(length * math.sin(angle)))
        pygame.draw.line(self._surface, color, base, tip, 2)
        n_leaflets = 7
        for i in range(n_leaflets):
            t = (i + 1) / (n_leaflets + 1)
            ang_t = angle + 0.25 * (t - 0.5)
            cx = bx + int(mirror * (t * length) * math.cos(ang_t))
            cy = by - int((t * length) * math.sin(ang_t))
            dx = mirror * math.cos(ang_t); dy = -math.sin(ang_t)
            nx, ny = -dy, dx
            side = -1 if (i % 2 == 0) else 1
            leaf_len = width * (1.5 - t) * 2.2
            leaf_w = width * (1.0 - 0.5 * t)
            p0 = (cx, cy)
            p1 = (int(cx + side * nx * leaf_w), int(cy + side * ny * leaf_w))
            p2 = (int(cx + dx * leaf_len + side * nx * (leaf_w * 0.4)),
                  int(cy + dy * leaf_len + side * ny * (leaf_w * 0.4)))
            pygame.draw.polygon(self._surface, color, [p0, p1, p2])

    # --------------------- Traveler (agent) ----------------------

    def _rotated_ellipse(self, center, size, angle_deg, color, width=0):
        """Draw a rotated ellipse by blitting a rotated Surface."""
        w, h = size
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.ellipse(surf, color, pygame.Rect(0, 0, w, h), width)
        rsurf = pygame.transform.rotate(surf, angle_deg)
        rect = rsurf.get_rect(center=center)
        self._surface.blit(rsurf, rect.topleft)

    def _draw_traveler(self):
        x, y = int(self.agent_x), int(self.agent_y)

        # Body
        body_w, body_h = 14, 18
        body_cx, body_cy = x, y - 6
        body_rect = pygame.Rect(body_cx - body_w // 2, body_cy - body_h // 2, body_w, body_h)
        pygame.draw.ellipse(self._surface, (235, 205, 170), body_rect)
        pygame.draw.ellipse(self._surface, (160, 120, 90), body_rect, 2)

        # Head with a short neck connecting to the body
        head_r = 5
        head_cx = body_cx
        head_cy = body_rect.top - head_r + 1
        pygame.draw.rect(self._surface, (235, 205, 170),
                        pygame.Rect(head_cx - 3, head_cy + head_r - 3, 6, 4))
        pygame.draw.circle(self._surface, (245, 220, 185), (head_cx, head_cy), head_r)
        pygame.draw.circle(self._surface, (160, 120, 90), (head_cx, head_cy), head_r, 1)

        # Arms angled outward (/0\), gentle sway with walk phase
        sway = 10.0 * math.sin(self.walk_phase)
        base_angle = 35  # outward, slightly up
        right_angle = -base_angle + (sway * 0.4)  # right arm tilts up-right
        left_angle  =  base_angle - (sway * 0.4)  # left arm tilts up-left
        shoulder_y = body_rect.top + 5
        right_center = (body_rect.right - 1, shoulder_y)
        left_center  = (body_rect.left + 1,  shoulder_y)
        self._rotated_ellipse(right_center, (10, 4), right_angle, (160, 120, 90))
        self._rotated_ellipse(left_center,  (10, 4), left_angle,  (160, 120, 90))

        # Feet (close to body), slight alternating step
        step = 2.0 * math.sin(self.walk_phase)
        foot_y = body_rect.bottom + 0
        pygame.draw.ellipse(self._surface, (90, 70, 55),
                            pygame.Rect(x - 6, foot_y + int(-0.6 * step), 6, 3))  # left
        pygame.draw.ellipse(self._surface, (90, 70, 55),
                            pygame.Rect(x + 1, foot_y + int(0.6 * step), 6, 3))   # right

    def _draw_coconut(self, b):
        cx, cy, r = int(b["x"]), int(b["y"]), b["r"]
        shell = (133, 102, 62) if b["variant"]["tone"] == 0 else (110, 82, 46)
        border = (90, 66, 38)
        pygame.draw.circle(self._surface, shell, (cx, cy), r)
        pygame.draw.circle(self._surface, border, (cx, cy), r, 2)
        if b["variant"]["stripe"]:
            ang = 0.6
            for t in np.linspace(-ang, ang, 10):
                x = cx + int((r - 2) * math.cos(t))
                y = cy + int((r - 2) * math.sin(t))
                if (x - cx) ** 2 + (y - cy) ** 2 <= (r - 1) ** 2:
                    self._surface.set_at((x, y), border)
        for dx, dy, sr in b["variant"]["speckles"]:
            x, y = cx + dx, cy + dy
            if (x - cx) ** 2 + (y - cy) ** 2 <= (r - 1) ** 2:
                pygame.draw.circle(self._surface, (70, 52, 30), (x, y), sr)

    def _semicircle_points(self, cx, cy, r, start, end, steps=14):
        return [(cx + r * math.cos(t), cy + r * math.sin(t))
                for t in np.linspace(start, end, steps)]

    def _draw_coconut_cracked(self, b):
        # Sideways split: left half always left, right always right (no flips)
        cx, cy, r = int(b["x"]), int(b["y"]), b["r"]
        shell = (123, 92, 52)
        border = (90, 66, 38)
        flesh = (240, 240, 240)
        t = max(0, min(10, b["crack_t"]))
        prog = (10 - t) / 10.0
        xsep = int(6 * prog)
        bias = int(2 * prog) * b["split_dir"]
        droop = int(2 * prog)

        lx = cx - xsep - bias
        rx = cx + xsep + bias
        ly = ry = cy + droop

        left_pts  = self._semicircle_points(lx, ly, r,  math.pi/2, 3*math.pi/2)
        right_pts = self._semicircle_points(rx, ry, r, -math.pi/2,  math.pi/2)

        pygame.draw.polygon(self._surface, shell, left_pts)
        pygame.draw.lines(self._surface, border, False, left_pts, 2)
        inner_l = self._semicircle_points(lx, ly, r - 3, math.pi/2, 3*math.pi/2)
        pygame.draw.lines(self._surface, flesh, False, inner_l, 2)

        pygame.draw.polygon(self._surface, shell, right_pts)
        pygame.draw.lines(self._surface, border, False, right_pts, 2)
        inner_r = self._semicircle_points(rx, ry, r - 3, -math.pi/2, math.pi/2)
        pygame.draw.lines(self._surface, flesh, False, inner_r, 2)

    def render(self):
        if self._surface is None:
            pygame.init()
            self._surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            self.clock = pygame.time.Clock()
        if self.font is None:
            self.font = pygame.font.SysFont("Arial", 16, bold=True)

        self._draw_background()

        # Footprints (draw under coconuts/agent)
        for f in self.footprints:
            alpha = int(190 * (f["life"] / 6))
            col = (70, 55, 45, max(0, min(255, alpha)))
            surf = pygame.Surface((6, 3), pygame.SRCALPHA)
            pygame.draw.ellipse(surf, col, pygame.Rect(0, 0, 6, 3))
            self._surface.blit(surf, (int(f["x"] - 3), int(f["y"] - 1)))

        # Coconuts
        for b in self.balls:
            if b["state"] == "fall":
                self._draw_coconut(b)
            else:
                self._draw_coconut_cracked(b)

        # Dust particles
        for p in self.particles:
            alpha = int(230 * (p["life"] / 16))
            col = (215, 195, 150, max(0, min(255, alpha)))
            surf = pygame.Surface((p["r"]*2+2, p["r"]*2+2), pygame.SRCALPHA)
            pygame.draw.circle(surf, col, (p["r"]+1, p["r"]+1), p["r"])
            self._surface.blit(surf, (int(p["x"] - p["r"] - 1), int(p["y"] - p["r"] - 1)))

        # Traveler
        self._draw_traveler()

        # Collision overlay + shake
        shake_x, shake_y = 0, 0
        if self.collision_timer > 0:
            shake_x = np.random.randint(-3, 4)
            shake_y = np.random.randint(-3, 4)
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((186, 108, 24, 70))
            self._surface.blit(overlay, (0, 0))

        # Shake canvas and return rgb array
        canvas = pygame.Surface((self.width + 6, self.height + 6))
        canvas.fill((0, 0, 0))
        canvas.blit(self._surface, (3 + shake_x, 3 + shake_y))
        arr = np.transpose(np.array(pygame.surfarray.pixels3d(canvas))[3:-3, 3:-3], (1, 0, 2)).copy()
        return arr

    def close(self):
        if self._surface is not None:
            pygame.quit()
            self._surface = None
