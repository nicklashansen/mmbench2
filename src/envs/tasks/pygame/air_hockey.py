import gymnasium as gym
import numpy as np
import pygame


class AirHockeyEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, max_episode_steps=500):
        super().__init__()
        self.max_episode_steps = max_episode_steps

        # obs = [puck_x, puck_y, puck_vx, puck_vy,
        #        agent_x, agent_y, agent_vx, agent_vy,
        #        opp_x,   opp_y,   opp_vx,   opp_vy]
        self.obs_dim = 12
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # continuous 2D thrust in [-1,1]^2  → mallet velocity each step
        self.action_space = gym.spaces.Box(
            low=-np.ones(2, dtype=np.float32), high=np.ones(2, dtype=np.float32)
        )

        # Screen
        self.width, self.height = 224, 224

        # Margins to fit scores
        self.margin_top = 24
        self.margin_bottom = 24

        # Rink keeps original aspect (~0.9 width/height), scaled to fit margins
        self.rink_h = self.height - (self.margin_top + self.margin_bottom)  # 176
        self.rink_w = int(self.rink_h * 0.9)
        self.rink_w = min(self.rink_w, self.width - 20)

        self.rink_x = (self.width - self.rink_w) // 2
        self.rink_y = (self.height - self.rink_h) // 2

        # Goals at top (opponent) and bottom (agent)
        self.goal_w = max(56, int(self.rink_w * 0.35))
        self.goal_left = self.rink_x + (self.rink_w - self.goal_w) // 2
        self.goal_right = self.goal_left + self.goal_w
        self.goal_depth = 10  # px behind the border to count as a goal

        # Entities
        self.puck_r = 6
        self.agent_r = 10
        self.opp_r = 10

        # Speeds / physics
        self.paddle_speed = 7.5
        self.base_puck_speed = 6.0
        self.max_puck_speed = 12.0
        self.restitution = 0.9
        self.damping = 0.002
        self.hit_boost = 0.35

        # State
        self.agent_x = self.agent_y = 0.0
        self.opp_x = self.opp_y = 0.0
        self.agent_vx = self.agent_vy = 0.0
        self.opp_vx = self.opp_vy = 0.0
        self.puck_x = self.puck_y = 0.0
        self.puck_vx = self.puck_vy = 0.0

        # Opponent AI
        self.midline = self.rink_y + self.rink_h * 0.5
        self.defense_line = self.rink_y + int(self.rink_h * 0.35)  # keeps AI conservative
        self.opp_reaction_delay = 0
        self.opp_in_zone = False
        self.opp_home = None  # set at reset

        # Score, rendering
        self.agent_score = 0
        self.opp_score = 0
        self.steps = 0
        self._surface = None
        self.clock = None
        self.font = None

    # ------------------------------- Utils -------------------------------

    def _center(self):
        return self.rink_x + self.rink_w / 2, self.rink_y + self.rink_h / 2

    def _spawn_positions(self):
        cx, cy = self._center()
        # Agent bottom half, opponent top half
        self.agent_x = cx
        self.agent_y = self.rink_y + self.rink_h - 30
        self.opp_x = cx
        self.opp_y = self.rink_y + 30
        self.agent_vx = self.agent_vy = 0.0
        self.opp_vx = self.opp_vy = 0.0
        self.opp_home = (cx, self.rink_y + 28)

    def _reset_puck(self, launch_to_top: bool | None = None):
        cx, cy = self._center()
        self.puck_x, self.puck_y = cx, cy
        angle = self.np_random.uniform(-0.35, 0.35) * np.pi
        speed = self.base_puck_speed
        if launch_to_top is None:
            vy_sign = self.np_random.choice([-1, 1])
        else:
            vy_sign = -1 if launch_to_top else 1
        self.puck_vx = speed * np.cos(angle) * self.np_random.choice([-1, 1])
        self.puck_vy = vy_sign * speed * abs(np.sin(angle) * 0.7 + 0.3)

    def _clamp_paddle(self, x, y, r, top_half=False, bottom_half=False):
        """Keep paddle in rink and on its own half (no crossing center line)."""
        x = np.clip(x, self.rink_x + r, self.rink_x + self.rink_w - r)
        mid = self.midline
        if top_half:
            y = np.clip(y, self.rink_y + r, mid - r)
        elif bottom_half:
            y = np.clip(y, mid + r, self.rink_y + self.rink_h - r)
        else:
            y = np.clip(y, self.rink_y + r, self.rink_y + self.rink_h - r)
        return x, y

    def _limit_puck_speed(self):
        s = (self.puck_vx**2 + self.puck_vy**2) ** 0.5
        if s > self.max_puck_speed:
            scale = self.max_puck_speed / (s + 1e-8)
            self.puck_vx *= scale
            self.puck_vy *= scale

    def _circle_collision(self, cx, cy, cr, px, py, pr):
        dx, dy = px - cx, py - cy
        dist = np.hypot(dx, dy)
        overlap = cr + pr - dist
        if overlap > 0:
            nx, ny = (dx / (dist + 1e-8), dy / (dist + 1e-8))
            return True, nx, ny, overlap
        return False, 0.0, 0.0, 0.0

    def _resolve_paddle_hit(self, px, py, pvx, pvy, cx, cy, cr):
        hit, nx, ny, overlap = self._circle_collision(cx, cy, cr, px, py, self.puck_r)
        if not hit:
            return px, py, pvx, pvy

        # Separate puck out of the paddle along normal
        px += nx * (overlap + 0.5)
        py += ny * (overlap + 0.5)

        # Relative velocity along normal
        rel_vx = pvx - self._cur_paddle_vx
        rel_vy = pvy - self._cur_paddle_vy
        vrel_n = rel_vx * nx + rel_vy * ny
        if vrel_n < 0:
            pvx = pvx - (1 + self.restitution) * vrel_n * nx + self.hit_boost * self._cur_paddle_vx
            pvy = pvy - (1 + self.restitution) * vrel_n * ny + self.hit_boost * self._cur_paddle_vy

        return px, py, pvx, pvy

    # ------------------------------- Gym API -----------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_score = 0
        self.opp_score = 0
        self.steps = 0
        self._spawn_positions()
        self._reset_puck(launch_to_top=self.np_random.choice([True, False]))
        self.opp_reaction_delay = 0
        self.opp_in_zone = False
        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape == ():
            action = np.array([0.0, float(np.clip(action, -1, 1))], dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        reward = 0.0
        terminated = False
        truncated = False

        # ---------------- Agent paddle move (continuous 2D) ----------------
        self.agent_vx = float(action[0]) * self.paddle_speed
        self.agent_vy = float(action[1]) * self.paddle_speed
        self.agent_x += self.agent_vx
        self.agent_y += self.agent_vy
        self.agent_x, self.agent_y = self._clamp_paddle(
            self.agent_x, self.agent_y, self.agent_r, bottom_half=True
        )

        # ---------------- Opponent AI (beginner-ish) ----------------
        mid_y = self.midline
        puck_in_top = self.puck_y < (mid_y + 8)
        puck_approaching = self.puck_vy < 0

        if puck_in_top and puck_approaching:
            if not self.opp_in_zone:
                speed = float(np.hypot(self.puck_vx, self.puck_vy))
                min_s, max_s = 2.0, 10.0
                norm = np.clip((speed - min_s) / (max_s - min_s), 0.0, 1.0)
                slow_rng = (5, 10)
                fast_rng = (1, 4)
                low = int(slow_rng[0] + (fast_rng[0] - slow_rng[0]) * norm)
                high = int(slow_rng[1] + (fast_rng[1] - slow_rng[1]) * norm)
                self.opp_reaction_delay = int(self.np_random.integers(low, high + 1))
                self.opp_in_zone = True

            if self.opp_reaction_delay > 0:
                self.opp_reaction_delay -= 1
                target_x, target_y = self.opp_home
            else:
                target_x = np.clip(self.puck_x, self.rink_x + self.opp_r, self.rink_x + self.rink_w - self.opp_r)
                target_y = min(self.puck_y + 12.0, self.defense_line - 2.0)
        else:
            self.opp_in_zone = False
            target_x, target_y = self.opp_home

        # Move toward target
        dx, dy = target_x - self.opp_x, target_y - self.opp_y
        dist = float(np.hypot(dx, dy))
        step_speed = self.paddle_speed * 0.85
        if dist > 1e-5:
            ux, uy = dx / dist, dy / dist
            mag = min(step_speed, dist)
            jitter = self.np_random.normal(0.0, 0.25, size=2)
            self.opp_vx = ux * mag + jitter[0]
            self.opp_vy = uy * mag + jitter[1]
        else:
            self.opp_vx = self.opp_vy = 0.0

        self.opp_x += self.opp_vx
        self.opp_y += self.opp_vy
        self.opp_x, self.opp_y = self._clamp_paddle(
            self.opp_x, self.opp_y, self.opp_r, top_half=True
        )

        # --------------- Puck integration before collisions ---------------
        self.puck_x += self.puck_vx
        self.puck_y += self.puck_vy

        # --------------- Paddle–puck collisions (agent then opp) ----------
        self._cur_paddle_vx, self._cur_paddle_vy = self.agent_vx, self.agent_vy
        self.puck_x, self.puck_y, self.puck_vx, self.puck_vy = self._resolve_paddle_hit(
            self.puck_x, self.puck_y, self.puck_vx, self.puck_vy,
            self.agent_x, self.agent_y, self.agent_r
        )
        self._cur_paddle_vx, self._cur_paddle_vy = self.opp_vx, self.opp_vy
        self.puck_x, self.puck_y, self.puck_vx, self.puck_vy = self._resolve_paddle_hit(
            self.puck_x, self.puck_y, self.puck_vx, self.puck_vy,
            self.opp_x, self.opp_y, self.opp_r
        )

        # --------------- Walls / goals ------------------------------------
        # Left / right walls
        if self.puck_x - self.puck_r < self.rink_x:
            self.puck_x = self.rink_x + self.puck_r
            self.puck_vx *= -self.restitution
        elif self.puck_x + self.puck_r > self.rink_x + self.rink_w:
            self.puck_x = self.rink_x + self.rink_w - self.puck_r
            self.puck_vx *= -self.restitution

        in_mouth = (self.goal_left <= self.puck_x <= self.goal_right)

        # Top goal (agent scores when puck center passes deeper line)
        if self.puck_y <= self.rink_y - self.goal_depth and in_mouth:
            reward = 1.0
            self.agent_score += 1
            self._spawn_positions()
            self._reset_puck(launch_to_top=True)
        elif self.puck_y - self.puck_r < self.rink_y:
            if not in_mouth:
                # Bounce off top wall outside goal mouth
                self.puck_y = self.rink_y + self.puck_r
                self.puck_vy *= -self.restitution
            # else: allow puck to continue into the goal area until it crosses goal_depth

        # Bottom goal (opponent scores when puck center passes deeper line)
        if self.puck_y >= self.rink_y + self.rink_h + self.goal_depth and in_mouth:
            reward = -1.0
            self.opp_score += 1
            self._spawn_positions()
            self._reset_puck(launch_to_top=False)
        elif self.puck_y + self.puck_r > self.rink_y + self.rink_h:
            if not in_mouth:
                # Bounce off bottom wall outside goal mouth
                self.puck_y = self.rink_y + self.rink_h - self.puck_r
                self.puck_vy *= -self.restitution
            # else: allow puck to continue into the goal area until it crosses goal_depth

        # --------------- Damping & clamping -------------------------------
        self.puck_vx *= (1.0 - self.damping)
        self.puck_vy *= (1.0 - self.damping)
        self._limit_puck_speed()

        # --------------- Time & truncation --------------------------------
        self.steps += 1
        if self.steps >= self.max_episode_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        return np.array([
            self.puck_x / self.width,
            self.puck_y / self.height,
            self.puck_vx / self.max_puck_speed,
            self.puck_vy / self.max_puck_speed,
            self.agent_x / self.width,
            self.agent_y / self.height,
            self.agent_vx / self.paddle_speed,
            self.agent_vy / self.paddle_speed,
            self.opp_x / self.width,
            self.opp_y / self.height,
            self.opp_vx / self.paddle_speed,
            self.opp_vy / self.paddle_speed,
        ], dtype=np.float32)

    # ------------------------------- Render ------------------------------

    def render(self):
        if self._surface is None:
            pygame.init()
            self._surface = pygame.Surface((self.width, self.height))
            self.clock = pygame.time.Clock()
        if self.font is None:
            self.font = pygame.font.SysFont("Arial", 20, bold=True)

        self._surface.fill((0, 0, 0))

        # Rink border
        border_col = (200, 200, 200)
        pygame.draw.rect(
            self._surface, border_col,
            pygame.Rect(self.rink_x, self.rink_y, self.rink_w, self.rink_h), 2
        )

        # Carve goal mouth gaps (cover the border line with black rects)
        gap_h = 4
        pygame.draw.rect(
            self._surface, (0, 0, 0),
            pygame.Rect(self.goal_left, self.rink_y - 2, self.goal_w, gap_h + 4)
        )
        pygame.draw.rect(
            self._surface, (0, 0, 0),
            pygame.Rect(self.goal_left, self.rink_y + self.rink_h - 2, self.goal_w, gap_h + 4)
        )

        # Center line (thicker, still dark)
        pygame.draw.line(
            self._surface, (80, 80, 80),
            (self.rink_x, int(self.midline)),
            (self.rink_x + self.rink_w, int(self.midline)), 2
        )

        # Puck
        pygame.draw.circle(
            self._surface, (255, 255, 255),
            (int(self.puck_x), int(self.puck_y)), self.puck_r
        )

        # Paddles (agent orange, opponent gray/white)
        agent_col = (255, 140, 0)
        opp_col = (180, 180, 180)
        pygame.draw.circle(self._surface, agent_col, (int(self.agent_x), int(self.agent_y)), self.agent_r)
        pygame.draw.circle(self._surface, opp_col,   (int(self.opp_x),   int(self.opp_y)),   self.opp_r)

        # Scores (centered over/under the rink)
        agent_text = self.font.render(str(self.agent_score), True, agent_col)
        opp_text   = self.font.render(str(self.opp_score), True, opp_col)

        opp_y = max(2, (self.rink_y - opp_text.get_height()) // 2)
        agent_y = min(
            self.height - agent_text.get_height() - 2,
            self.rink_y + self.rink_h + (self.margin_bottom - agent_text.get_height()) // 2
        )
        cx = self.rink_x + self.rink_w // 2
        self._surface.blit(opp_text,   (cx - opp_text.get_width() // 2, opp_y))
        self._surface.blit(agent_text, (cx - agent_text.get_width() // 2, agent_y))

        return np.transpose(np.array(pygame.surfarray.pixels3d(self._surface)), (1, 0, 2)).copy()

    def close(self):
        if self._surface is not None:
            pygame.quit()
            self._surface = None
