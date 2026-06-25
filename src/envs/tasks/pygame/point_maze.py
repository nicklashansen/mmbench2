import gymnasium as gym
import numpy as np
import pygame
import math
from collections import deque


class PointMazeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, max_episode_steps=200, grid_id=None):
        super().__init__()
        self.max_episode_steps = max_episode_steps

        # canvas / play area
        self.width, self.height = 224, 224
        self.area_size = 184
        self.area_x = (self.width - self.area_size) // 2
        self.area_y = (self.height - self.area_size) // 2

        # agent / goal
        self.agent_r = 6
        self.goal_r = 6
        self.speed = 5.0

        # goal distance constraints (pixels)
        self.min_goal_d = 0.2 * self.area_size
        self.max_goal_d = 0.7 * self.area_size

        # 5x5 grid templates (1 = obstacle, 0 = free)
        self.grid_templates = [
            # Maze 1
            np.array([
                [0, 1, 1, 0, 1],
                [0, 0, 1, 0, 0],
                [1, 0, 0, 0, 1],
                [1, 1, 0, 0, 0],
                [1, 0, 0, 1, 0],
            ], dtype=np.int32),
            # Maze 2
            np.array([
                [1, 0, 1, 1, 0],
                [0, 0, 1, 0, 0],
                [1, 0, 0, 0, 1],
                [0, 0, 1, 0, 0],
                [0, 1, 1, 0, 1],
            ], dtype=np.int32),
            # Maze 3
            np.array([
                [0, 1, 1, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0],
                [1, 0, 0, 0, 0],
            ], dtype=np.int32),
            # U-shape
            np.array([
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
            ], dtype=np.int32),
            # Diamond
            np.array([
                [1, 1, 0, 1, 1],
                [1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1],
                [1, 1, 0, 1, 1],
            ], dtype=np.int32),
        ]
        self.grid_id = grid_id

        # observation: [ax,ay,gx,gy,vx,vy] + 25 grid + 4 dir + aux(9)
        obs_dim = 6 + 25 + 4 + 9
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # continuous 2D thrust in [-1,1]^2
        self.action_space = gym.spaces.Box(
            low=-np.ones(2, dtype=np.float32), high=np.ones(2, dtype=np.float32)
        )

        # runtime state
        self.grid = None
        self.obstacles = []  # list[pygame.Rect] for obstacle cells
        self.ax = self.ay = 0.0
        self.vx = self.vy = 0.0
        self.gx = self.gy = 0.0
        self.steps = 0

        # directional-force cell ("wind" cell)
        self.wind_strength = 2.0  # px/step; < speed so agent can compensate
        self.force_cell = None        # (i, j) in grid
        self.force_dir = None         # 0=up,1=right,2=down,3=left
        self.force_dir_onehot = np.zeros(4, dtype=np.float32)
        self.force_cell_rect = None   # pygame.Rect cache

        # visuals
        self.trail = deque(maxlen=24)
        self._grad_surf = None
        self._glow_period_steps = 40
        self._glow_amp = 0.08
        self._glow_noise_std = 0.02
        self._glow_base_alpha = 95
        self._glow_outer_pad = 32
        self.goal_phase = 0.0

        # arrow animation in force cell
        self.arrow_gap = 16
        self.arrow_speed = 1.2  # px/step along the force direction
        self.arrow_color = (120, 230, 255, 150)

        self._surface = None
        self.clock = None

    # ---------- helpers ----------
    def _cell_rect(self, i, j):
        n = 5
        cell_w = self.area_size / n
        cell_h = self.area_size / n
        x = self.area_x + j * cell_w
        y = self.area_y + i * cell_h
        return pygame.Rect(int(round(x)), int(round(y)),
                           int(round(cell_w)), int(round(cell_h)))

    def _build_obstacles_from_grid(self):
        self.obstacles = []
        for i in range(5):
            for j in range(5):
                if self.grid[i, j] == 1:  # 1 is obstacle, 0 is free
                    self.obstacles.append(self._cell_rect(i, j))

    def _collides(self, x, y, r):
        for rect in self.obstacles:
            if rect.inflate(2 * r, 2 * r).collidepoint(int(x), int(y)):
                return True
        return False

    def _clamp_to_bounds(self, x, y, r):
        x = np.clip(x, self.area_x + r + 1, self.area_x + self.area_size - r - 1)
        y = np.clip(y, self.area_y + r + 1, self.area_y + self.area_size - r - 1)
        return x, y

    def _sample_free_xy(self, min_dist_from=None, min_d=0.0, max_d=np.inf, r=None):
        if r is None:
            r = self.agent_r
        for _ in range(4000):
            x = self.np_random.uniform(self.area_x + r + 2, self.area_x + self.area_size - r - 2)
            y = self.np_random.uniform(self.area_y + r + 2, self.area_y + self.area_size - r - 2)
            if self._collides(x, y, r):
                continue
            if min_dist_from is not None:
                d = np.hypot(x - min_dist_from[0], y - min_dist_from[1])
                if not (min_d <= d <= max_d):
                    continue
            return x, y
        return self.area_x + self.area_size / 2, self.area_y + self.area_size / 2

    def _pick_force_cell(self):
        # choose a free (0) cell uniformly, set it to -1 in obs only (kept free for movement)
        free_cells = np.argwhere(self.grid == 0)
        if len(free_cells) == 0:
            self.force_cell = None
            self.force_cell_rect = None
            self.force_dir = None
            self.force_dir_onehot[:] = 0
            return
        idx = int(self.np_random.integers(0, len(free_cells)))
        i, j = map(int, free_cells[idx])
        self.force_cell = (i, j)
        self.force_cell_rect = self._cell_rect(i, j)
        self.force_dir = int(self.np_random.integers(0, 4))  # 0=up,1=right,2=down,3=left
        self.force_dir_onehot[:] = 0
        self.force_dir_onehot[self.force_dir] = 1.0

    def _in_force_cell(self, x, y):
        return (self.force_cell_rect is not None and
                self.force_cell_rect.collidepoint(int(x), int(y)))

    def _grid_obs_flat(self):
        g = self.grid.astype(np.float32).copy()
        if self.force_cell is not None:
            i, j = self.force_cell
            g[i, j] = -1.0
        return g.reshape(-1)
    
    def _inside_goal(self, x, y):
        return np.hypot(x - self.gx, y - self.gy) <= (self.agent_r + self.goal_r)

    def _agent_cell_rc(self, x, y):
        # 5x5 indices in [0..4], clamped
        rel_x = (x - self.area_x) / self.area_size
        rel_y = (y - self.area_y) / self.area_size
        j = int(np.clip(np.floor(5 * rel_x), 0, 4))
        i = int(np.clip(np.floor(5 * rel_y), 0, 4))
        return i, j

    def _force_vec_here(self, x, y):
        if self._in_force_cell(x, y) and self.force_dir is not None:
            s = self.wind_strength / self.speed  # normalize to ~[-1,1]
            if self.force_dir == 0:   return (0.0, -s)
            if self.force_dir == 1:   return ( s,  0.0)
            if self.force_dir == 2:   return (0.0,  s)
            else:                     return (-s, 0.0)
        return (0.0, 0.0)

    # ---------- gym API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # random grid
        if self.grid_id is not None:
            idx = self.grid_id
        else:
            idx = self.np_random.integers(0, len(self.grid_templates))
        self.grid = self.grid_templates[idx].copy()
        self._build_obstacles_from_grid()

        # choose force cell + direction
        self._pick_force_cell()

        # agent & goal
        self.ax, self.ay = self._sample_free_xy(r=self.agent_r)
        self.vx = self.vy = 0.0
        self.gx, self.gy = self._sample_free_xy(
            min_dist_from=(self.ax, self.ay),
            min_d=self.min_goal_d,
            max_d=self.max_goal_d,
            r=self.goal_r,
        )

        self.goal_phase = float(self.np_random.uniform(0.0, 2.0 * np.pi))
        self.trail.clear()
        self.trail.append((self.ax, self.ay))
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # base velocity from action
        vx, vy = action * self.speed

        # add directional push if inside force cell
        if self._in_force_cell(self.ax, self.ay) and self.force_dir is not None:
            if self.force_dir == 0:      vy += -self.wind_strength
            elif self.force_dir == 1:    vx +=  self.wind_strength
            elif self.force_dir == 2:    vy +=  self.wind_strength
            else:                        vx += -self.wind_strength

        intended_dx, intended_dy = float(vx), float(vy)
        self.vx, self.vy = intended_dx, intended_dy

        # axis-separated move with collision; track blocks
        nx, ny = self.ax + self.vx, self.ay
        nx, ny = self._clamp_to_bounds(nx, ny, self.agent_r)
        if not self._collides(nx, ny, self.agent_r):
            self.ax = nx
        nx, ny = self.ax, self.ay + self.vy
        nx, ny = self._clamp_to_bounds(nx, ny, self.agent_r)
        if not self._collides(nx, ny, self.agent_r):
            self.ay = ny

        goal_dist = np.clip((np.hypot(self.ax - self.gx, self.ay - self.gy)) / self.area_size, 0, 1)
        reward = 1-goal_dist

        # bookkeeping
        info = {
            'success': self._inside_goal(self.ax, self.ay),
            'goal_distance': goal_dist,
        }
        self.steps += 1
        self.trail.append((self.ax, self.ay))
        terminated = False
        truncated = self.steps >= self.max_episode_steps

        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self):
        # base normalized features
        base = np.array([
            self.ax / self.width,
            self.ay / self.height,
            self.gx / self.width,
            self.gy / self.height,
            self.vx / self.speed,
            self.vy / self.speed,
        ], dtype=np.float32)

        # aux: geom to goal
        dx = (self.gx - self.ax) / self.area_size
        dy = (self.gy - self.ay) / self.area_size
        dist = np.hypot(dx, dy).astype(np.float32)
        in_goal_flag = np.float32(self._inside_goal(self.ax, self.ay))

        # aux: force at current pos
        fx, fy = self._force_vec_here(self.ax, self.ay)
        in_force_flag = np.float32(1.0 if (fx != 0.0 or fy != 0.0) else 0.0)

        # aux: agent cell (row,col) normalized
        ci, cj = self._agent_cell_rc(self.ax, self.ay)
        cell_rc = np.array([ci / 4.0, cj / 4.0], dtype=np.float32)

        aux = np.array([dx, dy, dist, in_goal_flag, in_force_flag, fx, fy], dtype=np.float32)

        return np.concatenate([
            base,
            self._grid_obs_flat(),                # 25
            self.force_dir_onehot.astype(np.float32),  # 4
            aux,                                  # 7
            cell_rc,                               # 2
        ], dtype=np.float32)

    # ---------- rendering ----------
    def render(self):
        if self._surface is None:
            pygame.init()
            self._surface = pygame.Surface((self.width, self.height))

        surf = self._surface

        # background
        surf.fill((6, 8, 10))
        if self._grad_surf is None:
            self._grad_surf = pygame.Surface((self.area_size, self.area_size))
            for i in range(self.area_size):
                t = i / (self.area_size - 1)
                c = (int(20 + 10*(1-t)), int(28 + 8*(1-t)), int(36 + 6*(1-t)))
                pygame.draw.line(self._grad_surf, c, (0, i), (self.area_size, i))
        surf.blit(self._grad_surf, (self.area_x, self.area_y))

        # border
        pygame.draw.rect(
            surf, (180, 180, 190),
            pygame.Rect(self.area_x, self.area_y, self.area_size, self.area_size), 2
        )

        # force cell highlight + animated arrows (bounded + centered)
        if self.force_cell_rect is not None:
            r = self.force_cell_rect

            # soft tint
            tint = pygame.Surface((r.width, r.height), pygame.SRCALPHA)
            tint.fill((90, 150, 210, 55))
            surf.blit(tint, r.topleft)

            # draw arrows into a cell-local surface so they can't exceed the cell
            arrows = pygame.Surface((r.width, r.height), pygame.SRCALPHA)

            gap = self.arrow_gap
            shift = (self.steps * self.arrow_speed) % gap
            cx, cy = r.width * 0.5, r.height * 0.5

            # arrow geometry (half-sizes) and safety margin
            hx, hy = 4, 10
            margin = max(hx, hy) + 2  # ensure arrows stay fully inside the cell

            def draw_arrow_local(x, y, dir_idx):
                a = self.arrow_color
                if dir_idx == 1:   # right
                    pts = [(x - hx, y - hy), (x - hx, y + hy), (x + hx, y)]
                elif dir_idx == 3: # left
                    pts = [(x + hx, y - hy), (x + hx, y + hy), (x - hx, y)]
                elif dir_idx == 0: # up
                    pts = [(x - hy, y + hx), (x + hy, y + hx), (x, y - hx)]
                else:              # down
                    pts = [(x - hy, y - hx), (x + hy, y - hx), (x, y + hx)]
                pygame.draw.polygon(arrows, a, [(int(px), int(py)) for px, py in pts])

            if self.force_dir in (1, 3):  # horizontal flow
                # rows centered vertically
                ky_min = math.ceil((margin - cy) / gap)
                ky_max = math.floor((r.height - margin - cy) / gap)
                ys = [cy + ky * gap for ky in range(ky_min, ky_max + 1)]

                # columns shift along x; base_x chosen so arrows wrap but never cross margins
                base_x = cx + (shift if self.force_dir == 1 else -shift)
                kx_min = math.ceil((margin - base_x) / gap)
                kx_max = math.floor((r.width - margin - base_x) / gap)
                xs = [base_x + kx * gap for kx in range(kx_min, kx_max + 1)]

                for y in ys:
                    for x in xs:
                        draw_arrow_local(x, y, self.force_dir)

            else:  # vertical flow
                # columns centered horizontally
                kx_min = math.ceil((margin - cx) / gap)
                kx_max = math.floor((r.width - margin - cx) / gap)
                xs = [cx + kx * gap for kx in range(kx_min, kx_max + 1)]

                # rows shift along y
                base_y = cy + (shift if self.force_dir == 2 else -shift)
                ky_min = math.ceil((margin - base_y) / gap)
                ky_max = math.floor((r.height - margin - base_y) / gap)
                ys = [base_y + ky * gap for ky in range(ky_min, ky_max + 1)]

                for x in xs:
                    for y in ys:
                        draw_arrow_local(x, y, self.force_dir)

            surf.blit(arrows, r.topleft)

        # goal glow (lantern-style)
        max_r_base = self.goal_r + self._glow_outer_pad
        phase = (2.0 * math.pi) * (self.steps / float(self._glow_period_steps)) + self.goal_phase
        base = math.sin(phase)
        noise = float(self.np_random.normal(0.0, self._glow_noise_std))
        flicker = 1.0 + self._glow_amp * base + noise
        flicker = max(0.85, min(1.15, flicker))
        max_r = int(round(max_r_base * flicker))
        overlay = pygame.Surface((max_r * 2, max_r * 2), pygame.SRCALPHA)
        cx = cy = max_r
        base_alpha = int(self._glow_base_alpha * flicker)
        inner_rgb = (90, 235, 150)
        outer_rgb = (60, 190, 120)

        for rr in range(max_r, 0, -2):
            t = rr / max_r
            alpha = int(base_alpha * ((1.0 - t) ** 2.2))
            if alpha <= 0:
                continue
            col = (
                int(inner_rgb[0] * (1 - t) + outer_rgb[0] * t),
                int(inner_rgb[1] * (1 - t) + outer_rgb[1] * t),
                int(inner_rgb[2] * (1 - t) + outer_rgb[2] * t),
                alpha,
            )
            pygame.draw.circle(overlay, col, (cx, cy), rr)
        surf.blit(overlay, (int(self.gx) - max_r, int(self.gy) - max_r))
        pygame.draw.circle(surf, (80, 220, 120), (int(self.gx), int(self.gy)), self.goal_r)

        # agent trail (faint)
        trail_layer = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for k, (x, y) in enumerate(self.trail):
            a = int(20 + 220 * (k / max(1, len(self.trail)-1)))
            r = max(2, int(self.agent_r * (0.6 + 0.4 * k / max(1, len(self.trail)-1))))
            pygame.draw.circle(trail_layer, (255, 160, 60, a//8), (int(x), int(y)), r+2)
            pygame.draw.circle(trail_layer, (255, 140, 0, a//3), (int(x), int(y)), r)
        surf.blit(trail_layer, (0, 0))

        # obstacles
        for rect in self.obstacles:
            shadow = rect.move(1, 1)
            pygame.draw.rect(surf, (30, 30, 35), shadow, border_radius=4)
            pygame.draw.rect(surf, (180, 180, 190), rect, border_radius=4)
            pygame.draw.rect(surf, (120, 120, 130), rect, width=1, border_radius=4)

        # agent
        pygame.draw.circle(surf, (255, 180, 80), (int(self.ax), int(self.ay)), self.agent_r)
        pygame.draw.circle(surf, (255, 120, 0), (int(self.ax), int(self.ay)), self.agent_r, width=1)

        arr = np.transpose(np.array(pygame.surfarray.pixels3d(surf)), (1, 0, 2)).copy()
        return arr

    def close(self):
        if self._surface is not None:
            pygame.quit()
            self._surface = None


class PointMazeVariant1Env(PointMazeEnv):
    def __init__(self, max_episode_steps=200):
        super().__init__(max_episode_steps=max_episode_steps, grid_id=1)


class PointMazeVariant2Env(PointMazeEnv):
    def __init__(self, max_episode_steps=200):
        super().__init__(max_episode_steps=max_episode_steps, grid_id=2)


class PointMazeVariant3Env(PointMazeEnv):
    def __init__(self, max_episode_steps=200):
        super().__init__(max_episode_steps=max_episode_steps, grid_id=3)


class PointMazeVariant4Env(PointMazeEnv):
    def __init__(self, max_episode_steps=200):
        super().__init__(max_episode_steps=max_episode_steps, grid_id=4)
