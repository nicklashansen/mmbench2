import gymnasium as gym
import numpy as np
import pygame
import math


class FrozenLakeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, max_episode_steps=500, grid_size=6, n_flags=2):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.grid_size = grid_size
        self.n_flags = n_flags

        # canvas / play area
        self.width, self.height = 224, 224
        self.area_size = 184
        self.area_x = (self.width - self.area_size) // 2
        self.area_y = (self.height - self.area_size) // 2

        # agent (penguin)
        self.agent_r = 8 if grid_size == 4 else 7
        self.speed = 2.5
        self.friction = 0.7  # sliding behavior
        self.max_speed = self.speed * 2.0
        self.pushback_strength = self.speed * 1.9

        # flags (research beacons)
        self.flag_r = 9 if grid_size == 4 else 7

        # grid (0 = ice, 1 = cracked, 2 = water)
        self.cell_w = self.area_size / self.grid_size
        self.cell_h = self.area_size / self.grid_size
        self.grid_state = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.grid_timer = np.zeros_like(self.grid_state, dtype=np.int32)
        self.crack_spawn_prob = 0.0004 * (grid_size**2)
        self.crack_to_water_steps = 80

        # runtime state
        self.ax = self.ay = 0.0
        self.vx = self.vy = 0.0
        self.last_safe_x = self.last_safe_y = 0.0
        self.flag_x = np.zeros(self.n_flags, dtype=np.float32)
        self.flag_y = np.zeros(self.n_flags, dtype=np.float32)
        self.flag_active = np.zeros(self.n_flags, dtype=bool)
        self.steps = 0

        # observation: agent(2) + vel(2) + flags(n*3) + grid(6*6)
        obs_dim = 2 + 2 + 3 * self.n_flags + self.grid_size * self.grid_size
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # continuous 2D thrust in [-1,1]^2
        self.action_space = gym.spaces.Box(
            low=-np.ones(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )

        # visuals
        self._surface = None
        self._bg_snow = (235, 242, 250)
        self._lake_ice = (170, 210, 230)
        self._lake_ice_cracked = (150, 190, 215)
        self._lake_water = (25, 70, 130)
        self._grid_line = (190, 220, 235)
        self._border_color = (210, 220, 230)
        self._crack_line = (120, 150, 170)

        # penguin colors
        self._penguin_body = (30, 30, 45)
        self._penguin_belly = (235, 245, 252)
        self._penguin_outline = (20, 30, 50)
        self._penguin_beak = (240, 180, 60)
        self._penguin_feet = (240, 180, 60)
        self._penguin_eye = (250, 250, 255)

        # flag colors
        self._flag_pole = (80, 80, 90)
        self._flag_color = (230, 80, 90)

        # score popups
        self._score_popups = []  # list of dicts: {'x','y','start_step'}
        self._score_popup_duration = 25
        self._score_color = (70, 150, 255)
        self._font_score = None

    # ---------- helpers ----------
    def _clamp_to_bounds(self, x, y, r):
        x = np.clip(x, self.area_x + r + 1, self.area_x + self.area_size - r - 1)
        y = np.clip(y, self.area_y + r + 1, self.area_y + self.area_size - r - 1)
        return x, y

    def _cell_from_pos(self, x, y):
        if (
            x < self.area_x
            or x >= self.area_x + self.area_size
            or y < self.area_y
            or y >= self.area_y + self.area_size
        ):
            return None
        col = int((x - self.area_x) / self.cell_w)
        row = int((y - self.area_y) / self.cell_h)
        col = max(0, min(self.grid_size - 1, col))
        row = max(0, min(self.grid_size - 1, row))
        return row, col

    def _cell_center(self, row, col):
        cx = self.area_x + (col + 0.5) * self.cell_w
        cy = self.area_y + (row + 0.5) * self.cell_h
        return cx, cy

    def _spawn_flag(self, idx):
        # choose a non-water cell
        mask = self.grid_state != 2
        candidates = np.argwhere(mask)

        if len(candidates) == 0:
            # fallback: anywhere if things degenerate
            row = self.np_random.integers(0, self.grid_size)
            col = self.np_random.integers(0, self.grid_size)
        else:
            row, col = candidates[self.np_random.integers(0, len(candidates))]

        x, y = self._cell_center(row, col)
        self.flag_x[idx] = x
        self.flag_y[idx] = y
        self.flag_active[idx] = True

    def _apply_water_pushback(self, row, col):
        cx, cy = self._cell_center(row, col)
        dx = self.ax - cx
        dy = self.ay - cy
        d = math.hypot(dx, dy)
        if d == 0.0:
            dx, dy = 1.0, 0.0
            d = 1.0

        # push back to last safe position and bounce outward
        self.ax, self.ay = self.last_safe_x, self.last_safe_y
        self.vx = (dx / d) * self.pushback_strength
        self.vy = (dy / d) * self.pushback_strength

    def _update_cracks(self):
        """
        Update crack timers and turn some cracked cells into water.
        If the cell under the agent becomes water this step, apply pushback.
        Returns True if a water collision happened here.
        """
        # spawn new crack
        if self.np_random.random() < self.crack_spawn_prob:
            candidates = np.argwhere(self.grid_state == 0)
            if len(candidates) > 0:
                row, col = candidates[self.np_random.integers(0, len(candidates))]
                self.grid_state[row, col] = 1
                self.grid_timer[row, col] = 0

        # advance timers for cracked cells
        cracked = self.grid_state == 1
        self.grid_timer[cracked] += 1
        to_water = cracked & (self.grid_timer >= self.crack_to_water_steps)

        water_collision = False

        # check if the cell under the agent is transitioning to water
        if np.any(to_water):
            cell = self._cell_from_pos(self.ax, self.ay)
            if cell is not None:
                row_a, col_a = cell
                if to_water[row_a, col_a]:
                    water_collision = True
                    self._apply_water_pushback(row_a, col_a)

        # apply transition
        self.grid_state[to_water] = 2

        # ensure flags are not on water
        for i in range(self.n_flags):
            if not self.flag_active[i]:
                continue
            cell = self._cell_from_pos(self.flag_x[i], self.flag_y[i])
            if cell is None:
                continue
            row, col = cell
            if self.grid_state[row, col] == 2:
                self._spawn_flag(i)

        return water_collision

    def _get_obs(self):
        ax_n = self.ax / self.width
        ay_n = self.ay / self.height
        vx_n = np.clip(self.vx / self.max_speed, -1.0, 1.0)
        vy_n = np.clip(self.vy / self.max_speed, -1.0, 1.0)

        flag_feats = []
        for i in range(self.n_flags):
            if self.flag_active[i]:
                fx = self.flag_x[i] / self.width
                fy = self.flag_y[i] / self.height
                active = 1.0
            else:
                fx = fy = active = 0.0
            flag_feats.extend([fx, fy, active])

        grid_flat = (self.grid_state.flatten().astype(np.float32) / 2.0).tolist()

        obs = np.array(
            [ax_n, ay_n, vx_n, vy_n] + flag_feats + grid_flat,
            dtype=np.float32,
        )
        return obs

    # ---------- gym API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # reset grid
        self.grid_state.fill(0)
        self.grid_timer.fill(0)

        # agent in middle of lake
        self.ax = self.area_x + self.area_size * 0.5
        self.ay = self.area_y + self.area_size * 0.5
        self.vx = self.vy = 0.0
        self.last_safe_x, self.last_safe_y = self.ax, self.ay

        # flags
        self.flag_active[:] = False
        for i in range(self.n_flags):
            self._spawn_flag(i)

        # clear score popups
        self._score_popups.clear()

        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # velocity with ice-like sliding
        self.vx = self.vx * self.friction + float(action[0]) * self.speed
        self.vy = self.vy * self.friction + float(action[1]) * self.speed

        # clamp max speed
        v_norm = math.hypot(self.vx, self.vy)
        if v_norm > self.max_speed:
            scale = self.max_speed / (v_norm + 1e-8)
            self.vx *= scale
            self.vy *= scale

        # move & clamp
        nx = self.ax + self.vx
        ny = self.ay + self.vy
        self.ax, self.ay = self._clamp_to_bounds(nx, ny, self.agent_r)

        water_collision_move = False
        collected_flags = 0

        # water collision / pushback from existing water
        cell = self._cell_from_pos(self.ax, self.ay)
        if cell is not None:
            row, col = cell
            if self.grid_state[row, col] == 2:
                water_collision_move = True
                self._apply_water_pushback(row, col)
            elif self.grid_state[row, col] == 0:
                # only update last_safe on solid ice, not cracked
                self.last_safe_x, self.last_safe_y = self.ax, self.ay
        else:
            self.last_safe_x, self.last_safe_y = self.ax, self.ay

        # collect flags (after any pushback)
        for i in range(self.n_flags):
            if not self.flag_active[i]:
                continue
            d = math.hypot(self.ax - self.flag_x[i], self.ay - self.flag_y[i])
            if d <= self.agent_r + self.flag_r:
                collected_flags += 1
                self.flag_active[i] = False

                self._spawn_flag(i)

                # spawn icy +1 popup just above the penguin
                popup_x = int(self.ax)
                popup_y = int(self.ay - self.agent_r - 4)
                self._score_popups.append(
                    {"x": popup_x, "y": popup_y, "start_step": self.steps}
                )

        # crack dynamics (may also cause a water collision under the agent)
        water_collision_crack = self._update_cracks()
        water_collision = water_collision_move or water_collision_crack

        # dense shaping w.r.t. nearest active flag
        if np.any(self.flag_active):
            dists = [
                math.hypot(self.ax - self.flag_x[i], self.ay - self.flag_y[i])
                for i in range(self.n_flags)
                if self.flag_active[i]
            ]
            min_dist = min(dists)
            dist_norm = np.clip(min_dist / (self.area_size/2), 0.0, 1.0)
        else:
            dist_norm = 1.0

        base_reward = 1.0 - dist_norm
        shaped = 0.02 * base_reward
        step_penalty = -0.01

        reward = shaped + step_penalty
        reward += 1.0 * collected_flags      # beacon collection
        if water_collision:
            reward -= 1.0                    # falling into/through water

        self.steps += 1
        terminated = False
        truncated = self.steps >= self.max_episode_steps

        info = {
            "collected_flags": bool(collected_flags),
            "collected_count": collected_flags,
            "water_collision": water_collision,
            "distance_to_target": dist_norm,
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    # ---------- rendering ----------
    def render(self):
        if self._surface is None:
            pygame.init()
            self._surface = pygame.Surface((self.width, self.height))

        if self._font_score is None:
            pygame.font.init()
            self._font_score = pygame.font.SysFont("arial", 15 if self.grid_size == 4 else 13)

        surf = self._surface
        surf.fill(self._bg_snow)

        area_rect = pygame.Rect(self.area_x, self.area_y, self.area_size, self.area_size)

        # draw ice grid (6x6) with state-based colors
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x0 = self.area_x + col * self.cell_w
                x1 = self.area_x + (col + 1) * self.cell_w
                y0 = self.area_y + row * self.cell_h
                y1 = self.area_y + (row + 1) * self.cell_h

                rect = pygame.Rect(
                    int(x0),
                    int(y0),
                    max(1, int(x1 - x0)),
                    max(1, int(y1 - y0)),
                )

                state = self.grid_state[row, col]
                if state == 0:
                    color = self._lake_ice
                elif state == 1:
                    color = self._lake_ice_cracked
                else:
                    color = self._lake_water

                pygame.draw.rect(surf, color, rect)

                if state == 1:
                    # simple crack pattern
                    pygame.draw.line(
                        surf,
                        self._crack_line,
                        rect.topleft,
                        rect.bottomright,
                        1,
                    )
                    pygame.draw.line(
                        surf,
                        self._crack_line,
                        rect.topright,
                        rect.bottomleft,
                        1,
                    )

        # grid lines
        for c in range(self.grid_size + 1):
            x = int(self.area_x + c * self.cell_w)
            pygame.draw.line(
                surf,
                self._grid_line,
                (x, self.area_y),
                (x, self.area_y + self.area_size),
                1,
            )
        for r in range(self.grid_size + 1):
            y = int(self.area_y + r * self.cell_h)
            pygame.draw.line(
                surf,
                self._grid_line,
                (self.area_x, y),
                (self.area_x + self.area_size, y),
                1,
            )

        # subtle, uneven snow edge around the lake
        edge_offset = 3
        # top / bottom edges
        for x in range(area_rect.left - 8, area_rect.right + 8, 16):
            # top edge bumps
            pygame.draw.circle(
                surf, self._bg_snow,
                (x, area_rect.top - edge_offset),
                6,
            )
            pygame.draw.circle(
                surf, self._bg_snow,
                (x + 5, area_rect.top - edge_offset + 2),
                4,
            )
            # bottom edge bumps
            pygame.draw.circle(
                surf, self._bg_snow,
                (x, area_rect.bottom + edge_offset),
                6,
            )
            pygame.draw.circle(
                surf, self._bg_snow,
                (x - 5, area_rect.bottom + edge_offset - 2),
                4,
            )

        # left / right edges
        for y in range(area_rect.top - 8, area_rect.bottom + 8, 16):
            # left edge bumps
            pygame.draw.circle(
                surf, self._bg_snow,
                (area_rect.left - edge_offset, y),
                6,
            )
            pygame.draw.circle(
                surf, self._bg_snow,
                (area_rect.left - edge_offset + 2, y + 5),
                4,
            )
            # right edge bumps
            pygame.draw.circle(
                surf, self._bg_snow,
                (area_rect.right + edge_offset, y),
                6,
            )
            pygame.draw.circle(
                surf, self._bg_snow,
                (area_rect.right + edge_offset - 2, y - 5),
                4,
            )

        # flags (scaled up a bit)
        for i in range(self.n_flags):
            if not self.flag_active[i]:
                continue
            fx = int(self.flag_x[i])
            fy = int(self.flag_y[i])

            pole_top = (fx, fy - (15 if self.grid_size == 4 else 11))   # slightly taller
            pole_bottom = (fx, fy + (4 if self.grid_size == 4 else 3))
            pygame.draw.line(surf, self._flag_pole, pole_bottom, pole_top, (3 if self.grid_size == 4 else 2))

            flag_pts = [
                pole_top,
                (pole_top[0] + (12 if self.grid_size == 4 else 8), pole_top[1] + (3 if self.grid_size == 4 else 2)),   # wider
                (pole_top[0], pole_top[1] + (9 if self.grid_size == 4 else 6)),       # a bit taller
            ]
            pygame.draw.polygon(surf, self._flag_color, flag_pts)

        # penguin agent
        ax = int(self.ax)
        ay = int(self.ay)
        r = self.agent_r

        # body (tall ellipse)
        body_rect = pygame.Rect(ax - r, ay - r, 2 * r, 2 * r + 3)
        pygame.draw.ellipse(surf, self._penguin_body, body_rect)
        pygame.draw.ellipse(surf, self._penguin_outline, body_rect, 1)

        # belly
        belly_rect = body_rect.inflate(-int(0.6 * r), -int(0.4 * r))
        pygame.draw.ellipse(surf, self._penguin_belly, belly_rect)

        # head
        head_r = int(r * 0.7)
        head_center = (ax, ay - r - head_r // 3)
        pygame.draw.circle(surf, self._penguin_body, head_center, head_r)
        pygame.draw.circle(surf, self._penguin_outline, head_center, head_r, 1)

        # eyes
        eye_dx = int(head_r * 0.35)
        eye_dy = int(head_r * 0.15)
        pygame.draw.circle(
            surf, self._penguin_eye,
            (head_center[0] - eye_dx, head_center[1] - eye_dy), 1
        )
        pygame.draw.circle(
            surf, self._penguin_eye,
            (head_center[0] + eye_dx, head_center[1] - eye_dy), 1
        )

        # beak (small triangle)
        beak_pts = [
            (head_center[0], head_center[1] + int(head_r * 0.2)),
            (head_center[0] - 3, head_center[1] + int(head_r * 0.6)),
            (head_center[0] + 3, head_center[1] + int(head_r * 0.6)),
        ]
        pygame.draw.polygon(surf, self._penguin_beak, beak_pts)

        # flippers
        flipper_h = int(r * 1.2)
        flipper_w = int(r * 0.7)
        left_flipper = pygame.Rect(
            ax - r - flipper_w // 2,
            ay - flipper_h // 4,
            flipper_w,
            flipper_h,
        )
        right_flipper = pygame.Rect(
            ax + r - flipper_w // 2,
            ay - flipper_h // 4,
            flipper_w,
            flipper_h,
        )
        pygame.draw.ellipse(surf, self._penguin_body, left_flipper)
        pygame.draw.ellipse(surf, self._penguin_body, right_flipper)

        # feet
        foot_y = ay + r + 2
        foot_w = int(r * 0.9)
        foot_h = int(r * 0.4)
        left_foot = pygame.Rect(ax - foot_w, foot_y, foot_w, foot_h)
        right_foot = pygame.Rect(ax, foot_y, foot_w, foot_h)
        pygame.draw.ellipse(surf, self._penguin_feet, left_foot)
        pygame.draw.ellipse(surf, self._penguin_feet, right_foot)

        # floating "+1" score popups
        if self._score_popups:
            updated = []
            for popup in self._score_popups:
                age = self.steps - popup["start_step"]
                if age < 0 or age > self._score_popup_duration:
                    continue

                t = age / float(self._score_popup_duration)
                y = popup["y"] - int(8 * t)
                x = popup["x"]
                alpha = int(255 * (1.0 - t))
                if alpha <= 0:
                    continue

                text_surf = self._font_score.render("+1", True, self._score_color)
                text_surf.set_alpha(alpha)
                rect = text_surf.get_rect(center=(x, y))

                # simple dark shadow for contrast
                shadow_surf = self._font_score.render("+1", True, (20, 40, 70))
                shadow_surf.set_alpha(alpha)
                shadow_rect = shadow_surf.get_rect(center=(x + 1, y + 1))

                surf.blit(shadow_surf, shadow_rect)
                surf.blit(text_surf, rect)

                updated.append(popup)

            self._score_popups = updated

        arr = np.transpose(np.array(pygame.surfarray.pixels3d(surf)), (1, 0, 2)).copy()
        return arr

    def close(self):
        if self._surface is not None:
            pygame.quit()
            self._surface = None


class FrozenLake6x6Env(FrozenLakeEnv):
    def __init__(self, max_episode_steps=500):
        super().__init__(max_episode_steps=max_episode_steps, grid_size=6, n_flags=2)


class FrozenLake5x5Env(FrozenLakeEnv):
    def __init__(self, max_episode_steps=500):
        super().__init__(max_episode_steps=max_episode_steps, grid_size=5, n_flags=2)


class FrozenLake4x4Env(FrozenLakeEnv):
    def __init__(self, max_episode_steps=500):
        super().__init__(max_episode_steps=max_episode_steps, grid_size=4, n_flags=1)
