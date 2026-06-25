import gymnasium as gym
import numpy as np
import pygame
import math


class ForagingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, max_episode_steps=500, n_coins=3):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.n_coins = n_coins

        # canvas / play area
        self.width, self.height = 224, 224
        self.area_size = 184
        self.area_x = (self.width - self.area_size) // 2
        self.area_y = (self.height - self.area_size) // 2

        # agent / lumber / base
        self.agent_r = 8
        self.coin_r = 6          # visual scale for logs
        self.base_r = 12
        self.speed = 4.0

        # base position
        self.base_x = self.area_x + self.area_size * 0.5
        self.base_y = self.area_y + self.area_size * 0.8

        # runtime state
        self.ax = self.ay = 0.0
        self.vx = self.vy = 0.0
        self.coin_x = np.zeros(self.n_coins, dtype=np.float32)
        self.coin_y = np.zeros(self.n_coins, dtype=np.float32)
        self.coin_active = np.zeros(self.n_coins, dtype=bool)
        self.carrying = False
        self.carrying_idx = -1
        self.steps = 0

        # observation: agent(2) + vel(2) + base(2) + carrying(1) + lumber(n*3)
        obs_dim = 7 + 3 * self.n_coins
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # continuous 2D thrust in [-1,1]^2
        self.action_space = gym.spaces.Box(
            low=-np.ones(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )

        # visuals: "cabin in the woods" palette
        self._surface = None
        self._bg_color = (12, 16, 20)       # background / sky
        self._grass_color = (32, 80, 36)    # base grass tone
        self._grass_dark = (22, 60, 28)     # grid line color
        self._border_color = (190, 200, 210)

        # grass grid texture (larger cells with correlated variation)
        self._grass_cell_size = 16          # larger cells
        self._grass_rows = 0
        self._grass_cols = 0
        self._grass_cells = None            # (rows, cols, 3) uint8

        # cabin colors
        self._cabin_body = (140, 90, 50)
        self._cabin_body_dark = (90, 55, 30)
        self._cabin_roof = (70, 40, 25)
        self._cabin_foundation = (120, 120, 130)
        self._cabin_door = (70, 45, 25)
        self._cabin_door_dark = (40, 25, 15)
        self._chimney_color = (110, 80, 70)
        self._smoke_color = (190, 190, 190)

        # lumber colors
        self._log_main = (150, 98, 52)
        self._log_edge = (100, 65, 35)
        self._log_inner = (190, 135, 70)

        # agent colors (little human) – slightly darker skin tone
        self._agent_body = (70, 120, 200)   # shirt
        self._agent_pants = (40, 60, 110)
        self._agent_head = (190, 160, 120)

        # score popups (for "+1" when delivering lumber)
        self._score_popups = []  # list of dicts: {'x','y','start_step'}
        self._score_popup_duration = 20
        self._score_color = (80, 240, 120)
        self._font_score = None

    # ---------- helpers ----------
    def _clamp_to_bounds(self, x, y, r):
        x = np.clip(x, self.area_x + r + 1, self.area_x + self.area_size - r - 1)
        y = np.clip(y, self.area_y + r + 1, self.area_y + self.area_size - r - 1)
        return x, y

    def _sample_point(self, min_dist_from=None, min_d=0.0):
        for _ in range(4000):
            x = self.np_random.uniform(
                self.area_x + 6, self.area_x + self.area_size - 6
            )
            y = self.np_random.uniform(
                self.area_y + 6, self.area_y + self.area_size - 6
            )
            if min_dist_from is not None:
                d = math.hypot(x - min_dist_from[0], y - min_dist_from[1])
                if d < min_d:
                    continue
            return x, y
        return self.area_x + self.area_size / 2, self.area_y + self.area_size / 2

    def _respawn_coin(self, idx):
        # logs spawn away from base a bit
        self.coin_x[idx], self.coin_y[idx] = self._sample_point(
            min_dist_from=(self.base_x, self.base_y),
            min_d=self.area_size * 0.2,
        )
        self.coin_active[idx] = True

    def _nearest_active_coin(self):
        active_idxs = np.where(self.coin_active)[0]
        if len(active_idxs) == 0:
            return None
        dists = [
            math.hypot(self.ax - self.coin_x[i], self.ay - self.coin_y[i])
            for i in active_idxs
        ]
        j = int(np.argmin(dists))
        return int(active_idxs[j])

    def _get_obs(self):
        # normalize positions to [0,1]
        ax_n = self.ax / self.width
        ay_n = self.ay / self.height
        bx_n = self.base_x / self.width
        by_n = self.base_y / self.height
        vx_n = self.vx / self.speed
        vy_n = self.vy / self.speed
        carrying_flag = 1.0 if self.carrying else 0.0

        coins_feat = []
        for i in range(self.n_coins):
            if self.coin_active[i]:
                cx = self.coin_x[i] / self.width
                cy = self.coin_y[i] / self.height
                active = 1.0
            else:
                cx = cy = active = 0.0
            coins_feat.extend([cx, cy, active])

        obs = np.array(
            [ax_n, ay_n, vx_n, vy_n, bx_n, by_n, carrying_flag] + coins_feat,
            dtype=np.float32,
        )
        return obs

    def _generate_grass_texture(self):
        cell = self._grass_cell_size
        # number of cells to cover the area, allow partial cells at edges
        self._grass_cols = int(math.ceil(self.area_size / cell))
        self._grass_rows = int(math.ceil(self.area_size / cell))

        base = np.array(self._grass_color, dtype=np.float32)
        # random per-cell color offsets
        noise = self.np_random.uniform(
            -20.0, 20.0, size=(self._grass_rows, self._grass_cols)
        )

        # simple smoothing: average with neighbors (correlated variation)
        smooth = np.zeros_like(noise)
        for i in range(self._grass_rows):
            for j in range(self._grass_cols):
                acc = 0.0
                count = 0
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        ii = i + di
                        jj = j + dj
                        if 0 <= ii < self._grass_rows and 0 <= jj < self._grass_cols:
                            acc += noise[ii, jj]
                            count += 1
                smooth[i, j] = acc / max(1, count)

        cells = np.zeros((self._grass_rows, self._grass_cols, 3), dtype=np.uint8)
        for i in range(self._grass_rows):
            for j in range(self._grass_cols):
                offset = smooth[i, j]
                # map smoothed noise offset onto per-channel color deltas
                r = base[0] + 0.8 * offset
                g = base[1] + 1.2 * offset
                b = base[2] + 0.6 * offset
                cells[i, j, 0] = int(np.clip(r, 15, 110))
                cells[i, j, 1] = int(np.clip(g, 40, 150))
                cells[i, j, 2] = int(np.clip(b, 15, 110))

        self._grass_cells = cells

    def _draw_lumber_stack(self, surf, cx, cy, scale=1.0):
        """
        Draws a 3-log stack centered roughly at (cx, cy).
        scale=1.0: ground logs; scale<1: carried logs.
        """
        r = self.coin_r

        # base log (largest)
        base_len = int(round(scale * 3.0 * r))
        base_h   = int(round(scale * 1.2 * r))

        # second log: smaller than base
        second_len = int(round(scale * 0.85 * 3.0 * r))
        second_h   = int(round(scale * 0.85 * 1.2 * r))

        # third log: smaller still
        third_len  = int(round(scale * 0.7 * 3.0 * r))
        third_h    = int(round(scale * 0.7 * 1.2 * r))

        # base log
        base_rect = pygame.Rect(
            cx - base_len // 2,
            cy - base_h // 2,
            base_len,
            base_h,
        )
        pygame.draw.rect(surf, self._log_main, base_rect, border_radius=2)
        pygame.draw.rect(surf, self._log_edge, base_rect, width=1, border_radius=2)
        pygame.draw.line(
            surf,
            self._log_inner,
            (base_rect.left + 3, base_rect.top + 2),
            (base_rect.left + 3, base_rect.bottom - 2),
        )
        pygame.draw.line(
            surf,
            self._log_inner,
            (base_rect.right - 4, base_rect.top + 2),
            (base_rect.right - 4, base_rect.bottom - 2),
        )

        # second log on top of base
        second_rect = pygame.Rect(
            cx - second_len // 2,
            base_rect.top - second_h + 2,
            second_len,
            second_h,
        )
        pygame.draw.rect(surf, self._log_main, second_rect, border_radius=2)
        pygame.draw.rect(surf, self._log_edge, second_rect, width=1, border_radius=2)

        # third log on top of second
        third_rect = pygame.Rect(
            cx - third_len // 2,
            second_rect.top - third_h + 2,
            third_len,
            third_h,
        )
        pygame.draw.rect(surf, self._log_main, third_rect, border_radius=2)
        pygame.draw.rect(surf, self._log_edge, third_rect, width=1, border_radius=2)

        return base_rect, second_rect, third_rect

    # ---------- gym API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # agent near base
        self.ax, self.ay = self._sample_point(
            min_dist_from=(self.base_x, self.base_y),
            min_d=self.area_size * 0.05,
        )
        self.vx = self.vy = 0.0
        self.carrying = False
        self.carrying_idx = -1

        # logs
        for i in range(self.n_coins):
            self._respawn_coin(i)

        # regenerate grass texture per episode for visual robustness
        self._generate_grass_texture()

        # clear score popups
        self._score_popups.clear()

        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # instant velocity from action
        self.vx, self.vy = action * self.speed

        # move & clamp
        nx = self.ax + float(self.vx)
        ny = self.ay + float(self.vy)
        self.ax, self.ay = self._clamp_to_bounds(nx, ny, self.agent_r)

        pickup = False
        delivered = False

        # pickup lumber
        if not self.carrying:
            for i in range(self.n_coins):
                if not self.coin_active[i]:
                    continue
                d = math.hypot(self.ax - self.coin_x[i], self.ay - self.coin_y[i])
                if d <= self.agent_r + self.coin_r:
                    self.carrying = True
                    self.carrying_idx = i
                    self.coin_active[i] = False
                    pickup = True
                    break

        # deliver to base
        if self.carrying:
            d_base = math.hypot(self.ax - self.base_x, self.ay - self.base_y)
            if d_base <= self.agent_r + self.base_r:
                delivered = True
                idx = self.carrying_idx
                self.carrying = False
                self.carrying_idx = -1
                if idx >= 0:
                    self._respawn_coin(idx)

                # spawn score popup "+1" slightly above the agent
                popup_x = int(self.ax)
                popup_y = int(self.ay - self.agent_r - 4)
                self._score_popups.append(
                    {"x": popup_x, "y": popup_y, "start_step": self.steps}
                )

        # dense shaping: distance to current "target"
        if self.carrying:
            target_x, target_y = self.base_x, self.base_y
        else:
            idx = self._nearest_active_coin()
            if idx is None:
                target_x, target_y = self.base_x, self.base_y
            else:
                target_x, target_y = self.coin_x[idx], self.coin_y[idx]

        dist = math.hypot(self.ax - target_x, self.ay - target_y)
        dist_norm = np.clip(dist / self.area_size, 0.0, 1.0)
        base_reward = 1.0 - dist_norm  # in [0,1]

        # small shaping + small step penalty
        shaped = 0.2 * base_reward
        step_penalty = -0.005
        reward = shaped + step_penalty

        # event bonuses dominate objectives
        if pickup:
            reward += 0.3
        if delivered:
            reward += 1.5

        info = {
            "pickup": pickup,
            "delivered": delivered,
            "distance_to_target": dist_norm,
        }

        self.steps += 1
        terminated = False
        truncated = self.steps >= self.max_episode_steps

        return self._get_obs(), float(reward), terminated, truncated, info

    # ---------- rendering ----------
    def render(self):
        if self._surface is None:
            pygame.init()
            self._surface = pygame.Surface((self.width, self.height))

        if self._font_score is None:
            pygame.font.init()
            self._font_score = pygame.font.SysFont("arial", 10)

        surf = self._surface

        # background
        surf.fill(self._bg_color)
        area_rect = pygame.Rect(self.area_x, self.area_y, self.area_size, self.area_size)

        # grass cells with correlated color variation
        if self._grass_cells is None:
            self._generate_grass_texture()

        cell = self._grass_cell_size
        for i in range(self._grass_rows):
            for j in range(self._grass_cols):
                x = self.area_x + j * cell
                y = self.area_y + i * cell
                w = min(cell, self.area_x + self.area_size - x)
                h = min(cell, self.area_y + self.area_size - y)
                if w <= 0 or h <= 0:
                    continue
                color = tuple(int(c) for c in self._grass_cells[i, j])
                rect = pygame.Rect(x, y, w, h)
                pygame.draw.rect(surf, color, rect)

        # grid lines aligned with playable area
        for x in range(self.area_x, self.area_x + self.area_size, cell):
            pygame.draw.line(
                surf, self._grass_dark, (x, self.area_y),
                (x, self.area_y + self.area_size)
            )
        pygame.draw.line(
            surf, self._grass_dark,
            (self.area_x + self.area_size, self.area_y),
            (self.area_x + self.area_size, self.area_y + self.area_size),
        )

        for y in range(self.area_y, self.area_y + self.area_size, cell):
            pygame.draw.line(
                surf, self._grass_dark, (self.area_x, y),
                (self.area_x + self.area_size, y)
            )
        pygame.draw.line(
            surf, self._grass_dark,
            (self.area_x, self.area_y + self.area_size),
            (self.area_x + self.area_size, self.area_y + self.area_size),
        )

        # border
        pygame.draw.rect(
            surf,
            self._border_color,
            area_rect,
            width=2,
        )

        # trees along top edge (evenly spaced)
        tree_spacing_target = 24
        n_trees = max(2, int(self.area_size // tree_spacing_target) + 1)
        margin = 10
        left_x = self.area_x + margin
        right_x = self.area_x + self.area_size - margin
        for k in range(n_trees):
            tx = int(round(left_x + k * (right_x - left_x) / (n_trees - 1)))
            trunk = pygame.Rect(tx - 2, self.area_y - 4, 4, 10)
            pygame.draw.rect(surf, (80, 55, 35), trunk)
            foliage_pts = [
                (tx, self.area_y - 18),
                (tx - 8, self.area_y - 4),
                (tx + 8, self.area_y - 4),
            ]
            pygame.draw.polygon(surf, (30, 90, 40), foliage_pts)

        # cabin (base)
        bx = int(self.base_x)
        by = int(self.base_y)
        cabin_w = self.base_r * 2 + 4
        cabin_h = self.base_r * 2
        cabin_rect = pygame.Rect(
            bx - cabin_w // 2, by - cabin_h // 2, cabin_w, cabin_h
        )

        # concrete foundation under cabin
        foundation_h = 4
        foundation_rect = pygame.Rect(
            cabin_rect.left - 2,
            cabin_rect.bottom,
            cabin_rect.width + 4,
            foundation_h,
        )
        pygame.draw.rect(surf, self._cabin_foundation, foundation_rect)

        # cabin body
        pygame.draw.rect(surf, self._cabin_body, cabin_rect)
        pygame.draw.rect(surf, self._cabin_body_dark, cabin_rect, width=1)

        # door on cabin
        door_w = int(cabin_w * 0.25)
        door_h = int(cabin_h * 0.55)
        door_rect = pygame.Rect(
            bx - door_w // 2,
            cabin_rect.bottom - door_h,
            door_w,
            door_h,
        )
        pygame.draw.rect(surf, self._cabin_door, door_rect)
        pygame.draw.rect(surf, self._cabin_door_dark, door_rect, width=1)
        # door knob
        knob_x = door_rect.right - 3
        knob_y = door_rect.centery
        pygame.draw.circle(surf, (210, 200, 160), (knob_x, knob_y), 1)

        # roof
        roof_y = cabin_rect.top
        roof_pts = [
            (cabin_rect.left - 2, roof_y),
            (cabin_rect.right + 2, roof_y),
            (bx, roof_y - 10),
        ]
        pygame.draw.polygon(surf, self._cabin_roof, roof_pts)

        # chimney
        chimney_w = 4
        chimney_h = 8
        chimney_rect = pygame.Rect(
            cabin_rect.right - chimney_w - 4,
            roof_y - chimney_h,
            chimney_w,
            chimney_h,
        )
        pygame.draw.rect(surf, self._chimney_color, chimney_rect)

        # animated smoke puffs
        puff_base_x = chimney_rect.centerx
        puff_base_y = chimney_rect.top - 2
        for i in range(3):
            t = self.steps * 0.2 + i * 0.4
            base_offset_y = -4 * i
            osc_y = -2.0 * math.sin(t)
            osc_x = 1.0 * math.sin(t * 0.7) * 0.5
            cx = puff_base_x + int(osc_x)
            cy = puff_base_y + int(base_offset_y + osc_y)
            radius = max(1, 3 - i // 1)
            pygame.draw.circle(surf, self._smoke_color, (cx, cy), radius)

        # lumber stacks on the ground
        for i in range(self.n_coins):
            if not self.coin_active[i]:
                continue
            cx = int(self.coin_x[i])
            cy = int(self.coin_y[i])
            self._draw_lumber_stack(surf, cx, cy, scale=1.0)

        # agent: human-like sprite
        ax = int(self.ax)
        ay = int(self.ay)
        r = self.agent_r

        # legs
        leg_h = max(3, r // 2)
        leg_w = max(2, r // 2)
        legs_rect = pygame.Rect(ax - leg_w, ay + r - leg_h, leg_w * 2, leg_h)
        pygame.draw.rect(surf, self._agent_pants, legs_rect)

        # body
        body_h = r + 2
        body_w = r
        body_rect = pygame.Rect(ax - body_w // 2, ay - body_h // 2, body_w, body_h)
        pygame.draw.rect(surf, self._agent_body, body_rect)

        # head (darker skin tone)
        head_r = max(3, r // 2)
        head_center = (ax, ay - body_h // 2 - head_r + 2)
        pygame.draw.circle(surf, self._agent_head, head_center, head_r)

        # carried lumber: same 3-log stack, scaled down
        if self.carrying:
            stack_scale = 0.6
            stack_cx = head_center[0]
            stack_cy = head_center[1] - head_r - int(self.coin_r * stack_scale)
            self._draw_lumber_stack(surf, stack_cx, stack_cy, scale=stack_scale)

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
                surf.blit(text_surf, rect)

                updated.append(popup)

            self._score_popups = updated

        arr = np.transpose(np.array(pygame.surfarray.pixels3d(surf)), (1, 0, 2)).copy()
        return arr

    def close(self):
        if self._surface is not None:
            pygame.quit()
            self._surface = None
