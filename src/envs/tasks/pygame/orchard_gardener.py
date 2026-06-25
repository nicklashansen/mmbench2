import gymnasium as gym
import numpy as np
import pygame
import math


class OrchardGardenerEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, max_episode_steps=500, n_trees=5):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.n_trees = n_trees

        # canvas / play area
        self.width, self.height = 224, 224
        self.area_size = 184
        self.area_x = (self.width - self.area_size) // 2
        self.area_y = (self.height - self.area_size) // 2

        # agent / trees / base
        self.agent_r = 8
        self.tree_r = 10
        self.base_r = 12
        self.speed = 4.0

        # base position
        self.base_x = self.area_x + self.area_size * 0.5
        self.base_y = self.area_y + self.area_size * 0.8

        # runtime state
        self.ax = self.ay = 0.0
        self.vx = self.vy = 0.0
        self.tree_x = np.zeros(self.n_trees, dtype=np.float32)
        self.tree_y = np.zeros(self.n_trees, dtype=np.float32)
        self.tree_progress = np.zeros(self.n_trees, dtype=np.float32)   # growth
        self.tree_ripe_time = np.zeros(self.n_trees, dtype=np.int32)    # steps since ripe
        self.carrying = False
        self.carrying_from = -1
        self.steps = 0

        # tree growth + rot dynamics
        self._growth_rate = 1.0 / 500.0
        self._ripe_threshold = 0.7
        self._rot_steps = 110               # steps a tree can stay ripe before rotting
        self._rot_penalty = 0.4             # per-tree penalty when fruit rots

        # falling (rotted) apples
        self._falling_apples = []           # list of dicts: {'x','y','start_step'}
        self._fall_duration = 30            # steps for fall + fade

        # observation: agent(2) + vel(2) + base(2) + carrying(1) + trees(n*3)
        obs_dim = 7 + 3 * self.n_trees
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # continuous 2D thrust
        self.action_space = gym.spaces.Box(
            low=-np.ones(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )

        # visuals
        self._surface = None
        self._bg_color = (38, 10, 38)
        self._grass_color = (32, 80, 36)
        self._grass_dark = (22, 60, 28)
        self._border_color = (30, 56, 32)

        # grass grid texture (randomized per episode)
        self._grass_cell_size = 16
        self._grass_rows = 0
        self._grass_cols = 0
        self._grass_cells = None  # (rows, cols, 3) uint8

        # cabin colors
        self._cabin_body = (140, 90, 50)
        self._cabin_body_dark = (90, 55, 30)
        self._cabin_roof = (70, 40, 25)
        self._cabin_foundation = (120, 120, 130)
        self._cabin_door = (70, 45, 25)
        self._cabin_door_dark = (40, 25, 15)
        self._chimney_color = (110, 80, 70)
        self._smoke_color = (190, 190, 190)

        # tree / apple colors
        self._trunk_color = (110, 80, 55)
        self._canopy_empty = (40, 70, 40)
        self._canopy_growing = (60, 110, 60)
        self._canopy_ripe = (60, 130, 60)
        self._canopy_overripe = (55, 80, 45)
        self._apple_color = (200, 40, 40)
        self._apple_rot_color = (130, 70, 110)  # brownish / purplish

        # agent + basket colors
        self._agent_body = (170, 95, 55)
        self._agent_pants = (145, 75, 35)
        self._agent_head = (152, 116, 92)
        self._basket_color = (150, 98, 52)

        # score popups ("+1" on delivery)
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
                self.area_x + 8, self.area_x + self.area_size - 8
            )
            y = self.np_random.uniform(
                self.area_y + 8, self.area_y + self.area_size - 8
            )
            if min_dist_from is not None:
                d = math.hypot(x - min_dist_from[0], y - min_dist_from[1])
                if d < min_d:
                    continue
            return x, y
        return self.area_x + self.area_size / 2, self.area_y + self.area_size / 2

    def _init_tree_line_positions(self):
        """Place trees in a horizontal line near the top of the clearing."""
        margin = 14
        left_x = self.area_x + margin
        right_x = self.area_x + self.area_size - margin
        # line towards the top, inside the playable area
        y_line = int(self.area_y + 48)

        if self.n_trees == 1:
            xs = [0.5 * (left_x + right_x)]
        else:
            xs = [
                left_x + i * (right_x - left_x) / (self.n_trees - 1)
                for i in range(self.n_trees)
            ]

        for i in range(self.n_trees):
            self.tree_x[i] = xs[i]
            self.tree_y[i] = y_line

    def _nearest_ripe_tree(self):
        ripe_idxs = np.where(self.tree_progress >= self._ripe_threshold)[0]
        if len(ripe_idxs) == 0:
            return None
        dists = [
            math.hypot(self.ax - self.tree_x[i], self.ay - self.tree_y[i])
            for i in ripe_idxs
        ]
        j = int(np.argmin(dists))
        return int(ripe_idxs[j])

    def _nearest_tree(self):
        if self.n_trees == 0:
            return None
        dists = [
            math.hypot(self.ax - self.tree_x[i], self.ay - self.tree_y[i])
            for i in range(self.n_trees)
        ]
        j = int(np.argmin(dists))
        return j

    def _get_obs(self):
        ax_n = self.ax / self.width
        ay_n = self.ay / self.height
        bx_n = self.base_x / self.width
        by_n = self.base_y / self.height
        vx_n = self.vx / self.speed
        vy_n = self.vy / self.speed
        carrying_flag = 1.0 if self.carrying else 0.0

        trees_feat = []
        for i in range(self.n_trees):
            tx_n = self.tree_x[i] / self.width
            ty_n = self.tree_y[i] / self.height
            ripeness = float(np.clip(self.tree_progress[i], 0.0, 1.0))
            trees_feat.extend([tx_n, ty_n, ripeness])

        obs = np.array(
            [ax_n, ay_n, vx_n, vy_n, bx_n, by_n, carrying_flag] + trees_feat,
            dtype=np.float32,
        )
        return obs

    def _generate_grass_texture(self):
        """Random grass color texture with correlated variation."""
        cell = self._grass_cell_size
        # number of cells to cover the area
        self._grass_cols = int(math.ceil(self.area_size / cell))
        self._grass_rows = int(math.ceil(self.area_size / cell))

        base = np.array(self._grass_color, dtype=np.float32)
        # initial random offsets
        noise = self.np_random.uniform(
            -20.0, 20.0, size=(self._grass_rows, self._grass_cols)
        )

        # simple smoothing
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
                r = base[0] + 0.8 * offset
                g = base[1] + 1.2 * offset
                b = base[2] + 0.6 * offset
                cells[i, j, 0] = int(np.clip(r, 15, 110))
                cells[i, j, 1] = int(np.clip(g, 40, 150))
                cells[i, j, 2] = int(np.clip(b, 15, 110))

        self._grass_cells = cells

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
        self.carrying_from = -1

        # orchard trees: line formation with random initial ripeness
        self._init_tree_line_positions()
        self.tree_progress[:] = self.np_random.uniform(0.0, 1.0, size=self.n_trees)
        self.tree_ripe_time[:] = 0

        # visuals state
        self._generate_grass_texture()
        self._falling_apples.clear()
        self._score_popups.clear()

        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # trees grow every step
        self.tree_progress = (self.tree_progress + self._growth_rate).astype(np.float32)

        # update ripeness timers and rot events
        rot_count = 0
        for i in range(self.n_trees):
            if self.tree_progress[i] >= self._ripe_threshold:
                self.tree_ripe_time[i] += 1
            else:
                self.tree_ripe_time[i] = 0

            if self.tree_ripe_time[i] == self._rot_steps:
                # fruit fully rots and falls: spawn falling apples, reset tree
                cx = self.tree_x[i]
                cy = self.tree_y[i]
                offsets = [(-4, -2), (3, 0), (-1, 4)]
                for dx, dy in offsets:
                    self._falling_apples.append(
                        {"x": cx + dx, "y": cy + dy, "start_step": self.steps}
                    )
                self.tree_progress[i] = 0.0
                self.tree_ripe_time[i] = 0
                rot_count += 1

        # instant velocity from action
        self.vx, self.vy = action * self.speed

        # move & clamp
        nx = self.ax + float(self.vx)
        ny = self.ay + float(self.vy)
        self.ax, self.ay = self._clamp_to_bounds(nx, ny, self.agent_r)

        harvested = False
        delivered = False

        # harvest apples from ripe tree
        if not self.carrying:
            for i in range(self.n_trees):
                if self.tree_progress[i] < self._ripe_threshold:
                    continue
                d = math.hypot(self.ax - self.tree_x[i], self.ay - self.tree_y[i])
                if d <= self.agent_r + self.tree_r:
                    self.carrying = True
                    self.carrying_from = i
                    self.tree_progress[i] = 0.0
                    self.tree_ripe_time[i] = 0
                    harvested = True
                    break

        # deliver basket to cabin
        if self.carrying:
            d_base = math.hypot(self.ax - self.base_x, self.ay - self.base_y)
            if d_base <= self.agent_r + self.base_r:
                delivered = True
                self.carrying = False
                self.carrying_from = -1

                # spawn "+1" popup just above the agent
                popup_x = int(self.ax)
                popup_y = int(self.ay - self.agent_r - 4)
                self._score_popups.append(
                    {"x": popup_x, "y": popup_y, "start_step": self.steps}
                )

        # dense shaping: distance to current "target"
        if self.carrying:
            target_x, target_y = self.base_x, self.base_y
        else:
            idx = self._nearest_ripe_tree()
            if idx is None:
                idx = self._nearest_tree()
            if idx is None:
                target_x, target_y = self.base_x, self.base_y
            else:
                target_x, target_y = self.tree_x[idx], self.tree_y[idx]

        dist = math.hypot(self.ax - target_x, self.ay - target_y)
        dist_norm = np.clip(dist / self.area_size, 0.0, 1.0)
        base_reward = 1.0 - dist_norm  # in [0,1]

        # small shaping + small step penalty
        shaped = 0.2 * base_reward
        step_penalty = -0.005
        reward = shaped + step_penalty

        # event bonuses / penalties
        if harvested:
            reward += 0.3
        if delivered:
            reward += 1.5
        if rot_count > 0:
            reward -= self._rot_penalty * rot_count

        info = {
            "harvested": harvested,
            "delivered": delivered,
            "rotted_count": rot_count,
            "num_ripe": int(np.sum(self.tree_progress >= self._ripe_threshold)),
            "distance_to_target": dist_norm,
        }

        self.steps += 1
        terminated = False
        truncated = self.steps >= self.max_episode_steps

        return self._get_obs(), float(reward), terminated, truncated, info

    # ---------- rendering ----------
    def _draw_cabin(self, surf):
        bx = int(self.base_x)
        by = int(self.base_y)
        cabin_w = self.base_r * 2 + 4
        cabin_h = self.base_r * 2
        cabin_rect = pygame.Rect(
            bx - cabin_w // 2, by - cabin_h // 2, cabin_w, cabin_h
        )

        # concrete foundation
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

        # door
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
        # knob
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

        # simple smoke animation
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

    def _draw_tree(self, surf, x, y, progress, ripe_time):
        # trunk
        trunk_h = int(self.tree_r * 1.4)
        trunk_w = max(3, self.tree_r // 2)
        trunk_rect = pygame.Rect(
            int(x - trunk_w / 2),
            int(y),
            trunk_w,
            trunk_h,
        )
        pygame.draw.rect(surf, self._trunk_color, trunk_rect)

        # canopy centered at (x, y), smoothly interpolated by growth
        cx = int(x)
        cy = int(y)

        p = float(np.clip(progress, 0.0, 1.0))

        def lerp_color(c1, c2, t):
            t = max(0.0, min(1.0, t))
            return (
                int(c1[0] + (c2[0] - c1[0]) * t),
                int(c1[1] + (c2[1] - c1[1]) * t),
                int(c1[2] + (c2[2] - c1[2]) * t),
            )

        if p <= self._ripe_threshold:
            denom = self._ripe_threshold if self._ripe_threshold > 1e-6 else 1.0
            t_grow = p / denom
            canopy_color = lerp_color(self._canopy_empty, self._canopy_growing, t_grow)
        else:
            denom = (1.0 - self._ripe_threshold) if (1.0 - self._ripe_threshold) > 1e-6 else 1.0
            t_ripe = (p - self._ripe_threshold) / denom
            canopy_color = lerp_color(self._canopy_growing, self._canopy_ripe, t_ripe)

        # as tree stays ripe, nudge canopy slightly towards "overripe"
        if ripe_time > 0:
            over_t = min(1.0, ripe_time / float(self._rot_steps))
            canopy_color = lerp_color(canopy_color, self._canopy_overripe, over_t)

        pygame.draw.circle(surf, canopy_color, (cx, cy), self.tree_r)

        # apples: slowly rot in color while ripe, before they fall
        if progress >= self._ripe_threshold:
            phase = min(1.0, ripe_time / float(max(self._rot_steps, 1)))

            # keep them mostly red for the first ~30% of ripe time,
            # then accelerate rot so they darken notably before falling.
            if phase < 0.3:
                t_rot = 0.2 * (phase / 0.3)  # 0 → 0.2
            else:
                t_rot = 0.2 + 0.8 * ((phase - 0.3) / 0.7)  # 0.2 → 1.0
            t_rot = max(0.0, min(1.0, t_rot))

            apple_color = lerp_color(self._apple_color, self._apple_rot_color, t_rot)

            offsets = [(-4, -2), (3, 0), (-1, 4)]
            for dx, dy in offsets:
                pygame.draw.circle(
                    surf, apple_color,
                    (cx + dx, cy + dy),
                    max(2, self.tree_r // 5),
                )

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

        # grass with correlated variation
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

        # grid lines
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

        # cabin
        self._draw_cabin(surf)

        # trees
        for i in range(self.n_trees):
            self._draw_tree(
                surf, self.tree_x[i], self.tree_y[i],
                self.tree_progress[i], self.tree_ripe_time[i]
            )

        # falling (rotted) apples
        if self._falling_apples:
            alive = []
            for a in self._falling_apples:
                age = self.steps - a["start_step"]
                if age < 0 or age > self._fall_duration:
                    continue

                t = age / float(self._fall_duration)
                fall_dist = 12
                x = int(a["x"])
                y = int(a["y"] + fall_dist * t)
                alpha = int(255 * (1.0 - t))
                if alpha <= 0:
                    continue

                radius = max(2, self.tree_r // 5)
                apple_surf = pygame.Surface(
                    (radius * 2 + 2, radius * 2 + 2), pygame.SRCALPHA
                )
                pygame.draw.circle(
                    apple_surf,
                    (*self._apple_rot_color, alpha),
                    (radius + 1, radius + 1),
                    radius,
                )
                surf.blit(apple_surf, (x - radius - 1, y - radius - 1))
                alive.append(a)

            self._falling_apples = alive

        # agent + basket
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

        # head
        head_r = max(3, r // 2)
        head_center = (ax, ay - body_h // 2 - head_r + 2)
        pygame.draw.circle(surf, self._agent_head, head_center, head_r)

        # basket with apples when carrying (closer to body, apples above rim)
        if self.carrying:
            basket_w = r
            basket_h = max(3, r // 2)
            # tuck basket near the right side of the body
            basket_center_x = body_rect.right - basket_w // 3
            basket_center_y = body_rect.centery + 1
            basket_rect = pygame.Rect(
                basket_center_x - basket_w // 2,
                basket_center_y - basket_h // 2,
                basket_w,
                basket_h,
            )
            pygame.draw.rect(surf, self._basket_color, basket_rect, border_radius=2)

            # apples slightly above the basket rim
            apple_radius = max(2, r // 4)
            apple_y1 = basket_rect.top - apple_radius // 2
            apple_y2 = basket_rect.top - apple_radius // 3

            pygame.draw.circle(
                surf,
                self._apple_color,
                (basket_rect.left + basket_w // 3, apple_y1),
                apple_radius,
            )
            pygame.draw.circle(
                surf,
                self._apple_color,
                (basket_rect.left + 2 * basket_w // 3, apple_y2),
                apple_radius,
            )

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
