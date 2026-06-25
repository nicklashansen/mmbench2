import gymnasium as gym
import numpy as np
import pygame
import math


class PotionEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, max_episode_steps=250, recipe=None):
        super().__init__()
        self.max_episode_steps = max_episode_steps

        # canvas / play area
        self.width, self.height = 224, 224
        # logical grid: 5 cols x 4 rows (playable)
        self.grid_cols = 5
        self.grid_rows = 4
        # one extra non-playable row at the top for background / UI
        self.reserved_top_rows = 1
        self.total_rows = self.grid_rows + self.reserved_top_rows

        self.area_size = 184
        self.area_x = (self.width - self.area_size) // 2
        self.area_y = (self.height - self.area_size) // 2

        self.cell_w = self.area_size / self.grid_cols
        self.cell_h = self.area_size / self.total_rows  # includes reserved row

        # agent (little witch)
        self.agent_r = 7
        self.speed = 2.5
        self.friction = 0.7
        self.max_speed = self.speed * 2.0

        # cauldron roughly centered (logical coords)
        self.cauldron_row = 1  # logical row index in [0, grid_rows-1]
        self.cauldron_col = self.grid_cols // 2
        self.cauldron_r = 12

        # ingredients (4 total)
        self.n_ingredients = 4
        self.max_recipe_len = 3

        # layout (0 = empty, 1 = cauldron, 2-6 = ingredient slots)
        self.layout_grid = np.zeros(
            (self.grid_rows, self.grid_cols), dtype=np.int32
        )
        self.ingredient_id_grid = -np.ones_like(self.layout_grid, dtype=np.int32)
        self.ingredient_positions = [None] * self.n_ingredients

        # table bounds for collision & rendering
        self._compute_table_bounds()
        self._setup_layout()

        # runtime state
        self.ax = self.ay = 0.0
        self.vx = self.vy = 0.0
        self.carrying = -1  # -1 = nothing, 0..3 = ingredient id

        # recipe info (in-order)
        self.base_recipe = list(recipe) if recipe is not None else [0, 2, 3]
        self.recipe = []
        self.collected = []  # bool flags per recipe element
        self.recipe_step = 0  # index into recipe
        self.brew_progress = 0.0
        self.brew_wrong = 0
        self.success = False
        self.steps = 0

        # reward scales
        self.pick_reward_scale = 0.3      # stage 1: move to correct ingredient
        self.carry_reward_scale = 0.4     # stage 2: move to cauldron with correct ingredient
        self.progress_per_step = 0.5      # stage 3: per-step bonus per completed ingredient
        self.completion_bonus = 10.0       # one-time bonus when recipe is fully completed

        # sparkles around cauldron when correct ingredient is added
        self._sparkles = []  # list of dicts: {'start_step', 'ing_id'}
        self._sparkle_duration = 24

        # completion visuals
        self._completion_step = None  # step index when recipe first completed
        self._completion_fade_frames = 30
        self._completion_max_alpha = 70
        self._completion_overlay_color = (120, 220, 160)

        # pygame visuals
        self._surface = None           # final (possibly shaken/flashed) surface
        self._world_surface = None     # base world (no shake/flash)
        self._font_small = None

        # shake / flash for wrong ingredient
        self._shake_frames = 0
        self._shake_intensity = 2
        self._flash_frames = 0
        self._flash_max_frames = 6

        # palette
        self._bg_wall = (245, 236, 222)
        self._floor_color = (220, 190, 160)
        self._floor_line = (205, 175, 145)
        self._table_color = (176, 140, 110)
        self._table_edge = (140, 105, 80)

        self._cauldron_body = (40, 40, 55)
        self._cauldron_rim = (25, 25, 40)
        self._cauldron_shadow = (30, 25, 35)
        self._steam_color = (235, 235, 245)

        self._witch_body = (70, 60, 90)
        self._witch_accent = (200, 180, 220)
        self._witch_hat = (55, 40, 80)
        self._witch_skin = (240, 210, 190)
        self._witch_boot = (50, 40, 60)

        # high-contrast ingredient colors (simple bottles)
        self._ingredient_colors = [
            (150, 210, 150),  # 0 herb - green
            (150, 190, 240),  # 1 blue vial
            (230, 170, 230),  # 2 crystal - purple/pink
            (235, 205, 140),  # 3 golden spice
        ]

        # observation: agent(2) + vel(2)
        #            + carrying (1 + n_ingredients one-hot)
        #            + recipe_ids(3) + recipe_mask(3) + recipe_progress(1)
        #            + layout_grid(5*4)
        obs_dim = (
            2
            + 2
            + (1 + self.n_ingredients)
            + self.max_recipe_len
            + self.max_recipe_len
            + 1
            + self.grid_rows * self.grid_cols
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # continuous 2D thrust in [-1,1]^2
        self.action_space = gym.spaces.Box(
            low=-np.ones(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )

    # ---------- helpers ----------
    def _compute_table_bounds(self):
        x0 = self.area_x + 4
        x1 = self.area_x + self.area_size - 4

        # top table (between physical row 1 and 2)
        top_y0 = self.area_y + self.cell_h + 2
        top_y1 = self.area_y + 2 * self.cell_h - 4

        # bottom table hugging the bottom row (physical last row)
        bottom_y0 = self.area_y + self.area_size - self.cell_h + 4
        bottom_y1 = self.area_y + self.area_size - 2

        self._top_table_bounds = (float(x0), float(top_y0), float(x1), float(top_y1))
        self._bottom_table_bounds = (float(x0), float(bottom_y0), float(x1), float(bottom_y1))

    def _setup_layout(self):
        self.layout_grid.fill(0)
        self.ingredient_id_grid.fill(-1)

        # cauldron in center (logical coords)
        self.layout_grid[self.cauldron_row, self.cauldron_col] = 1
        self.cauldron_cx, self.cauldron_cy = self._cell_center(
            self.cauldron_row, self.cauldron_col
        )

        # ingredients:
        positions = [
            (0, 0),  # id 0
            (0, 4),  # id 1
            (3, 0),  # id 2
            (3, 4),  # id 3
        ]
        for idx, (r, c) in enumerate(positions):
            self.layout_grid[r, c] = 2 + idx
            self.ingredient_id_grid[r, c] = idx
            self.ingredient_positions[idx] = (r, c)

    def _clamp_to_bounds(self, x, y, r):
        min_x = self.area_x + r + 1
        max_x = self.area_x + self.area_size - r - 1

        min_y = self.area_y + self.reserved_top_rows * self.cell_h + r + 1
        max_y = self.area_y + self.area_size - r - 1

        x = np.clip(x, min_x, max_x)
        y = np.clip(y, min_y, max_y)
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
        row_phys = int((y - self.area_y) / self.cell_h)

        if (
            row_phys < self.reserved_top_rows
            or row_phys >= self.total_rows
        ):
            return None

        row = row_phys - self.reserved_top_rows  # logical row
        col = max(0, min(self.grid_cols - 1, col))
        row = max(0, min(self.grid_rows - 1, row))
        return row, col

    def _cell_center(self, row, col):
        cx = self.area_x + (col + 0.5) * self.cell_w
        cy = self.area_y + (self.reserved_top_rows + row + 0.5) * self.cell_h
        return cx, cy

    def _push_out_of_rect(self, x0, y0, x1, y1):
        ax, ay, r = self.ax, self.ay, self.agent_r

        nx = min(max(ax, x0), x1)
        ny = min(max(ay, y0), y1)
        dx = ax - nx
        dy = ay - ny
        dist = math.hypot(dx, dy)

        if dist < r:
            if dist < 1e-6:
                mid_y = 0.5 * (y0 + y1)
                if ay < mid_y:
                    self.ay = y0 - r - 1
                else:
                    self.ay = y1 + r + 1
            else:
                overlap = r - dist + 0.5
                self.ax += (dx / dist) * overlap
                self.ay += (dy / dist) * overlap

            self.vx *= 0.2
            self.vy *= 0.2

    def _avoid_cauldron(self):
        dx = self.ax - self.cauldron_cx
        dy = self.ay - self.cauldron_cy
        dist = math.hypot(dx, dy)
        min_dist = self.cauldron_r + self.agent_r + 2.0
        if dist < 1e-6:
            dx, dy = 1.0, 0.0
            dist = 1.0
        if dist < min_dist:
            scale = min_dist / dist
            self.ax = self.cauldron_cx + dx * scale
            self.ay = self.cauldron_cy + dy * scale
            self.vx *= 0.0
            self.vy *= 0.0

    def _avoid_tables(self):
        self._push_out_of_rect(*self._top_table_bounds)
        self._push_out_of_rect(*self._bottom_table_bounds)

    def _emit_sparkles(self, ing_id):
        self._sparkles.append({"start_step": self.steps, "ing_id": int(ing_id)})

    def _trigger_wrong_fx(self):
        self._shake_frames = max(self._shake_frames, 10)
        self._flash_frames = self._flash_max_frames

    def _add_to_cauldron(self, ing_id):
        correct = False
        wrong = False

        if self.recipe_step >= len(self.recipe):
            wrong = True
            self.brew_wrong += 1
            return correct, wrong

        required = self.recipe[self.recipe_step]

        if ing_id == required:
            self.collected[self.recipe_step] = True
            self.recipe_step += 1
            self.brew_progress = self.recipe_step / float(len(self.recipe))
            correct = True
            self._emit_sparkles(ing_id)

            # mark recipe completion moment
            if (
                self.recipe_step == len(self.recipe)
                and self._completion_step is None
            ):
                self._completion_step = self.steps

        elif ing_id in self.recipe[: self.recipe_step]:
            correct = False
            wrong = False
        else:
            wrong = True
            self.brew_wrong += 1

        return correct, wrong

    def _get_obs(self):
        ax_n = self.ax / self.width
        ay_n = self.ay / self.height
        vx_n = np.clip(self.vx / self.max_speed, -1.0, 1.0)
        vy_n = np.clip(self.vy / self.max_speed, -1.0, 1.0)

        carry_oh = np.zeros(1 + self.n_ingredients, dtype=np.float32)
        if self.carrying < 0:
            carry_oh[0] = 1.0
        else:
            carry_oh[1 + int(self.carrying)] = 1.0

        sentinel = float(self.n_ingredients)  # "none" id
        recipe_ids = np.full(self.max_recipe_len, sentinel, dtype=np.float32)
        recipe_mask = np.zeros(self.max_recipe_len, dtype=np.float32)
        for i in range(min(len(self.recipe), self.max_recipe_len)):
            recipe_ids[i] = float(self.recipe[i])
            recipe_mask[i] = 1.0
        denom = max(1.0, float(self.n_ingredients - 1))
        recipe_ids_norm = recipe_ids / denom

        if len(self.recipe) > 0:
            step_norm = self.recipe_step / float(len(self.recipe))
        else:
            step_norm = 0.0

        layout_flat = (self.layout_grid.flatten().astype(np.float32) / 7.0).tolist()

        obs = np.array(
            [ax_n, ay_n, vx_n, vy_n]
            + carry_oh.tolist()
            + recipe_ids_norm.tolist()
            + recipe_mask.tolist()
            + [step_norm]
            + layout_flat,
            dtype=np.float32,
        )
        return obs

    # ---------- gym API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # random starting cell among interior rows (1 and 2) that are empty
        candidates = []
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                if self.layout_grid[r, c] == 0 and r in (1, 2):
                    candidates.append((r, c))
        assert candidates, "No empty cells for spawn."
        idx = self.np_random.integers(0, len(candidates))
        start_row, start_col = candidates[idx]
        self.ax, self.ay = self._cell_center(start_row, start_col)
        self.ax, self.ay = self._clamp_to_bounds(self.ax, self.ay, self.agent_r)

        self.vx = self.vy = 0.0
        self.carrying = -1

        self.recipe = list(self.base_recipe)
        self.collected = [False] * len(self.recipe)
        self.recipe_step = 0
        self.brew_progress = 0.0
        self.brew_wrong = 0
        self.success = False

        self._sparkles.clear()

        self._shake_frames = 0
        self._flash_frames = 0

        # reset completion visuals
        self._completion_step = None

        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # velocity with light inertia
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

        # avoid obstacles
        self._avoid_cauldron()
        self._avoid_tables()
        self.ax, self.ay = self._clamp_to_bounds(self.ax, self.ay, self.agent_r)

        ingredient_picked = False
        correct_add = False
        wrong_add = False
        completed_now = False

        # remaining required ingredients (for info, not reward directly)
        remaining_ids = [
            ing for ing, done in zip(self.recipe, self.collected) if not done
        ]

        # distances for logging / interactions
        dist_c = math.hypot(self.ax - self.cauldron_cx, self.ay - self.cauldron_cy)

        # front-of-cauldron delivery point
        deliver_tx = self.cauldron_cx
        deliver_ty = self.cauldron_cy + self.cauldron_r * 0.7
        dist_deliver = math.hypot(self.ax - deliver_tx, self.ay - deliver_ty)
        deliver_radius = self.cauldron_r + self.agent_r + 7.0

        # --- ingredient pickup (no direct reward here, just state) ---
        if self.carrying == -1:
            best_ing = -1
            best_dist = float("inf")
            pickup_radius = max(self.cell_w, self.cell_h) * 0.8
            for ing_id, pos in enumerate(self.ingredient_positions):
                r_i, c_i = pos
                cx, cy = self._cell_center(r_i, c_i)
                d = math.hypot(self.ax - cx, self.ay - cy)
                if d < pickup_radius and d < best_dist:
                    best_ing = ing_id
                    best_dist = d

            if best_ing >= 0:
                self.carrying = int(best_ing)
                ingredient_picked = True

        # --- dropping into cauldron ---
        if dist_deliver <= deliver_radius:
            if self.carrying != -1:
                ing_id = self.carrying
                self.carrying = -1

                prev_step = self.recipe_step  # for completion detection
                correct, wrong = self._add_to_cauldron(ing_id)
                correct_add = correct
                wrong_add = wrong
                if wrong_add:
                    # purely visual feedback, no reward effect
                    self._trigger_wrong_fx()

                # did we just finish the entire recipe this step?
                if (correct_add
                    and prev_step < len(self.recipe)
                    and self.recipe_step == len(self.recipe)):
                    completed_now = True
                    self.success = True

        r_pick = 0.0
        r_carry = 0.0

        # next required ingredient id (None if recipe complete)
        if self.recipe_step < len(self.recipe):
            next_ing_id = self.recipe[self.recipe_step]
        else:
            next_ing_id = None

        dist_to_next = None
        dist_norm_target = 1.0  # for logging: normalized distance to current target

        if next_ing_id is not None:
            # world position of next required ingredient
            r_next, c_next = self.ingredient_positions[next_ing_id]
            next_x, next_y = self._cell_center(r_next, c_next)
            dist_to_next = math.hypot(self.ax - next_x, self.ay - next_y)
            dist_next_norm = np.clip(dist_to_next / self.area_size, 0.0, 1.0)

            # stage 1: moving toward next ingredient (only if not carrying anything)
            if self.carrying == -1:
                r_pick = self.pick_reward_scale * (1.0 - dist_next_norm)
                dist_norm_target = dist_next_norm

            # stage 2: carrying correct ingredient to cauldron
            elif self.carrying == next_ing_id:
                dist_deliver_norm = np.clip(dist_deliver / self.area_size, 0.0, 1.0)
                # constant = max of stage 1 so there is no drop at pickup
                r_carry = self.pick_reward_scale + self.carry_reward_scale * (1.0 - dist_deliver_norm)
                dist_norm_target = dist_deliver_norm

            else:
                # carrying wrong ingredient -> no stage 1/2 shaping
                dist_norm_target = 1.0
        else:
            # recipe complete: no stage 1/2 shaping, only progress bonus below
            dist_norm_target = 1.0

        # stage 3: per-step progress bonus (k = recipe_step)
        r_progress = self.recipe_step * self.progress_per_step

        reward = r_pick + r_carry + r_progress

        # bonus for completing the recipe
        if self.success:
            reward = self.completion_bonus

        self.steps += 1
        terminated = False
        truncated = self.steps >= self.max_episode_steps

        info = {
            "pickup": bool(ingredient_picked),
            "correct_add": bool(correct_add),
            "wrong_add": bool(wrong_add),
            "recipe": list(self.recipe),
            "recipe_step": int(self.recipe_step),
            "carrying": int(self.carrying),
            "brew_progress": float(self.brew_progress),
            "distance_to_target": float(dist_norm_target),
            "dist_to_next_ingredient": float(dist_to_next if dist_to_next is not None else -1.0),
            "dist_to_cauldron_center": float(dist_c),
            "dist_to_delivery_point": float(dist_deliver),
            "r_pick": float(r_pick),
            "r_carry": float(r_carry),
            "r_progress": float(r_progress),
            "completed_now": bool(completed_now),
            "success": bool(self.success),
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    # ---------- rendering ----------
    def _liquid_color(self):
        t = self.steps * 0.12
        p = self.brew_progress
        r = 80 + int(60 * p + 20 * math.sin(t))
        g = 140 + int(40 * (1.0 - p) + 10 * math.cos(t * 0.7))
        b = 170 + int(30 * (p**2) + 15 * math.sin(t * 1.3))
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        return (r, g, b)

    def _draw_bottle(self, surf, ing_id, cx, cy):
        base_color = self._ingredient_colors[ing_id]
        bw = int(self.cell_w * 0.26)
        bh = int(self.cell_h * 0.6)
        x = int(cx - bw / 2)
        y = int(cy - bh / 2) + 2

        bottle_rect = pygame.Rect(x, y + 4, bw, int(bh * 0.7))
        neck_w = int(bw * 0.6)
        neck_h = int(bh * 0.25)
        neck_rect = pygame.Rect(
            int(cx - neck_w / 2), y, neck_w, neck_h
        )

        pygame.draw.rect(surf, base_color, bottle_rect, border_radius=4)
        pygame.draw.rect(surf, base_color, neck_rect, border_radius=3)
        highlight = bottle_rect.inflate(-bw // 2, -bottle_rect.height // 2)
        pygame.draw.rect(surf, (255, 255, 255), highlight, 1, border_radius=3)

    def _draw_cauldron(self, surf):
        cx, cy = int(self.cauldron_cx), int(self.cauldron_cy)

        shadow_rect = pygame.Rect(
            int(cx - self.cauldron_r * 1.4),
            int(cy + self.cauldron_r * 0.7),
            int(self.cauldron_r * 2.8),
            int(self.cauldron_r * 0.7),
        )
        pygame.draw.ellipse(surf, self._cauldron_shadow, shadow_rect)

        body_rect = pygame.Rect(
            int(cx - self.cauldron_r),
            int(cy - self.cauldron_r * 0.1),
            int(self.cauldron_r * 2),
            int(self.cauldron_r * 1.9),
        )
        pygame.draw.ellipse(surf, self._cauldron_body, body_rect)

        rim_rect = pygame.Rect(
            int(cx - self.cauldron_r * 1.1),
            int(cy - self.cauldron_r * 0.7),
            int(self.cauldron_r * 2.2),
            int(self.cauldron_r * 0.9),
        )
        pygame.draw.ellipse(surf, self._cauldron_rim, rim_rect)
        pygame.draw.ellipse(surf, (10, 10, 20), rim_rect, 1)

        liquid_rect = rim_rect.inflate(
            -int(self.cauldron_r * 0.5), -int(self.cauldron_r * 0.5)
        )
        pygame.draw.ellipse(surf, self._liquid_color(), liquid_rect)

        lx, ly = liquid_rect.center
        inner_r = liquid_rect.width * 0.22
        for i in range(3):
            phase = self.steps * 0.25 + i * 2.0
            jitter = 1.5 * math.sin(self.steps * 0.4 + i)
            bx = int(lx + (inner_r * 0.5 + jitter) * math.cos(phase))
            by = int(ly + (inner_r * 0.3 + jitter) * math.sin(phase))
            pygame.draw.circle(surf, (245, 250, 255), (bx, by), 2)

        for i in range(3):
            offset = (i - 1) * 6
            sway = int(2 * math.sin(self.steps * 0.2 + i))
            rise = int(2 * math.cos(self.steps * 0.3 + i))
            steam_rect = pygame.Rect(
                cx - 3 + offset + sway,
                int(liquid_rect.top - 16 + rise),
                6,
                18,
            )
            pygame.draw.ellipse(surf, self._steam_color, steam_rect)

    def _draw_sparkles(self, surf):
        if not self._sparkles:
            return
        cx, cy = int(self.cauldron_cx), int(self.cauldron_cy)
        updated = []
        for s in self._sparkles:
            age = self.steps - s["start_step"]
            if age < 0 or age > self._sparkle_duration:
                continue
            t = age / float(self._sparkle_duration)

            ing_id = s["ing_id"]
            br, bg, bb = self._ingredient_colors[ing_id]
            br = min(255, int(br + 70))
            bg = min(255, int(bg + 70))
            bb = min(255, int(bb + 70))

            alpha_scale = max(0.0, 1.0 - t)
            color = (
                int(br * alpha_scale + 255 * (1.0 - alpha_scale) * 0.1),
                int(bg * alpha_scale + 255 * (1.0 - alpha_scale) * 0.1),
                int(bb * alpha_scale + 255 * (1.0 - alpha_scale) * 0.1),
            )

            radius = self.cauldron_r + 6 + t * 6.0
            dot_r = 3

            for i in range(6):
                angle = 2.0 * math.pi * i / 6.0
                x = int(cx + radius * math.cos(angle))
                y = int(cy + radius * math.sin(angle) * 0.8)
                pygame.draw.circle(surf, color, (x, y), dot_r)
            updated.append(s)
        self._sparkles = updated

    def _draw_agent(self, surf):
        ax = int(self.ax)
        ay = int(self.ay)
        r = self.agent_r

        body_rect = pygame.Rect(ax - r, ay - r + 3, 2 * r, 2 * r + 2)
        pygame.draw.ellipse(surf, self._witch_body, body_rect)
        accent_rect = body_rect.inflate(-int(0.6 * r), -int(0.4 * r))
        pygame.draw.ellipse(surf, self._witch_accent, accent_rect)

        speed = math.hypot(self.vx, self.vy)
        phase = self.steps * 0.35
        amp = min(3.0, (speed / (self.max_speed + 1e-6)) * 3.0)

        foot_y_base = ay + r + 2
        offset = int(amp * math.sin(phase))

        foot_w = int(r * 0.9)
        foot_h = int(r * 0.4)

        left_foot = pygame.Rect(ax - foot_w, foot_y_base + offset, foot_w, foot_h)
        right_foot = pygame.Rect(ax, foot_y_base - offset, foot_w, foot_h)
        pygame.draw.ellipse(surf, self._witch_boot, left_foot)
        pygame.draw.ellipse(surf, self._witch_boot, right_foot)

        head_r = int(r * 0.7)
        head_center = (ax, ay - r)
        pygame.draw.circle(surf, self._witch_skin, head_center, head_r)

        brim_rect = pygame.Rect(ax - r, head_center[1] + 1, 2 * r, 3)
        pygame.draw.rect(surf, self._witch_hat, brim_rect)

        hat_pts = [
            (ax, head_center[1] - head_r - 2),
            (ax - head_r, head_center[1] + 1),
            (ax + head_r, head_center[1] + 1),
        ]
        pygame.draw.polygon(surf, self._witch_hat, hat_pts)

        eye_dx = int(head_r * 0.4)
        eye_y = head_center[1] - int(head_r * 0.1)
        pygame.draw.circle(surf, (20, 20, 30), (head_center[0] - eye_dx, eye_y), 1)
        pygame.draw.circle(surf, (20, 20, 30), (head_center[0] + eye_dx, eye_y), 1)

        if self.carrying >= 0:
            orb_color = self._ingredient_colors[self.carrying]
            orb_center = (ax, head_center[1] - head_r - 6)
            pygame.draw.circle(surf, orb_color, orb_center, 4)
            pygame.draw.circle(surf, (255, 255, 255), orb_center, 4, 1)

    def _draw_recipe_ui(self, surf):
        if self._font_small is None:
            return
        panel_rect = pygame.Rect(8, 8, 82, 40)

        # panel background
        pygame.draw.rect(surf, (250, 245, 240), panel_rect, border_radius=6)

        # border color depends on completion
        if len(self.recipe) > 0 and self.recipe_step == len(self.recipe):
            border_color = (90, 180, 110)
            border_width = 2
        else:
            border_color = (210, 180, 150)
            border_width = 1

        pygame.draw.rect(surf, border_color, panel_rect, border_width, border_radius=6)

        label = self._font_small.render("Recipe", True, (80, 60, 50))
        surf.blit(label, (panel_rect.x + 6, panel_rect.y + 4))

        cx0 = panel_rect.x + 18
        cy = panel_rect.y + 24
        step_spacing = 22

        for i in range(self.max_recipe_len):
            cx = cx0 + i * step_spacing
            if i < len(self.recipe):
                ing_id = self.recipe[i]
                base_color = self._ingredient_colors[ing_id]
                pygame.draw.circle(surf, base_color, (cx, cy), 6)

                if i < self.recipe_step:
                    border = (80, 150, 90)
                else:
                    border = (120, 100, 90)
                pygame.draw.circle(surf, border, (cx, cy), 6, 1)
            else:
                pygame.draw.circle(surf, (210, 210, 210), (cx, cy), 5, 1)

    def render(self):
        if self._surface is None:
            pygame.init()
            self._surface = pygame.Surface((self.width, self.height))
        if self._world_surface is None:
            self._world_surface = pygame.Surface((self.width, self.height))

        if self._font_small is None:
            pygame.font.init()
            self._font_small = pygame.font.SysFont("arial", 12)

        world = self._world_surface
        world.fill(self._bg_wall)

        floor_y0 = self.area_y + self.reserved_top_rows * self.cell_h
        floor_h = self.grid_rows * self.cell_h
        floor_rect = pygame.Rect(
            self.area_x,
            int(floor_y0),
            self.area_size,
            int(floor_h),
        )
        pygame.draw.rect(world, self._floor_color, floor_rect)

        for c in range(self.grid_cols + 1):
            x = int(self.area_x + c * self.cell_w)
            pygame.draw.line(
                world,
                self._floor_line,
                (x, floor_y0),
                (x, floor_y0 + floor_h),
                1,
            )
        for r in range(self.grid_rows + 1):
            y = int(floor_y0 + r * self.cell_h)
            pygame.draw.line(
                world,
                self._floor_line,
                (self.area_x, y),
                (self.area_x + self.area_size, y),
                1,
            )

        x0, y0, x1, y1 = self._top_table_bounds
        top_rect = pygame.Rect(int(x0), int(y0), int(x1 - x0), int(y1 - y0))
        x0b, y0b, x1b, y1b = self._bottom_table_bounds
        bottom_rect = pygame.Rect(int(x0b), int(y0b), int(x1b - x0b), int(y1b - y0b))

        pygame.draw.rect(world, self._table_color, top_rect, border_radius=4)
        pygame.draw.rect(world, self._table_color, bottom_rect, border_radius=4)
        pygame.draw.rect(world, self._table_edge, top_rect, 1, border_radius=4)
        pygame.draw.rect(world, self._table_edge, bottom_rect, 1, border_radius=4)

        for ing_id in range(self.n_ingredients):
            row, col = self.ingredient_positions[ing_id]
            cx, cy = self._cell_center(row, col)
            self._draw_bottle(world, ing_id, int(cx), int(cy))

        self._draw_cauldron(world)
        self._draw_sparkles(world)
        self._draw_agent(world)
        self._draw_recipe_ui(world)

        surf = self._surface

        dx = dy = 0
        if self._shake_frames > 0:
            dx = self.np_random.integers(-self._shake_intensity, self._shake_intensity + 1)
            dy = self.np_random.integers(-self._shake_intensity, self._shake_intensity + 1)
            self._shake_frames -= 1

        surf.fill(self._bg_wall)
        surf.blit(world, (dx, dy))

        # red flash for wrong ingredient
        if self._flash_frames > 0:
            alpha = int(120 * (self._flash_frames / float(self._flash_max_frames)))
            overlay = pygame.Surface((self.width, self.height))
            overlay.fill((255, 60, 60))
            overlay.set_alpha(alpha)
            surf.blit(overlay, (0, 0))
            self._flash_frames -= 1

        # green success overlay: fades in once recipe is complete, then stays
        if self._completion_step is not None:
            age = max(0, self.steps - self._completion_step)
            t = min(1.0, age / float(self._completion_fade_frames))
            alpha = int(self._completion_max_alpha * t)
            if alpha > 0:
                overlay = pygame.Surface((self.width, self.height))
                overlay.fill(self._completion_overlay_color)
                overlay.set_alpha(alpha)
                surf.blit(overlay, (0, 0))

        arr = np.transpose(np.array(pygame.surfarray.pixels3d(surf)), (1, 0, 2)).copy()
        return arr

    def close(self):
        if self._surface is not None:
            pygame.quit()
            self._surface = None
            self._world_surface = None


# --- recipe-specific variations ---

class PotionRecipe1Env(PotionEnv):
    def __init__(self, max_episode_steps=250):
        super().__init__(max_episode_steps=max_episode_steps, recipe=[0, 2, 3])


class PotionRecipe2Env(PotionEnv):
    def __init__(self, max_episode_steps=250):
        super().__init__(max_episode_steps=max_episode_steps, recipe=[1, 3, 0])


class PotionRecipe3Env(PotionEnv):
    def __init__(self, max_episode_steps=250):
        super().__init__(max_episode_steps=max_episode_steps, recipe=[2, 3, 1])


class PotionRecipe4Env(PotionEnv):
    def __init__(self, max_episode_steps=250):
        super().__init__(max_episode_steps=max_episode_steps, recipe=[3, 1, 2])


class PotionRecipe5Env(PotionEnv):
    def __init__(self, max_episode_steps=250):
        super().__init__(max_episode_steps=max_episode_steps, recipe=[3, 2, 1])


class PotionRecipe6Env(PotionEnv):
    def __init__(self, max_episode_steps=250):
        super().__init__(max_episode_steps=max_episode_steps, recipe=[2, 1, 3])


class PotionRecipe7Env(PotionEnv):
    def __init__(self, max_episode_steps=250):
        super().__init__(max_episode_steps=max_episode_steps, recipe=[0, 1, 2])


class PotionRecipe8Env(PotionEnv):
    def __init__(self, max_episode_steps=250):
        super().__init__(max_episode_steps=max_episode_steps, recipe=[0, 1])


class PotionRecipe9Env(PotionEnv):
    def __init__(self, max_episode_steps=250):
        super().__init__(max_episode_steps=max_episode_steps, recipe=[3, 2])


class PotionRecipe10Env(PotionEnv):
    def __init__(self, max_episode_steps=250):
        super().__init__(max_episode_steps=max_episode_steps, recipe=[2, 0])
