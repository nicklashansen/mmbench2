import gymnasium as gym
import numpy as np
import pygame
import math


class DungeonExplorerEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, max_episode_steps=250, layout=None):
        super().__init__()
        self.max_episode_steps = max_episode_steps

        # canvas / play area
        self.width, self.height = 224, 224
        self.area_size = 184
        self.area_x = (self.width - self.area_size) // 2
        self.area_y = (self.height - self.area_size) // 2

        # zoom factor for camera ( >1 = zoom in )
        self.zoom = 1.3

        # agent
        self.agent_r = 6
        self.speed = 2.0
        self.friction = 0.0  # no sliding
        self.max_speed = self.speed * 2.0

        # dungeon layouts (0 = floor, 1 = wall)
        self.preset_layout = layout
        self._layouts = self._build_layouts()
        self.grid_size = self._layouts[0].shape[0]
        self.cell_w = self.area_size / self.grid_size
        self.cell_h = self.area_size / self.grid_size

        # runtime state
        self.ax = self.ay = 0.0     # world coordinates
        self.vx = self.vy = 0.0
        self.steps = 0

        # last movement direction (for flipping and lantern shift)
        self.dir_x = 1.0
        self.dir_y = 0.0

        self.layout = self._layouts[0].copy()
        # mini-map memory: -1 unseen, 0 floor, 1 wall
        self.mini_map = np.full(self.layout.shape, -1, dtype=np.int32)
        self.visited = np.zeros(self.layout.shape, dtype=bool)
        self._fov_mask = np.zeros(self.layout.shape, dtype=bool)
        self._num_floor_cells = int(np.sum(self.layout == 0))

        self.chest_row = self.chest_col = 0
        self.chest_x = self.chest_y = 0.0
        self.chest_open = False

        # sparkles for treasure opening
        self._sparkles = []

        # observation: [agent(2) + vel(2) + chest_rel(2) + chest_open(1) + step_frac(1)] + mini_map (grid_size^2)
        map_dim = self.grid_size * self.grid_size
        obs_dim = 8 + map_dim
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # continuous 2D thrust in [-1,1]^2
        self.action_space = gym.spaces.Box(
            low=-np.ones(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )

        # visuals: warmer cave-like palette
        self._bg_color = (8, 6, 12)
        self._floor_color = (80, 60, 40)
        self._wall_color = (35, 25, 18)
        self._seen_dim_color = (20, 15, 12)

        # explorer + lantern
        self._explorer_outline = (20, 10, 8)
        self._explorer_cloak = (90, 55, 30)
        self._explorer_body = (120, 85, 55)
        self._explorer_head = (230, 200, 170)
        self._explorer_eye = (10, 5, 5)
        self._explorer_boots = (45, 30, 20)
        self._lantern_color = (255, 225, 150)
        self._lantern_glow_color = (255, 235, 180)

        # chest + glow
        self._chest_color_closed = (170, 120, 40)
        self._chest_color_open = (230, 190, 80)
        self._glow_color = (255, 225, 150)

        # pygame surface + font
        self._surface = None
        self._font_text = None

    # ---------- layouts ----------
    def _build_layouts(self):
        # 5 fixed 9x9 layouts, '#' = wall, '.' = floor
        raw_layouts = [
            [
                "#########",
                "#####...#",
                "#####.#.#",
                "####..#.#",
                "###..##.#",
                "#...###.#",
                "###.....#",
                "#####.###",
                "#########",
            ],
            [
                "#########",
                "#...#####",
                "#......##",
                "#.#...###",
                "#.##.####",
                "#......##",
                "#..#..###",
                "####.####",
                "#########",
            ],
            [
                "#########",
                "##......#",
                "###.##..#",
                "#.......#",
                "##..#.###",
                "#...#...#",
                "###...###",
                "####...##",
                "#########",
            ],
            [
                "#########",
                "##..#..##",
                "#...#...#",
                "###...###",
                "###...###",
                "#.......#",
                "#..###..#",
                "#########",
                "#########",
            ],
            [
                "#########",
                "#########",
                "#.......#",
                "#.#####.#",
                "#.#####.#",
                "#.#####.#",
                "#.......#",
                "#########",
                "#########",
            ],
            [
                "#########",
                "###.#####",
                "###...###",
                "#####.###",
                "###.....#",
                "#####.###",
                "#####.###",
                "#####.###",
                "#########",
            ],
            [
                "#########",
                "##.######",
                "##....###",
                "#..##.###",
                "##.##...#",
                "##.....##",
                "##.######",
                "##.######",
                "#########",
            ],
            [
                "#########",
                "#....####",
                "####..###",
                "####.####",
                "####.####",
                "###.....#",
                "###.##.##",
                "###....##",
                "#########",
            ],
            [
                "#########",
                "##......#",
                "##.####.#",
                "##......#",
                "#..###.##",
                "#.####.##",
                "###....##",
                "####.####",
                "#########",
            ],
            [
                "#########",
                "###....##",
                "#.##.####",
                "#.##.####",
                "#.......#",
                "#.###.###",
                "#.##..###",
                "#....####",
                "#########",
            ],
        ]
        layouts = []
        for lines in raw_layouts:
            h, w = len(lines), len(lines[0])
            grid = np.zeros((h, w), dtype=np.int32)
            for r, line in enumerate(lines):
                for c, ch in enumerate(line):
                    grid[r, c] = 1 if ch == "#" else 0
            layouts.append(grid)
        return layouts

    # ---------- helpers ----------
    def _cell_center(self, row, col):
        cx = self.area_x + (col + 0.5) * self.cell_w
        cy = self.area_y + (row + 0.5) * self.cell_h
        return cx, cy

    def _pos_to_cell(self, x, y):
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

    def _random_floor_cell(self):
        floor_cells = np.argwhere(self.layout == 0)
        idx = self.np_random.integers(0, len(floor_cells))
        return tuple(floor_cells[idx])

    def _update_visibility(self):
        """
        Field-of-view:
          - BFS radius in grid space (max_radius),
          - walls block vision (you see the wall but not behind it),
          - revealed cells get written into mini_map and FOV mask.
        After chest is opened, everything is always visible.
        """
        if self.chest_open:
            # after chest opens, full knowledge & full FOV
            self._fov_mask[:, :] = True
            self.mini_map[:, :] = self.layout
            return

        self._fov_mask[:, :] = False

        cell = self._pos_to_cell(self.ax, self.ay)
        if cell is None:
            return

        r0, c0 = cell
        max_radius = 3  # FOV radius in cells

        visited = np.zeros_like(self.layout, dtype=bool)
        queue = [(r0, c0, 0)]

        while queue:
            r, c, d = queue.pop(0)
            if visited[r, c]:
                continue
            visited[r, c] = True

            self._fov_mask[r, c] = True
            self.mini_map[r, c] = self.layout[r, c]

            if d >= max_radius:
                continue

            # walls are visible but do not propagate FOV behind them
            if self.layout[r, c] == 1:
                continue

            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.grid_size and 0 <= cc < self.grid_size:
                    if not visited[rr, cc]:
                        queue.append((rr, cc, d + 1))

    def _spawn_sparkles(self):
        self._sparkles.clear()
        # simple radial sparkles around the chest
        for i in range(16):
            angle = 2 * math.pi * i / 16.0
            self._sparkles.append(
                {
                    "angle": angle,
                    "start_step": self.steps,
                    "life": 30,
                }
            )

    def _reveal_entire_map(self):
        """Reveal all cells in the mini-map and mark entire dungeon as FOV."""
        self.mini_map[:, :] = self.layout
        self._fov_mask[:, :] = True

    def _get_obs(self):
        # normalize positions to [0,1]
        ax_n = self.ax / self.width
        ay_n = self.ay / self.height

        vx_n = np.clip(self.vx / self.max_speed, -1.0, 1.0)
        vy_n = np.clip(self.vy / self.max_speed, -1.0, 1.0)

        # chest relative vector, normalized by dungeon size
        dx = (self.chest_x - self.ax) / self.area_size
        dy = (self.chest_y - self.ay) / self.area_size
        dx = np.clip(dx, -1.0, 1.0)
        dy = np.clip(dy, -1.0, 1.0)

        chest_open = 1.0 if self.chest_open else 0.0
        step_frac = float(self.steps) / float(self.max_episode_steps)

        map_flat = self.mini_map.astype(np.float32).reshape(-1)

        obs = np.concatenate(
            [
                np.array(
                    [ax_n, ay_n, vx_n, vy_n, dx, dy, chest_open, step_frac],
                    dtype=np.float32,
                ),
                map_flat,
            ]
        )
        return obs

    # ---------- gym API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # choose one of the 5 layouts
        if self.preset_layout is not None:
            idx = self.preset_layout
        else:
            idx = self.np_random.integers(0, len(self._layouts))
        self.layout = self._layouts[idx].copy()

        self.grid_size = self.layout.shape[0]
        self.cell_w = self.area_size / self.grid_size
        self.cell_h = self.area_size / self.grid_size

        self.mini_map = np.full(self.layout.shape, -1, dtype=np.int32)
        self.visited = np.zeros(self.layout.shape, dtype=bool)
        self._fov_mask = np.zeros_like(self.layout, dtype=bool)
        self._num_floor_cells = int(np.sum(self.layout == 0))

        # sample agent start and chest positions on floor cells
        start_r, start_c = self._random_floor_cell()
        self.ax, self.ay = self._cell_center(start_r, start_c)
        self.vx = self.vy = 0.0
        self.dir_x = 1.0
        self.dir_y = 0.0

        while True:
            cr, cc = self._random_floor_cell()
            if cr != start_r or cc != start_c:
                break
        self.chest_row, self.chest_col = cr, cc
        self.chest_x, self.chest_y = self._cell_center(cr, cc)
        self.chest_open = False
        self._sparkles.clear()

        # mark start cell as visited (no reward)
        self.visited[start_r, start_c] = True
        self.steps = 0

        # reveal initial field-of-view in mini-map (and keep as memory)
        self._update_visibility()

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # no sliding: velocity is directly proportional to action each step
        self.vx = float(action[0]) * self.speed
        self.vy = float(action[1]) * self.speed

        # update facing direction if moving
        if abs(self.vx) > 0.05 or abs(self.vy) > 0.05:
            self.dir_x = self.vx
            self.dir_y = self.vy

        # clamp max speed (safety)
        v_norm = math.hypot(self.vx, self.vy)
        if v_norm > self.max_speed:
            scale = self.max_speed / (v_norm + 1e-8)
            self.vx *= scale
            self.vy *= scale

        # propose new position (world coordinates)
        nx = self.ax + self.vx
        ny = self.ay + self.vy

        # clamp to dungeon bounds
        nx = float(
            np.clip(
                nx,
                self.area_x + self.agent_r,
                self.area_x + self.area_size - self.agent_r,
            )
        )
        ny = float(
            np.clip(
                ny,
                self.area_y + self.agent_r,
                self.area_y + self.area_size - self.agent_r,
            )
        )

        # wall collision: if new cell is wall, stay put and stop velocity
        new_cell = self._pos_to_cell(nx, ny)
        if new_cell is not None:
            r_new, c_new = new_cell
            if self.layout[r_new, c_new] == 1:
                nx, ny = self.ax, self.ay
                self.vx = self.vy = 0.0

        self.ax, self.ay = nx, ny

        # exploration reward (new floor cell visited)
        explored_new = 0
        cell = self._pos_to_cell(self.ax, self.ay)
        if cell is not None:
            r, c = cell
            if self.layout[r, c] == 0 and not self.visited[r, c]:
                self.visited[r, c] = True
                explored_new = 1

        # update mini-map visibility (memory accumulates over episode)
        self._update_visibility()

        # treasure chest opening
        opened_now = False
        dist_to_chest = math.hypot(self.ax - self.chest_x, self.ay - self.chest_y)
        if (not self.chest_open) and dist_to_chest <= 0.4 * max(self.cell_w, self.cell_h):
            self.chest_open = True
            opened_now = True
            self._spawn_sparkles()
            self._reveal_entire_map()

        # has the chest been discovered (its cell revealed on mini-map)?
        chest_revealed = self.mini_map[self.chest_row, self.chest_col] != -1

        # dense distance reward ONLY after treasure is revealed (but before/while opening)
        dist_shaping = 0.0
        if chest_revealed and not self.chest_open:
            # map distance to [0, 1] using diagonal as max
            max_d = math.hypot(self.area_size, self.area_size)
            dist_norm = np.clip(dist_to_chest / (max_d/2), 0.0, 1.0)
            dist_shaping = 0.1 * (1.0 - dist_norm)

        # exploration coverage bonus
        if self._num_floor_cells > 0:
            coverage = float(np.count_nonzero(self.visited & (self.layout == 0))) / float(
                self._num_floor_cells
            )
        else:
            coverage = 0.0

        # big reward for opening the chest
        if self.chest_open:
            reward = 2.0

        else: # reward: exploration-focused shaping
            reward = -0.001  # per-step penalty

            if explored_new:
                reward += 0.1 * explored_new  # reward for discovering new floor cells
            if opened_now:
                reward += 1.0  # big bonus for finding the treasure
            if chest_revealed:
                reward += 0.03  # constant bonus after treasure is discovered

            # small coverage-based shaping to encourage exploring more of the map
            reward += 0.05 * coverage
            reward += dist_shaping  # only active when chest has been revealed

        self.steps += 1
        terminated = False
        truncated = self.steps >= self.max_episode_steps

        info = {
            "explored_new": bool(explored_new),
            "chest_open": self.chest_open,
            "opened_this_step": opened_now,
            "chest_revealed": bool(chest_revealed),
            "dist_to_chest": dist_to_chest,
            "dist_shaping": dist_shaping,
            "coverage": coverage,
            "success": bool(self.chest_open),
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    # ---------- rendering ----------
    def render(self):
        if self._surface is None:
            pygame.init()
            self._surface = pygame.Surface((self.width, self.height))

        if self._font_text is None:
            pygame.font.init()
            self._font_text = pygame.font.SysFont("arial", 18, bold=True)

        surf = self._surface
        surf.fill(self._bg_color)

        # camera center (agent stays here on screen)
        cam_cx = self.width / 2.0
        cam_cy = self.height / 2.0

        # agent cell for FOV ref
        cell = self._pos_to_cell(self.ax, self.ay)
        if cell is not None:
            ar, ac = cell
        else:
            ar, ac = 0, 0

        # smooth flicker used by lantern + chest glows / overlay
        phase1 = math.sin(0.18 * self.steps)
        phase2 = math.sin(0.47 * self.steps + 1.3)
        flicker = 1.0 + 0.25 * phase1 + 0.15 * phase2
        flicker = max(0.5, min(1.3, flicker))

        # draw dungeon tiles with camera-follow and lighting
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                val = self.mini_map[r, c]
                if val == -1:
                    # unseen: keep dark
                    continue

                world_x0 = self.area_x + c * self.cell_w
                world_y0 = self.area_y + r * self.cell_h

                screen_x0 = int(cam_cx + (world_x0 - self.ax) * self.zoom)
                screen_y0 = int(cam_cy + (world_y0 - self.ay) * self.zoom)

                rect = pygame.Rect(
                    screen_x0,
                    screen_y0,
                    max(1, int(self.cell_w * self.zoom) + 1),
                    max(1, int(self.cell_h * self.zoom) + 1),
                )

                # skip if completely off-screen
                if rect.right < 0 or rect.left > self.width or rect.bottom < 0 or rect.top > self.height:
                    continue

                base_color = self._wall_color if self.layout[r, c] == 1 else self._floor_color

                # brightness logic
                if self.chest_open:
                    # after opening chest, everything is fully "in FOV"
                    if self.layout[r, c] == 0:
                        base_intensity = 1.0
                    else:
                        base_intensity = 0.9
                else:
                    if self._fov_mask[r, c]:
                        base_intensity = 1.0
                    else:
                        # memory cells outside current FOV
                        if self.layout[r, c] == 0:
                            base_intensity = 0.50  # walkable
                        else:
                            base_intensity = 0.30  # wall

                tile_phase = math.sin(0.12 * self.steps + (r * 5 + c * 11))
                intensity = base_intensity + 0.06 * tile_phase
                intensity = max(0.15, min(1.0, intensity))

                color = tuple(int(ch * intensity) for ch in base_color)
                pygame.draw.rect(surf, color, rect)

        # determine if chest is in current FOV (grid-based)
        if self.chest_open:
            chest_in_fov = True
        else:
            chest_in_fov = self._fov_mask[self.chest_row, self.chest_col]

        # draw treasure chest (only if its cell has been revealed on the mini-map)
        if self.mini_map[self.chest_row, self.chest_col] != -1:
            # world -> screen
            cx_screen = cam_cx + (self.chest_x - self.ax) * self.zoom
            cy_screen = cam_cy + (self.chest_y - self.ay) * self.zoom

            chest_rect = pygame.Rect(
                0, 0,
                int(self.cell_w * 0.5 * self.zoom),
                int(self.cell_h * 0.4 * self.zoom),
            )
            chest_rect.center = (int(cx_screen), int(cy_screen) + 2)

            chest_color = self._chest_color_open if self.chest_open else self._chest_color_closed

            # dim chest if discovered but outside current FOV before opening
            if not self.chest_open and not chest_in_fov:
                chest_color = tuple(int(ch * 0.4) for ch in chest_color)

            pygame.draw.rect(surf, chest_color, chest_rect, border_radius=3)

            lid_rect = chest_rect.copy()
            lid_rect.height = max(3, lid_rect.height // 3)
            lid_rect.y -= 3
            lid_color = tuple(min(255, ch + 30) for ch in chest_color)
            pygame.draw.rect(surf, lid_color, lid_rect, border_radius=3)

            # simple keyhole
            key_w = max(2, chest_rect.width // 8)
            key_h = max(3, chest_rect.height // 2)
            key_rect = pygame.Rect(0, 0, key_w, key_h)
            key_rect.center = chest_rect.center
            pygame.draw.rect(surf, (0, 0, 0), key_rect, border_radius=1)

            # glow + sparkles if opened, with smooth radial alpha gradient
            if self.chest_open:
                glow_radius = int(self.cell_w * self.zoom)
                glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)

                base_alpha = 100
                for y in range(glow_radius * 2):
                    dy = y - glow_radius
                    for x in range(glow_radius * 2):
                        dx = x - glow_radius
                        dist = math.hypot(dx, dy) / glow_radius
                        if dist > 1.0:
                            continue
                        falloff = (1.0 - dist) ** 2  # smooth radial
                        alpha = int(base_alpha * falloff * flicker)
                        if alpha <= 0:
                            continue
                        glow_surf.set_at((x, y), (*self._glow_color, alpha))

                surf.blit(glow_surf, (int(cx_screen) - glow_radius, int(cy_screen) - glow_radius))

        # draw sparkles (bigger radius) when chest is open
        if self.chest_open and self._sparkles:
            new_sparkles = []
            for sp in self._sparkles:
                age = self.steps - sp["start_step"]
                life = sp["life"]
                if age < 0 or age > life:
                    continue
                t = age / float(life)
                radius = self.cell_w * 1.2 * t
                world_sx = self.chest_x + math.cos(sp["angle"]) * radius
                world_sy = self.chest_y + math.sin(sp["angle"]) * radius * 0.7

                sx = cam_cx + (world_sx - self.ax) * self.zoom
                sy = cam_cy + (world_sy - self.ay) * self.zoom

                alpha = int(255 * (1.0 - t))
                if alpha <= 0:
                    continue

                s_surf = pygame.Surface((6, 6), pygame.SRCALPHA)
                pygame.draw.circle(s_surf, (255, 240, 190, alpha), (3, 3), 3)
                surf.blit(s_surf, (int(sx) - 3, int(sy) - 3))
                new_sparkles.append(sp)
            self._sparkles = new_sparkles

        # ---------- draw explorer with directional lantern ----------
        ax_i, ay_i = int(cam_cx), int(cam_cy)
        r = self.agent_r

        facing_right = self.dir_x >= 0.0

        # cloak / body
        cloak_rect = pygame.Rect(ax_i - r, ay_i - r, 2 * r, 2 * r + 2)
        pygame.draw.ellipse(surf, self._explorer_cloak, cloak_rect)
        pygame.draw.ellipse(surf, self._explorer_outline, cloak_rect, 1)

        body_rect = cloak_rect.inflate(-int(0.6 * r), -int(0.3 * r))
        pygame.draw.ellipse(surf, self._explorer_body, body_rect)

        # head
        head_r = int(r * 0.7)
        head_center = (ax_i, ay_i - r - head_r // 3)
        pygame.draw.circle(surf, self._explorer_head, head_center, head_r)
        pygame.draw.circle(surf, self._explorer_outline, head_center, head_r, 1)

        # eye: flip horizontally with facing direction
        eye_offset = int(head_r * 0.25)
        eye_x = head_center[0] + (eye_offset if facing_right else -eye_offset)
        pygame.draw.circle(
            surf,
            self._explorer_eye,
            (eye_x, head_center[1]),
            1,
        )

        # feet / boots with simple walk animation
        foot_y_base = ay_i + r + 3
        foot_w = int(r * 0.6)
        foot_h = max(2, int(r * 0.4))
        speed_mag = math.hypot(self.vx, self.vy)

        if speed_mag > 0.05:
            walk_phase = 0.3 * self.steps
            amp = 2
            left_offset = int(math.sin(walk_phase) * amp)
            right_offset = int(math.sin(walk_phase + math.pi) * amp)
        else:
            left_offset = right_offset = 0

        left_foot_rect = pygame.Rect(
            ax_i - foot_w,
            foot_y_base + left_offset,
            foot_w,
            foot_h,
        )
        right_foot_rect = pygame.Rect(
            ax_i,
            foot_y_base + right_offset,
            foot_w,
            foot_h,
        )

        pygame.draw.ellipse(surf, self._explorer_boots, left_foot_rect)
        pygame.draw.ellipse(surf, self._explorer_boots, right_foot_rect)

        # lantern in hand, oriented + shifted by movement direction
        base_offset_x = int(r * 1.4)
        lantern_offset_x = base_offset_x if facing_right else -base_offset_x

        base_offset_y = int(r * 0.2)
        vertical_amp = int(r * 0.6)
        lantern_offset_y = base_offset_y
        if abs(self.dir_y) > 0.05:
            if self.dir_y < 0:  # moving up (y decreases)
                lantern_offset_y -= vertical_amp
            else:               # moving down
                lantern_offset_y += vertical_amp

        lantern_center = (ax_i + lantern_offset_x, ay_i + lantern_offset_y)
        lantern_r = max(2, int(r * 0.6))

        pygame.draw.circle(surf, self._lantern_color, lantern_center, lantern_r)
        pygame.draw.circle(surf, self._explorer_outline, lantern_center, lantern_r, 1)

        # lantern warm glow: smooth radial alpha gradient with flicker
        glow_radius = int(self.cell_w * 0.9 * self.zoom)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        base_alpha = 80
        for y in range(glow_radius * 2):
            dy = y - glow_radius
            for x in range(glow_radius * 2):
                dx = x - glow_radius
                dist = math.hypot(dx, dy) / glow_radius
                if dist > 1.0:
                    continue
                falloff = (1.0 - dist) ** 2
                alpha = int(base_alpha * falloff * flicker)
                if alpha <= 0:
                    continue
                glow_surf.set_at((x, y), (*self._lantern_glow_color, alpha))
        surf.blit(glow_surf, (lantern_center[0] - glow_radius, lantern_center[1] - glow_radius))

        # ---------- mini-map visualization (bottom-left) ----------
        mm_size = 64
        mm_margin = 4
        mm_x = mm_margin
        mm_y = self.height - mm_margin - mm_size

        # border
        mm_rect_border = pygame.Rect(mm_x - 1, mm_y - 1, mm_size + 2, mm_size + 2)
        pygame.draw.rect(surf, (20, 20, 25), mm_rect_border, border_radius=3)
        pygame.draw.rect(surf, (60, 60, 70), mm_rect_border, width=1, border_radius=3)

        # background under minimap
        mm_rect_bg = pygame.Rect(mm_x, mm_y, mm_size, mm_size)
        pygame.draw.rect(surf, (5, 5, 8), mm_rect_bg, border_radius=2)

        mm_cell_w = mm_size / self.grid_size
        mm_cell_h = mm_size / self.grid_size

        # current agent cell for overlay
        agent_cell = self._pos_to_cell(self.ax, self.ay)

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                val = self.mini_map[r, c]
                x0 = int(mm_x + c * mm_cell_w)
                y0 = int(mm_y + r * mm_cell_h)
                rect = pygame.Rect(
                    x0,
                    y0,
                    max(1, int(mm_cell_w)),
                    max(1, int(mm_cell_h)),
                )

                if val == -1:
                    color = (8, 8, 12)  # unseen
                elif val == 0:
                    color = (150, 120, 80)  # floor
                else:
                    color = (75, 55, 45)    # wall

                pygame.draw.rect(surf, color, rect)

        # overlay agent position on minimap
        if agent_cell is not None:
            ar_mm, ac_mm = agent_cell
            ax0 = int(mm_x + ac_mm * mm_cell_w)
            ay0 = int(mm_y + ar_mm * mm_cell_h)
            agent_rect = pygame.Rect(
                ax0,
                ay0,
                max(2, int(mm_cell_w * 0.7)),
                max(2, int(mm_cell_h * 0.7)),
            )
            pygame.draw.rect(surf, (210, 230, 255), agent_rect)

        # overlay chest position on minimap (only if revealed)
        if self.mini_map[self.chest_row, self.chest_col] != -1:
            cx0 = int(mm_x + self.chest_col * mm_cell_w)
            cy0 = int(mm_y + self.chest_row * mm_cell_h)
            chest_rect = pygame.Rect(
                cx0,
                cy0,
                max(2, int(mm_cell_w * 0.7)),
                max(2, int(mm_cell_h * 0.7)),
            )
            pygame.draw.rect(surf, (230, 190, 80), chest_rect)

        # ---------- golden overlay + "Success" text after chest opened ----------
        if self.chest_open:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay_alpha = int(35 + 30 * (flicker - 1.0))  # stronger, flickering
            overlay_alpha = max(80, min(140, overlay_alpha))
            overlay.fill((255, 235, 170, overlay_alpha))
            surf.blit(overlay, (0, 0))

            # "Success" label at top center
            text = "Success"
            text_surf = self._font_text.render(text, True, (255, 225, 150))
            shadow_surf = self._font_text.render(text, True, (30, 20, 10))
            text_rect = text_surf.get_rect(center=(self.width // 2, 18))
            shadow_rect = shadow_surf.get_rect(center=(self.width // 2 + 1, 19))
            surf.blit(shadow_surf, shadow_rect)
            surf.blit(text_surf, text_rect)

        arr = np.transpose(pygame.surfarray.pixels3d(surf), (1, 0, 2)).copy()
        return arr

    def close(self):
        if self._surface is not None:
            pygame.quit()
            self._surface = None


# --- layout variations ---

class DungeonExplorer1Env(DungeonExplorerEnv):
    def __init__(self, max_episode_steps=250):
        super().__init__(max_episode_steps=max_episode_steps, layout=0)


class DungeonExplorer2Env(DungeonExplorerEnv):
    def __init__(self, max_episode_steps=250):
        super().__init__(max_episode_steps=max_episode_steps, layout=1)


class DungeonExplorer3Env(DungeonExplorerEnv):
    def __init__(self, max_episode_steps=250):
        super().__init__(max_episode_steps=max_episode_steps, layout=2)


class DungeonExplorer4Env(DungeonExplorerEnv):
    def __init__(self, max_episode_steps=250):
        super().__init__(max_episode_steps=max_episode_steps, layout=3)


class DungeonExplorer5Env(DungeonExplorerEnv):
    def __init__(self, max_episode_steps=250):
        super().__init__(max_episode_steps=max_episode_steps, layout=4)


class DungeonExplorer6Env(DungeonExplorerEnv):
    def __init__(self, max_episode_steps=250):
        super().__init__(max_episode_steps=max_episode_steps, layout=5)


class DungeonExplorer7Env(DungeonExplorerEnv):
    def __init__(self, max_episode_steps=250):
        super().__init__(max_episode_steps=max_episode_steps, layout=6)


class DungeonExplorer8Env(DungeonExplorerEnv):
    def __init__(self, max_episode_steps=250):
        super().__init__(max_episode_steps=max_episode_steps, layout=7)


class DungeonExplorer9Env(DungeonExplorerEnv):
    def __init__(self, max_episode_steps=250):
        super().__init__(max_episode_steps=max_episode_steps, layout=8)


class DungeonExplorer10Env(DungeonExplorerEnv):
    def __init__(self, max_episode_steps=250):
        super().__init__(max_episode_steps=max_episode_steps, layout=9)
