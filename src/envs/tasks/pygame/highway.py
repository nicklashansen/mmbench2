import gymnasium as gym
import numpy as np
import pygame


class HighwayEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, max_episode_steps=500):
        super().__init__()
        self.max_episode_steps = max_episode_steps

        # obs = [car_x, car_y, vel_x, vel_y, collision] + max_cars * [x, y, is_truck]
        self.max_cars = 4
        obs_dim = 12 + self.max_cars * 3
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-np.ones(2, dtype=np.float32), high=np.ones(2, dtype=np.float32)
        )

        # canvas / physics
        self.width, self.height = 224, 224
        self.scroll_speed = 6
        self.thrust = 0.65
        self.damping = 0.95

        # road / lanes
        self.n_lanes = 5
        self.road_top = 18
        self.road_bottom = self.height - 18
        self.lane_h = (self.road_bottom - self.road_top) / self.n_lanes
        self.lane_centers = [self.road_top + (i + 0.5) * self.lane_h for i in range(self.n_lanes)]
        self.dash_offset = 0.0

        # agent car (red)
        self.agent_len = 28
        self.agent_h = 18
        self.car_x = 40.0
        self.car_y = self.lane_centers[self.n_lanes // 2] - self.agent_h / 2
        self.vel_x = 0.0
        self.vel_y = 0.0

        # incoming cars
        self.base_len = 28
        self.base_h = 18
        self.spawn_gap = 18
        self.car_types = [
            ("blue",   ( 95, 135, 212), 1.0),
            ("yellow", (227, 188,  34), 1.0),
            ("truck",  ( 82, 140,  66), 2.0),  # green truck: 2× length
        ]
        self.cars = []
        self.clear_window = int(self.base_len * 2.5)  # forward window for "lane clear" checks

        # --- reward shaping coefs (small) ---
        self.rs_lane_coef = 0.01             # max per-step lane-centering bonus
        self.rs_clear_ahead_bonus = 0.01     # bonus when lane ahead is clear
        self.rs_clear_ahead_pen = 0.02       # max penalty when too close ahead
        self.rs_clear_ahead_dist = self.base_len * 3  # "too close" threshold
        self.rs_action_l2 = 0.0005           # smooth-control penalty weight

        self._surface = None
        self.clock = None
        self.collided_flash = 0
        self.sticky_idx = None
        self.reset()

    # ------------- Gym API -------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.car_x = self.np_random.uniform(20, 60)
        self.car_y = (self.lane_centers[self.np_random.integers(1, self.n_lanes-1)] - self.agent_h / 2)
        self.vel_x = self.np_random.uniform(-0.2, 0.2)
        self.vel_y = self.np_random.uniform(-0.2, 0.2)
        self.steps = 0
        self.dash_offset = 0.0
        self.collided_flash = 0
        self.sticky_idx = None

        self.target_cars = self._target_cars_at_step(0)

        self.cars = []
        for _ in range(self.max_cars):
            self.cars.append({
                "active": False,
                "x": 0.0, "y": 0.0,
                "len": self.base_len, "h": self.base_h,
                "lane": 0,
                "color": (255, 255, 255),
                "kind": "blue",          # "blue" | "yellow" | "gray" | "truck"
                "is_truck": 0.0,
                "passed": False,
                "respawn": int(self.np_random.integers(0, 7)),
                "collision_timer": 0,
            })

        spawned, offset = 0, 0.0
        while spawned < self.target_cars:
            idx = spawned
            if self._spawn_car(idx, x=self.width + 40 + offset):
                spawned += 1
                offset += self.base_len * 2.5
            else:
                offset += self.base_len * 1.0

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        terminated = False
        truncated = False

        # agent dynamics
        self.vel_x += float(action[0]) * self.thrust
        self.vel_y += float(action[1]) * self.thrust
        self.car_x += self.vel_x
        self.car_y += self.vel_y
        self.vel_x *= self.damping
        self.vel_y *= self.damping

        # keep agent within road bounds
        self.car_x = np.clip(self.car_x, 0, self.width - self.agent_len)
        y_min = self.road_top
        y_max = self.road_bottom - self.agent_h
        self.car_y = np.clip(self.car_y, y_min, y_max)

        reward = 0.0
        self.dash_offset = (self.dash_offset + self.scroll_speed) % 32

        # increase allowed simultaneous cars as episode progresses
        self._update_target_cars()

        # --- Sticky collision (drag), with early stop near left edge
        if self.sticky_idx is not None:
            c = self.cars[self.sticky_idx]
            if c["active"]:
                speed = self._car_speed(c)  # respect yellow 1.2× logic even during stick
                next_cx = c["x"] - speed
                desired_x = next_cx - self.agent_len + 1
                next_ax = self.car_x - speed
                if next_ax < 0 or desired_x < 0:
                    # end early to avoid off-screen
                    c["active"] = False
                    c["respawn"] = int(self.np_random.integers(4, 11))
                    c["passed"] = False
                    c["collision_timer"] = 0
                    self.vel_x = 0.0
                    self.vel_y = 0.0
                    self.sticky_idx = None
                else:
                    # optional non-overlap clamp vs nearest-ahead while sticking
                    ah, gap = self._nearest_ahead(c)
                    if c["kind"] == "yellow" and ah is not None:
                        min_x = ah["x"] + ah["len"] + 1
                        next_cx = max(next_cx, min_x)
                    c["x"] = next_cx
                    c["collision_timer"] -= 1
                    self.car_x = next_ax
                    if self.car_x > desired_x:
                        self.car_x = desired_x
                    if c["x"] + c["len"] < 0 or c["collision_timer"] <= 0:
                        c["active"] = False
                        c["respawn"] = int(self.np_random.integers(4, 11))
                        c["passed"] = False
                        c["collision_timer"] = 0
                        self.vel_x = 0.0
                        self.vel_y = 0.0
                        self.sticky_idx = None

        agent_rect = pygame.Rect(int(self.car_x), int(self.car_y), self.agent_len, self.agent_h)

        # --- Advance other cars; passes; start collisions
        active_count = 0
        for idx, c in enumerate(self.cars):
            if not c["active"]:
                if c["respawn"] > 0:
                    c["respawn"] -= 1
                if c["respawn"] == 0 and active_count < self.target_cars:
                    if self._spawn_car_slot(c):
                        active_count += 1
                    else:
                        c["respawn"] = 1
                continue

            active_count += 1
            if idx == self.sticky_idx:
                continue  # updated above

            # speed with yellow 1.2× and approach control
            speed = self._car_speed(c)
            new_x = c["x"] - speed

            # hard non-overlap clamp when yellow follows another car in the same lane
            ah, gap = self._nearest_ahead(c)
            if c["kind"] == "yellow" and ah is not None:
                min_x = ah["x"] + ah["len"] + 1  # keep ≥1px gap
                if new_x < min_x:
                    new_x = min_x

            c["x"] = new_x

            # count pass when car’s right edge fully behind agent
            if (not c["passed"]) and (c["x"] + c["len"] < self.car_x):
                c["passed"] = True

            # begin sticky collision if not already in one
            if self.sticky_idx is None:
                car_rect = pygame.Rect(int(c["x"]), int(c["y"]), int(c["len"]), int(c["h"]))
                if agent_rect.colliderect(car_rect):
                    c["collision_timer"] = 5
                    self.sticky_idx = idx
                    self.collided_flash = 3
                    reward -= 1.0

            # off-screen (only if not mid-collision)
            if c["collision_timer"] == 0 and (c["x"] + c["len"] < 0):
                c["active"] = False
                c["respawn"] = int(self.np_random.integers(4, 11))
                c["passed"] = False

        # --- Reward shaping (small, dense) ---

        # 1) Lane-centering bonus (Gaussian over distance to nearest lane center)
        dist = self._lane_center_dist()
        norm = dist / (0.5 * self.lane_h)     # 0 at center, 1 at lane edge
        r_lane = self.rs_lane_coef * float(np.exp(-norm * norm))

        # 2) Clear-ahead term (same lane)
        dx = self._min_dx_ahead_same_lane()
        if dx is None:
            # empty lane ahead -> small bonus
            r_clear = self.rs_clear_ahead_bonus
        else:
            # within threshold -> ramped penalty; otherwise small bonus
            if dx < self.rs_clear_ahead_dist:
                r_clear = -self.rs_clear_ahead_pen * (1.0 - dx / self.rs_clear_ahead_dist)
            else:
                r_clear = 0.5 * self.rs_clear_ahead_bonus

        # 3) Smooth-control penalty (discourage thrashing)
        r_smooth = -self.rs_action_l2 * float(np.dot(action, action))

        reward += r_lane + r_clear + r_smooth

        self.steps += 1
        if self.steps >= self.max_episode_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    # ------------- Helpers -------------

    def _agent_center_y(self):
        return self.car_y + self.agent_h / 2

    def _agent_lane_index(self):
        cy = self._agent_center_y()
        return int(np.argmin([abs(cy - lc) for lc in self.lane_centers]))

    def _dx_ahead_in_lane(self, lane: int):
        """Distance from agent front to nearest car's left edge in `lane` (>=0 if ahead). None if no car ahead."""
        if lane < 0 or lane >= self.n_lanes:
            return None
        front = self.car_x + self.agent_len
        best = None
        for c in self.cars:
            if not c["active"] or c["lane"] != lane:
                continue
            dx = c["x"] - front  # car left − agent front
            if dx >= 0 and (best is None or dx < best):
                best = dx
        return best

    def _lane_clear_ahead(self, lane: int):
        """1.0 if no car within [front, front + clear_window] in `lane`, else 0.0. Returns 0.0 for invalid lane."""
        if lane < 0 or lane >= self.n_lanes:
            return 0.0
        front = self.car_x + self.agent_len
        for c in self.cars:
            if c["active"] and c["lane"] == lane:
                dx = c["x"] - front
                if 0 <= dx <= self.clear_window:
                    return 0.0
        return 1.0

    def _lane_center_dist(self):
        cy = self._agent_center_y()
        return min(abs(cy - lc) for lc in self.lane_centers)

    def _min_dx_ahead_same_lane(self):
        """Distance from agent front to nearest car’s left edge in the same lane (>=0 if ahead), or None."""
        lane = self._agent_lane_index()
        front = self.car_x + self.agent_len
        best = None
        for c in self.cars:
            if not c["active"] or c["lane"] != lane:
                continue
            dx = c["x"] - front  # car's left minus agent's front
            if dx >= 0 and (best is None or dx < best):
                best = dx
        return best

    def _nearest_ahead(self, car):
        """Return (nearest_car_ahead_in_same_lane, gap_left_edge_to_ahead_right_edge) or (None, None)."""
        best = None
        best_gap = None
        for o in self.cars:
            if not o["active"] or o is car or o["lane"] != car["lane"]:
                continue
            if o["x"] < car["x"]:  # ahead (closer to agent / left side)
                gap = car["x"] - (o["x"] + o["len"])
                if gap >= 0 and (best_gap is None or gap < best_gap):
                    best = o
                    best_gap = gap
        return best, best_gap

    def _car_speed(self, car):
        """Base scroll speed; yellow cars use 1.2× unless within one car length of a car ahead in same lane."""
        base = self.scroll_speed
        if car["kind"] != "yellow":
            return base
        ahead, gap = self._nearest_ahead(car)
        if ahead is not None:
            threshold = max(car["len"], ahead["len"])
            if gap <= threshold:
                return base  # slow down near another car
        return base * 1.2  # otherwise go faster

    # --- dynamic target-cars schedule ---
    def _target_cars_at_step(self, step: int) -> int:
        """Linearly scale simultaneous-car cap from 2 at t=0 to 6 at t=T-1."""
        if self.max_cars <= 1:
            return self.max_cars
        denom = max(1, self.max_episode_steps - 1)
        r = np.clip(step / denom, 0.0, 1.0)         # 0.0 → 1.0
        return int(np.floor(1 + (self.max_cars - 1) * r + 1e-9))  # 1..6

    def _update_target_cars(self):
        self.target_cars = self._target_cars_at_step(self.steps)

    def _spawn_car(self, idx, x=None):
        return self._spawn_car_slot(self.cars[idx], x=x)

    def _spawn_car_slot(self, slot, x=None):
        tname, color, lf = self.car_types[int(self.np_random.integers(0, len(self.car_types)))]
        length = int(self.base_len * lf)
        height = self.base_h

        for _ in range(8):
            lane = int(self.np_random.integers(0, self.n_lanes))
            spawn_x = x if x is not None else self.width + float(self.np_random.uniform(20, 60))
            ok = True
            for other in self.cars:
                if other["active"] and other["lane"] == lane:
                    if abs(other["x"] - spawn_x) < (max(length, other["len"]) + self.spawn_gap):
                        ok = False
                        break
            if ok:
                slot["active"] = True
                slot["lane"] = lane
                slot["len"] = length
                slot["h"] = height
                slot["x"] = spawn_x
                slot["y"] = self.lane_centers[lane] - height / 2
                slot["color"] = color
                slot["kind"] = tname
                slot["is_truck"] = 1.0 if lf > 1.0 else 0.0
                slot["passed"] = False
                slot["respawn"] = 0
                slot["collision_timer"] = 0
                return True
        return False

    def _get_obs(self):
        lane = self._agent_lane_index()
        lane_idx_norm = 0.0 if self.n_lanes <= 1 else lane / (self.n_lanes - 1)

        # signed offset to the *center* of current lane, normalized to [-1, 1]
        lane_center = self.lane_centers[lane]
        lane_offset = (self._agent_center_y() - lane_center) / (0.5 * self.lane_h)
        lane_offset = float(np.clip(lane_offset, -1.0, 1.0))

        # forward clearances (normalize by width; -1 if no car ahead)
        to_norm = lambda d: -1.0 if d is None else float(np.clip(d / self.width, 0.0, 1.0))
        dx_same  = to_norm(self._dx_ahead_in_lane(lane))
        dx_left  = to_norm(self._dx_ahead_in_lane(lane - 1))
        dx_right = to_norm(self._dx_ahead_in_lane(lane + 1))

        left_clear  = float(self._lane_clear_ahead(lane - 1))
        right_clear = float(self._lane_clear_ahead(lane + 1))

        # construct obs
        obs = [
            self.car_x / self.width,
            (self.car_y + self.agent_h / 2) / self.height,
            self.vel_x / 10.0,
            self.vel_y / 10.0,
            self.collided_flash / 3,
            dx_same,
            left_clear,
            right_clear,
            dx_left,
            dx_right,
            lane_offset,
            lane_idx_norm,
        ]
        for i in range(self.max_cars):
            if i < len(self.cars) and self.cars[i]["active"]:
                cx = self.cars[i]["x"] / self.width
                cy = (self.cars[i]["y"] + self.cars[i]["h"] / 2) / self.height
                it = self.cars[i]["is_truck"]
                obs += [cx, cy, it]
            else:
                obs += [-1.0, -1.0, -1.0]
        return np.array(obs, dtype=np.float32)

    # ------------- Rendering -------------

    def _draw_road(self, surf):
        surf.fill((97, 133, 80))
        # road
        road_rect = pygame.Rect(0, self.road_top, self.width, self.road_bottom - self.road_top)
        pygame.draw.rect(surf, (60, 60, 60), road_rect)

        # dashed lane separators
        dash_len, gap = 14, 10
        self.dash_offset = self.dash_offset % (dash_len + gap)
        x0 = -int(self.dash_offset)
        for k in range(1, self.n_lanes):
            y = int(self.road_top + k * self.lane_h)
            x = x0
            while x < self.width:
                pygame.draw.line(surf, (220, 220, 220), (x, y), (x + dash_len, y), 2)
                x += dash_len + gap

        # road edges
        pygame.draw.line(surf, (200, 200, 200), (0, self.road_top), (self.width, self.road_top), 2)
        pygame.draw.line(surf, (200, 200, 200), (0, self.road_bottom), (self.width, self.road_bottom), 2)

    def _draw_wheels(self, surf, rect):
        # four small black wheels (top-left, top-right, bottom-left, bottom-right)
        r = max(2, int(rect.height * 0.12))
        xt = rect.left + int(rect.width * 0.22)
        xb = rect.right - int(rect.width * 0.22)
        y_top = rect.top + 2
        y_bot = rect.bottom - 2
        pygame.draw.circle(surf, (20, 20, 20), (xt, y_top), r)
        pygame.draw.circle(surf, (20, 20, 20), (xb, y_top), r)
        pygame.draw.circle(surf, (20, 20, 20), (xt, y_bot), r)
        pygame.draw.circle(surf, (20, 20, 20), (xb, y_bot), r)

    def _draw_agent(self, surf):
        rect = pygame.Rect(int(self.car_x), int(self.car_y), self.agent_len, self.agent_h)
        pygame.draw.rect(surf, (220, 60, 60), rect)
        pygame.draw.rect(surf, (120, 20, 20), rect, 2)
        self._draw_wheels(surf, rect)

        # translucent (50%) direction triangle 4px in front of the car
        tri_overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        base_x = rect.right + 4
        tip = (base_x + 8, rect.centery)
        base_top = (base_x, rect.centery - 4)
        base_bot = (base_x, rect.centery + 4)
        pygame.draw.polygon(tri_overlay, (220, 60, 60, 128), [tip, base_top, base_bot])
        surf.blit(tri_overlay, (0, 0))

    def _draw_incoming(self, surf):
        for c in self.cars:
            if not c["active"]:
                continue
            rect = pygame.Rect(int(c["x"]), int(c["y"]), int(c["len"]), int(c["h"]))
            pygame.draw.rect(surf, c["color"], rect)
            pygame.draw.rect(surf, (40, 40, 40), rect, 2)
            self._draw_wheels(surf, rect)

    def render(self):
        if self._surface is None:
            pygame.init()
            self._surface = pygame.Surface((self.width, self.height))
            self.clock = pygame.time.Clock()

        self._draw_road(self._surface)
        self._draw_incoming(self._surface)
        self._draw_agent(self._surface)

        if self.collided_flash > 0:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((220, 60, 60, 50))
            self._surface.blit(overlay, (0, 0))
            self.collided_flash -= 1

        return np.transpose(np.array(pygame.surfarray.pixels3d(self._surface)), (1, 0, 2)).copy()

    def close(self):
        if self._surface is not None:
            pygame.quit()
            self._surface = None
