import gymnasium as gym
import numpy as np
import pygame


class LandingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, max_episode_steps=200):
        super().__init__()
        self.max_episode_steps = max_episode_steps

        # obs = [heli_x, heli_y, vel_x, vel_y, ship_x, ship_y, ship_vx]
        self.obs_dim = 7
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-np.ones(2, dtype=np.float32), high=np.ones(2, dtype=np.float32)
        )

        # canvas
        self.width, self.height = 224, 224
        self.sky_h = int(self.height * 0.80)  # more air above

        # physics
        self.thrust = 0.6
        self.damping = 0.96
        self.gravity = 0.20

        # helicopter
        self.heli_w, self.heli_h = 28, 18
        self.heli_x = 8
        self.heli_y = 8
        self.vel_x = 0.0
        self.vel_y = 0.0

        # ship deck geom (physics contact surface)
        self.ship_w, self.ship_h = 70, 12  # deck width/height
        self.deck_offset = 10
        self.ship_x = self.width // 3
        self.ship_y = self.sky_h + self.deck_offset
        self.ship_speed = 1.4
        self.ship_dir = 1  # +1 right, -1 left

        # visuals: hull/cabin
        self.hull_extra_w = 12   # hull fanout beyond deck
        self.hull_h = 16         # hull height below deck
        self.cabin_w, self.cabin_h = 18, 12

        # sticky deck
        self.in_contact = False
        self.locked_on_deck = False
        self.prev_in_contact = False

        # reward shaping / penalties
        self.down_vel_coef = 0.001          # small per-step penalty for downward speed
        self.hard_landing_scale = 2.0       # linear scale for touchdown impact, capped at -10
        self.contact_edge_reward = 0.1      # reward at deck edge
        self.contact_center_reward = 1.0    # reward at deck center
        self.below_deck_step_penalty = 0.5  # per step penalty when below deck
        self.prox_reward_max = 0.025        # small shaping for proximity

        # render
        self._surface = None
        self.clock = None

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # heli
        hx = self.np_random.uniform(12, 36)
        hy = self.np_random.uniform(12, 36)
        hvx = self.np_random.uniform(-0.3, 0.3)
        hvy = self.np_random.uniform(-0.3, 0.3)
        self.heli_x, self.heli_y = hx, hy
        self.vel_x, self.vel_y = hvx, hvy

        # ship
        self.ship_x = self.width // 3
        self.ship_y = self.sky_h + self.deck_offset
        self.ship_dir = 1 if self.np_random.random() < 0.5 else -1
        self.ship_speed = 1.4 + self.np_random.uniform(-0.3, 0.3)
        self.hull_taper_in = 10  # pixels: deck→water narrows by this on each side

        # sky stars
        self.num_stars = 40
        self.stars = [
            [self.np_random.uniform(0, self.width),
             self.np_random.uniform(0, self.sky_h)]
            for _ in range(self.num_stars)
        ]

        self.steps = 0
        self.in_contact = False
        self.prev_in_contact = False
        self.locked_on_deck = False
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1, 1)
        terminated = False
        truncated = False

        # helicopter dynamics
        self.vel_x += float(action[0]) * self.thrust
        self.vel_y += float(action[1]) * self.thrust
        self.vel_y += self.gravity
        self.heli_x += self.vel_x
        self.heli_y += self.vel_y
        self.vel_x *= self.damping
        self.vel_y *= self.damping

        # bounds
        self.heli_x = np.clip(self.heli_x, 0, self.width - self.heli_w)
        self.heli_y = np.clip(self.heli_y, 0, self.height - self.heli_h)

        # ship motion (bounce at edges)
        self.ship_x += self.ship_dir * self.ship_speed
        if self.ship_x - self.ship_w // 2 <= 0:
            self.ship_x = self.ship_w // 2
            self.ship_dir = 1
        elif self.ship_x + self.ship_w // 2 >= self.width:
            self.ship_x = self.width - self.ship_w // 2
            self.ship_dir = -1

        # rects
        heli_rect = pygame.Rect(int(self.heli_x), int(self.heli_y), self.heli_w, self.heli_h)
        ship_rect = pygame.Rect(
            int(self.ship_x - self.ship_w // 2),
            int(self.ship_y - self.ship_h // 2),
            self.ship_w, self.ship_h
        )

        vy_pre_contact = float(self.vel_y)  # for impact penalty

        # --- Collision resolution with deck ---
        if heli_rect.colliderect(ship_rect):
            overlap_left   = heli_rect.right - ship_rect.left
            overlap_right  = ship_rect.right - heli_rect.left
            overlap_top    = heli_rect.bottom - ship_rect.top
            overlap_bottom = ship_rect.bottom - heli_rect.top
            pen_x = min(overlap_left, overlap_right)
            pen_y = min(overlap_top, overlap_bottom)

            if pen_y <= pen_x:
                if heli_rect.centery < ship_rect.centery:
                    self.heli_y = ship_rect.top - self.heli_h
                    self.vel_y = min(0.0, self.vel_y)  # kill downward component
                    self.vel_x *= 0.90
                else:
                    self.heli_y = ship_rect.bottom
                    self.vel_y = max(0.0, self.vel_y)
            else:
                if heli_rect.centerx < ship_rect.centerx:
                    self.heli_x = ship_rect.left - self.heli_w
                    self.vel_x = min(0.0, self.vel_x)
                else:
                    self.heli_x = ship_rect.right
                    self.vel_x = max(0.0, self.vel_x)
            heli_rect.x, heli_rect.y = int(self.heli_x), int(self.heli_y)

        # --- Contact detection (skids on deck) ---
        tol = 3
        horizontal_overlap = (
            heli_rect.right > ship_rect.left + 4 and
            heli_rect.left < ship_rect.right - 4
        )
        vertical_contact = (
            heli_rect.bottom >= ship_rect.top - tol and
            heli_rect.bottom <= ship_rect.top + tol
        )
        self.in_contact = horizontal_overlap and vertical_contact

        # --- Sticky deck behavior ---
        ship_vx = self.ship_dir * self.ship_speed
        if self.in_contact:
            # blend vx toward ship
            alpha = 0.30
            self.vel_x = (1 - alpha) * self.vel_x + alpha * ship_vx
            # latch when nearly still
            if abs(self.vel_y) < 0.20 and abs(self.vel_x - ship_vx) < 0.20:
                self.locked_on_deck = True
        else:
            self.locked_on_deck = False

        if self.locked_on_deck:
            self.vel_x = ship_vx
            self.vel_y = min(0.0, self.vel_y)
            self.heli_y = ship_rect.top - self.heli_h

        # --- Reward shaping ---
        reward = 0.0

        # Contact reward shaped by distance to deck center
        if self.in_contact:
            dx = abs(heli_rect.centerx - ship_rect.centerx)
            max_dx = ship_rect.width * 0.5  # edge of deck
            d_norm = np.clip(dx / max_dx, 0.0, 1.0)
            contact_r = self.contact_edge_reward + (self.contact_center_reward - self.contact_edge_reward) * (1.0 - d_norm)
            reward += float(contact_r)

        # Small per-step penalty for downward velocity (positive vy = moving down)
        if self.vel_y > 0.0:
            reward -= self.down_vel_coef * float(self.vel_y)

        # One-shot hard-landing penalty on touchdown (downward speed just before contact)
        if (not self.prev_in_contact) and self.in_contact:
            impact_v = max(0.0, vy_pre_contact)
            hard_pen = min(10.0, self.hard_landing_scale * impact_v)
            reward -= hard_pen

        # Per-step penalty when heli is below the deck height
        if heli_rect.centery > ship_rect.top:
            reward -= self.below_deck_step_penalty

        # Encourages aligning above the landing area even when airborne
        if heli_rect.centery <= ship_rect.top:
            dx = abs(heli_rect.centerx - ship_rect.centerx)
            dy = abs(heli_rect.centery - ship_rect.centery)
            max_dx = ship_rect.width * 0.5
            max_dy = self.height
            dx_norm = np.clip(dx / max_dx, 0.0, 1.0)
            dy_norm = np.clip(dy / max_dy, 0.0, 1.0)
            reward += self.prox_reward_max * (1.0 - dx_norm)
            reward += 4*self.prox_reward_max * (1.0 - dy_norm)

        self.prev_in_contact = self.in_contact

        self.steps += 1
        if self.steps >= self.max_episode_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        return np.array([
            self.heli_x / self.width,
            self.heli_y / self.height,
            self.vel_x / 10.0,
            self.vel_y / 10.0,
            self.ship_x / self.width,
            self.ship_y / self.height,
            (self.ship_dir * self.ship_speed) / 5.0,
        ], dtype=np.float32)

    def _draw_helicopter(self, surf):
        x, y = int(self.heli_x), int(self.heli_y)
        body_col = (180, 200, 220)
        accent = (90, 120, 180)
        body = [(x, y + self.heli_h // 2),
                (x + self.heli_w - 6, y),
                (x + self.heli_w, y + self.heli_h // 2),
                (x + self.heli_w - 6, y + self.heli_h)]
        pygame.draw.polygon(surf, body_col, body)
        pygame.draw.ppolygon = pygame.draw.polygon
        pygame.draw.ppolygon(surf, accent, body, 2)
        pygame.draw.line(surf, accent, (x + 6, y + self.heli_h // 2), (x - 8, y + self.heli_h // 2), 2)
        # skids
        pygame.draw.line(surf, (80, 80, 80), (x + 4, y + self.heli_h), (x + self.heli_w - 8, y + self.heli_h), 3)
        pygame.draw.line(surf, (80, 80, 80), (x + 8, y + self.heli_h - 6), (x + 8, y + self.heli_h), 2)
        pygame.draw.line(surf, (80, 80, 80), (x + self.heli_w - 12, y + self.heli_h - 6), (x + self.heli_w - 12, y + self.heli_h), 2)
        # rotor
        cx, top, span = x + self.heli_w // 2, y - 3, 16
        pygame.draw.line(surf, (50, 50, 50), (cx - span, top), (cx + span, top), 2)

    def _draw_ship(self, surf):
        # deck rect (physics contact surface)
        deck = pygame.Rect(
            int(self.ship_x - self.ship_w // 2),
            int(self.ship_y - self.ship_h // 2),
            self.ship_w, self.ship_h
        )

        # --- Hull: widest at deck, narrower at waterline (taper-in) ---
        bottom_y = deck.bottom + self.hull_h
        hull_poly = [
            (deck.left,  deck.bottom),                      # top-left (under deck)
            (deck.right, deck.bottom),                      # top-right
            (deck.right - self.hull_taper_in, bottom_y),    # bottom-right (taper inward)
            (deck.left  + self.hull_taper_in, bottom_y),    # bottom-left  (taper inward)
        ]
        pygame.draw.polygon(surf, (120, 120, 140), hull_poly)
        pygame.draw.polygon(surf, (80, 80, 100), hull_poly, 2)

        # --- Deck (landing pad) ---
        pygame.draw.rect(surf, (240, 240, 240), deck)
        pygame.draw.rect(surf, (180, 180, 180), deck, 2)
        # center stripe
        pygame.draw.line(surf, (160, 160, 160),
                        (deck.left + 6, deck.centery),
                        (deck.right - 6, deck.centery), 1)

        # --- Bow/Stern flip based on motion ---
        moving_right = (self.ship_dir > 0)
        if moving_right:
            # Bow on the right
            bow = [(deck.right, deck.top),
                (deck.right + 10, deck.centery),
                (deck.right, deck.bottom)]
            cabin = pygame.Rect(deck.left + 6, deck.top - self.cabin_h,
                                self.cabin_w, self.cabin_h)  # stern-side (left)
        else:
            # Bow on the left
            bow = [(deck.left, deck.top),
                (deck.left - 10, deck.centery),
                (deck.left, deck.bottom)]
            cabin = pygame.Rect(deck.right - self.cabin_w - 6, deck.top - self.cabin_h,
                                self.cabin_w, self.cabin_h)  # stern-side (right)

        pygame.draw.polygon(surf, (240, 240, 240), bow)
        pygame.draw.polygon(surf, (180, 180, 180), bow, 2)

        # Cabin + tiny windows
        pygame.draw.rect(surf, (230, 230, 235), cabin)
        pygame.draw.rect(surf, (160, 160, 170), cabin, 2)
        for i in range(3):
            wx = cabin.left + 3 + i * 6
            wy = cabin.top + 3
            pygame.draw.rect(surf, (180, 210, 240), pygame.Rect(wx, wy, 4, 5))

        # --- Flag at deck center (reward target) ---
        pole_x = deck.centerx
        pole_top = deck.top - 10
        pygame.draw.line(surf, (70, 70, 70), (pole_x, deck.top), (pole_x, pole_top), 2)
        flag = [(pole_x, pole_top), (pole_x + 8, pole_top + 3), (pole_x, pole_top + 6)]
        pygame.draw.polygon(surf, (220, 40, 40), flag)

    def render(self):
        if self._surface is None:
            pygame.init()
            self._surface = pygame.Surface((self.width, self.height))
            self.clock = pygame.time.Clock()

        # sky + water
        self._surface.fill((150, 190, 255))
        pygame.draw.rect(self._surface, (50, 120, 200),
                         pygame.Rect(0, self.sky_h, self.width, self.height - self.sky_h))

        # stars
        for s in self.stars:
            pygame.draw.circle(self._surface, (240, 240, 255), (int(s[0]), int(s[1])), 1)

        # ship + heli
        self._draw_ship(self._surface)
        self._draw_helicopter(self._surface)

        # contact overlay
        if self.in_contact:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((80, 220, 120, 40))
            self._surface.blit(overlay, (0, 0))

        arr = pygame.surfarray.array3d(self._surface)
        return np.swapaxes(arr, 0, 1).copy()

    def close(self):
        if self._surface is not None:
            pygame.quit()
            self._surface = None
