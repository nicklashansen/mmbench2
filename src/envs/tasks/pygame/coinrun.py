import gymnasium as gym
import numpy as np
import pygame


class CoinRunEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, max_episode_steps=500):
        super().__init__()
        self.max_episode_steps = max_episode_steps

        # obs = [agent_y, agent_vel_y] + coin positions (x,y) for up to 5 coins
        self.max_coins = 5
        obs_dim = 2 + self.max_coins * 2
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )

        self.width, self.height = 224, 224
        self.ground_y = self.height - 40
        self.gravity = 1.0
        self.jump_strength = 12
        self.scroll_speed = 10.0

        # character sprite
        self.agent_width = 18
        self.agent_height = 30
        self.agent_x = 40

        # coin parameters
        self.coin_radius = 9

        self._surface = None
        self.clock = None
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_y = self.ground_y - self.agent_height
        self.agent_vel_y = 0.0
        self.steps = 0
        self.leg_pose = 0
        self.arm_pose = 0
        self.pose_timer = 0
        self.on_ground = True

        self.spawn_coin_sequence()

        # -------- Pebbles --------
        self.num_pebbles = 10
        self.pebbles = []
        for _ in range(self.num_pebbles):
            x = self.width * self.np_random.uniform(-0.1, 1.0)
            y = self.ground_y + self.np_random.uniform(1, 4)
            self.pebbles.append([x, y])

        # -------- Clouds --------
        self.num_clouds = 7
        self.clouds = []
        for i in range(self.num_clouds):
            x = self.np_random.uniform(0, self.width) if i < int(self.num_clouds * 0.6) \
                else self.np_random.uniform(-0.3 * self.width, 1.3 * self.width)
            y = self.np_random.uniform(15, self.ground_y * 0.45)
            base = self.np_random.uniform(16, 26)
            speed = self.scroll_speed * self.np_random.uniform(0.18, 0.42)
            self.clouds.append([x, y, base, speed])

        # -------- Grass (reduced density + depth-based color) --------
        self.num_grass = 8
        self.grass = []
        for i in range(self.num_grass):
            x = self.np_random.uniform(0, self.width) if i < int(self.num_grass * 0.65) \
                else self.np_random.uniform(-0.1 * self.width, 1.3 * self.width)
            base_y = self.ground_y + self.np_random.uniform(-2.0, 3.0)
            h = self.np_random.uniform(5.0, 10.0)
            spread = self.np_random.uniform(2.0, 4.0)
            layer = 1 if self.np_random.random() < 0.35 else 0  # foreground?
            col_a, col_b, base_col = self._grass_colors_from_y(base_y)
            self.grass.append([x, base_y, h, spread, layer, col_a, col_b, base_col])

        # -------- Flowers (reduced density) --------
        self.num_flowers = 4
        self.flowers = []
        flower_colors = [
            (255, 182, 193),  # light pink
            (255, 204, 229),  # pink pastel
            (255, 255, 153),  # pastel yellow
            (221, 160, 221),  # lavender
        ]
        for i in range(self.num_flowers):
            x = self.np_random.uniform(0, self.width) if i < int(self.num_flowers * 0.6) \
                else self.np_random.uniform(-0.1 * self.width, 1.2 * self.width)
            head_y = self.ground_y - self.np_random.uniform(1.0, 6.0) + self.np_random.uniform(-2.0, 2.0)
            petal_col = flower_colors[self.np_random.integers(0, len(flower_colors))]
            size = self.np_random.uniform(2.0, 3.2)
            layer = 1 if self.np_random.random() < 0.3 else 0
            self.flowers.append([x, head_y, petal_col, size, layer])

        # -------- Coin sparkle particles (event-gated visuals) --------
        self.coin_particles = []  # list of [x, y, vx, vy, life, max_life, size, (r,g,b)]

        return self._get_obs(), {}

    def spawn_coin_sequence(self):
        n = self.np_random.integers(2, self.max_coins + 1)
        start_x = self.width + np.random.randint(20, 60)
        lowest_y = self.ground_y - self.coin_radius - 2
        highest_y = lowest_y - (self.coin_radius * 2) * 5
        start_y = lowest_y - (self.coin_radius * 2) * np.random.randint(1, 5)
        x, y = start_x, start_y
        self.coins = []
        for _ in range(n):
            self.coins.append([x, y, True])
            x += self.coin_radius * 3
            if np.random.rand() < 0.6:
                y += self.coin_radius * 2 * np.random.choice([-1, 1])
            y = np.clip(y, highest_y, lowest_y)

    def step(self, action):
        action = float(np.clip(action[0], -1, 1))
        terminated = False
        truncated = False

        # jump
        if action > 0 and self.agent_y >= self.ground_y - self.agent_height - 1e-3:
            jump_impulse = max(action, 0) * self.jump_strength
            self.agent_vel_y = -jump_impulse

        # physics
        self.agent_vel_y += self.gravity
        self.agent_y += self.agent_vel_y

        # ground collision + contact flag
        if self.agent_y > self.ground_y - self.agent_height:
            self.agent_y = self.ground_y - self.agent_height
            self.agent_vel_y = 0.0
            self.on_ground = True
        else:
            self.on_ground = False

        # animate running only when on ground
        if self.on_ground:
            self.pose_timer += 1
            if self.pose_timer % 6 == 0:
                self.leg_pose = 1 - self.leg_pose
                self.arm_pose = 1 - self.arm_pose

        reward = 0.0

        # ---- Scroll world objects ----
        for coin in self.coins:
            coin[0] -= self.scroll_speed

        for p in self.pebbles:
            p[0] -= self.scroll_speed
            if p[0] < -5:
                p[0] = self.width + self.np_random.uniform(0, 20)
                p[1] = self.ground_y + self.np_random.uniform(1, 4)

        for g in self.grass:
            g[0] -= self.scroll_speed
            if g[0] < -6:
                g[0] = self.width + self.np_random.uniform(0, 40)
                g[1] = self.ground_y + self.np_random.uniform(-2.0, 3.0)
                g[2] = self.np_random.uniform(5.0, 10.0)
                g[3] = self.np_random.uniform(2.0, 4.0)
                g[4] = 1 if self.np_random.random() < 0.35 else 0
                g[5], g[6], g[7] = self._grass_colors_from_y(g[1])

        for f in self.flowers:
            f[0] -= self.scroll_speed
            if f[0] < -6:
                f[0] = self.width + self.np_random.uniform(0, 60)
                f[1] = self.ground_y - self.np_random.uniform(1.0, 6.0) + self.np_random.uniform(-2.0, 2.0)
                f[3] = self.np_random.uniform(2.0, 3.2)
                f[4] = 1 if self.np_random.random() < 0.3 else 0

        for c in self.clouds:
            c[0] -= c[3]
            if c[0] < -c[2] * 3:
                c[0] = self.width + self.np_random.uniform(0, 60)
                c[1] = self.np_random.uniform(15, self.ground_y * 0.45)
                c[2] = self.np_random.uniform(16, 26)
                c[3] = self.scroll_speed * self.np_random.uniform(0.18, 0.42)

        # respawn new coin sequence if old coins are gone
        if all(not c[2] or c[0] < -self.coin_radius for c in self.coins):
            self.spawn_coin_sequence()

        # coin collection (+ sparkle pop)
        agent_rect = pygame.Rect(self.agent_x, int(self.agent_y),
                                 self.agent_width, self.agent_height)
        for coin in self.coins:
            if coin[2]:
                cx, cy = coin[0], coin[1]
                coin_rect = pygame.Rect(cx - self.coin_radius,
                                        cy - self.coin_radius,
                                        self.coin_radius * 2,
                                        self.coin_radius * 2)
                if agent_rect.colliderect(coin_rect):
                    coin[2] = False
                    reward += 1.0
                    self._spawn_coin_sparkle(cx, cy)  # event-gated visual particles

        # update sparkle particles
        new_particles = []
        for p in self.coin_particles:
            # p: [x,y,vx,vy,life,max_life,size,(r,g,b)]
            p[0] += p[2]
            p[1] += p[3]
            p[3] += 0.05  # slight gravity
            p[4] -= 1
            if p[4] > 0:
                new_particles.append(p)
        self.coin_particles = new_particles

        # animate character
        if self.agent_y >= self.ground_y - self.agent_height - 1e-3:
            self.pose_timer += 1
            if self.pose_timer % 6 == 0:
                self.leg_pose = 1 - self.leg_pose
                self.arm_pose = 1 - self.arm_pose

        self.steps += 1
        if self.steps >= self.max_episode_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        obs = [
            (self.agent_y - (self.ground_y - self.agent_height)) / self.height,
            self.agent_vel_y / 10.0,
        ]
        for i in range(self.max_coins):
            if i < len(self.coins) and self.coins[i][2]:
                obs.append(self.coins[i][0] / self.width)
                obs.append(self.coins[i][1] / self.height)
            else:
                obs.append(-1.0)
                obs.append(-1.0)
        return np.array(obs, dtype=np.float32)

    # ----------- Drawing helpers -----------

    def _draw_gradient_sky(self, surf):
        top_color = (173, 216, 230)     # light blue
        bottom_color = (255, 182, 193)  # light pink
        for y in range(self.ground_y):
            ratio = y / self.ground_y
            r = int(top_color[0] * (1 - ratio) + bottom_color[0] * ratio)
            g = int(top_color[1] * (1 - ratio) + bottom_color[1] * ratio)
            b = int(top_color[2] * (1 - ratio) + bottom_color[2] * ratio)
            pygame.draw.line(surf, (r, g, b), (0, y), (self.width, y))

    def _draw_ground_gradient(self, surf):
        # darker near ground line, lighter toward bottom
        top_col = (76, 130, 78)
        bot_col = (112, 170, 112)
        h = self.height - self.ground_y
        for i, y in enumerate(range(self.ground_y, self.height)):
            t = i / max(1, h - 1)
            r = int(top_col[0] * (1 - t) + bot_col[0] * t)
            g = int(top_col[1] * (1 - t) + bot_col[1] * t)
            b = int(top_col[2] * (1 - t) + bot_col[2] * t)
            pygame.draw.line(surf, (r, g, b), (0, y), (self.width, y))

    def _draw_cloud(self, surf, x, y, base):
        col = (255, 204, 214)
        rect1 = pygame.Rect(int(x), int(y), int(base * 2.0), int(base * 1.0))
        rect2 = pygame.Rect(int(x - base * 0.6), int(y + base * 0.15), int(base * 1.6), int(base * 0.9))
        rect3 = pygame.Rect(int(x + base * 0.9), int(y + base * 0.1), int(base * 1.6), int(base * 0.9))
        pygame.draw.ellipse(surf, col, rect1)
        pygame.draw.ellipse(surf, col, rect2)
        pygame.draw.ellipse(surf, col, rect3)

    def _grass_colors_from_y(self, base_y):
        """
        Return (blade_col_a, blade_col_b, base_col) based on vertical position to create depth.
        Darker near the ground line (top of ground), lighter toward bottom.
        """
        t = (base_y - (self.ground_y - 6.0)) / 12.0
        t = float(np.clip(t, 0.0, 1.0))
        dark_a = (34, 110, 38)
        light_a = (60, 150, 64)
        dark_b = (40, 120, 42)
        light_b = (70, 160, 72)
        base_dark = (46, 102, 50)
        base_light = (66, 132, 70)

        def lerp(c0, c1, u):
            return (int(c0[0]*(1-u) + c1[0]*u),
                    int(c0[1]*(1-u) + c1[1]*u),
                    int(c0[2]*(1-u) + c1[2]*u))

        col_a = lerp(dark_a, light_a, t)
        col_b = lerp(dark_b, light_b, t)
        base_col = lerp(base_dark, base_light, t)
        return col_a, col_b, base_col

    def _draw_grass_tuft(self, surf, x, base_y, h, spread, col_a, col_b, base_col):
        pygame.draw.line(surf, base_col, (int(x - spread), int(base_y)), (int(x + spread), int(base_y)), 2)
        blade_count = 3 + int(self.np_random.integers(0, 3))
        for i in range(blade_count):
            t = (i - (blade_count - 1) / 2.0) / max(1, (blade_count - 1) / 2.0)  # [-1,1]
            tip_x = x + t * spread * 1.2
            tip_y = base_y - h * self.np_random.uniform(0.85, 1.1)
            col = col_a if (i % 2 == 0) else col_b
            pygame.draw.line(surf, col, (int(x + t * spread * 0.4), int(base_y)), (int(tip_x), int(tip_y)), 2)

    def _draw_flower(self, surf, x, head_y, petal_col, size):
        stem_col = (60, 120, 60)
        center_col = (255, 230, 120)
        pygame.draw.line(surf, stem_col, (int(x), int(head_y + size * 1.2)), (int(x), int(self.ground_y)), 2)
        pygame.draw.line(surf, stem_col, (int(x), int(head_y + size * 0.8)), (int(x - size), int(head_y + size * 0.9)), 1)
        pygame.draw.line(surf, stem_col, (int(x), int(head_y + size * 0.6)), (int(x + size), int(head_y + size * 0.7)), 1)
        angles = [0, 72, 144, 216, 288]
        r = size
        for a in angles:
            rad = np.deg2rad(a)
            px = x + np.cos(rad) * (r * 1.6)
            py = head_y + np.sin(rad) * (r * 1.6)
            pygame.draw.circle(surf, petal_col, (int(px), int(py)), int(max(1, r)))
        pygame.draw.circle(surf, center_col, (int(x), int(head_y)), int(max(1, r * 0.9)))

    def _draw_character(self, surf):
        x, y = self.agent_x, int(self.agent_y)
        skin_color  = (222, 184, 135)
        dress_color = (139, 69, 19)
        hair_color  = (220, 20, 60)

        # Hair (behind head)
        pygame.draw.rect(surf, hair_color, (x + 2, y - 1, 14, 18), border_radius=7)
        # Head
        pygame.draw.circle(surf, skin_color, (x + 9, y + 5), 5)

        # --- Arms: subtle tuck when airborne ---
        def lerp(a, b, t): return a + (b - a) * t
        tuck = 1.0 if not self.on_ground else 0.0  # interp param: 0 on ground, 1 in air

        # Shoulder anchors (fixed)
        Ls = (x + 6,  y + 12)
        Rs = (x + 12, y + 12)

        if self.arm_pose == 0:
            # Ground (pose 0) endpoints
            Lg = (x + 0,  y + 16)
            Rg = (x + 18, y + 16)
            # Air-tucked endpoints (closer + slightly higher)
            La = (x + 4,  y + 14)
            Ra = (x + 14, y + 14)
        else:
            # Ground (pose 1) endpoints
            Lg = (x + 1,  y + 17)
            Rg = (x + 21, y + 17)
            # Air-tucked endpoints
            La = (x + 5,  y + 15)
            Ra = (x + 17, y + 15)

        # Interpolate ground→air
        Le = (int(round(lerp(Lg[0], La[0], tuck))), int(round(lerp(Lg[1], La[1], tuck))))
        Re = (int(round(lerp(Rg[0], Ra[0], tuck))), int(round(lerp(Rg[1], Ra[1], tuck))))

        pygame.draw.line(surf, (0, 0, 0), Ls, Le, 2)
        pygame.draw.line(surf, (0, 0, 0), Rs, Re, 2)

        # --- Legs ---
        if self.leg_pose == 0:
            pygame.draw.line(surf, (0, 0, 0), (x + 8,  y + 25), (x + 4,  y + 35), 2)
            pygame.draw.line(surf, (0, 0, 0), (x + 10, y + 25), (x + 14, y + 35), 2)
        else:
            pygame.draw.line(surf, (0, 0, 0), (x + 8,  y + 25), (x + 2,  y + 35), 2)
            pygame.draw.line(surf, (0, 0, 0), (x + 10, y + 25), (x + 16, y + 35), 2)

        # --- Dress (two triangles) ---
        pygame.draw.polygon(
            surf, dress_color, [(x + 6, y + 10), (x + 12, y + 10), (x + 9, y + 25)]
        )
        pygame.draw.polygon(
            surf, dress_color, [(x + 3,  y + 30), (x + 15, y + 30), (x + 9, y + 18)]
        )

    def _draw_coin(self, surf, x, y, idx):
        base_col   = (255, 215, 0)
        rim_base   = (200, 160, 0)
        rim_bright = (255, 220, 60)

        # pulse period (frames); subtle rim-brightness amplitude
        period = 30.0
        t = (self.steps + idx * 7) / period
        s = np.sin(2.0 * np.pi * t)
        weight = 0.5 + 0.125 * s  # in [0.375, 0.625] -> subtle

        # Subtle rim lerp
        rim_col = (
            int(rim_base[0] * (1 - weight) + rim_bright[0] * weight),
            int(rim_base[1] * (1 - weight) + rim_bright[1] * weight),
            int(rim_base[2] * (1 - weight) + rim_bright[2] * weight),
        )

        # Body + rim
        pygame.draw.circle(surf, base_col, (int(x), int(y)), self.coin_radius)
        pygame.draw.circle(surf, rim_col, (int(x), int(y)), self.coin_radius, 2)

        # rotating glint highlight (rotates 2x the coin pulse rate)
        theta = (self.steps + idx * 7) * (2.0 * np.pi / 45.0)
        r = self.coin_radius - 1
        gx = int(x + r * np.cos(theta))
        gy = int(y + r * np.sin(theta))

        # short star arms and thin arc (mostly 1px)
        L = max(1, int(round(1 + 0.6 * weight)))
        thick = max(1, int(round(1 + 0.4 * weight)))
        c_star = (245, 245, 235)         # slightly off-white for subtlety
        c_arc  = (250, 245, 200)         # softer highlight

        pygame.draw.line(surf, c_star, (gx - L, gy), (gx + L, gy), 1)
        pygame.draw.line(surf, c_star, (gx, gy - L), (gx, gy + L), 1)

        arc_r = self.coin_radius + 2
        rect = pygame.Rect(int(x - arc_r), int(y - arc_r), int(arc_r * 2), int(arc_r * 2))
        pygame.draw.arc(surf, c_arc, rect, np.deg2rad(300), np.deg2rad(20), thick)

    # ---- Event-gated coin sparkle pop (particles) ----
    def _spawn_coin_sparkle(self, x, y):
        n = 8
        for k in range(n):
            ang = (2 * np.pi * k) / n + float(self.np_random.uniform(-0.25, 0.25))
            spd = float(self.np_random.uniform(1.0, 2.4))
            vx = np.cos(ang) * spd
            vy = np.sin(ang) * spd - float(self.np_random.uniform(0.0, 0.3))
            life = int(self.np_random.integers(10, 16))
            size = float(self.np_random.uniform(1.0, 2.0))
            # alternate yellow/white sparkles
            col = (255, 230, 120) if (k % 2 == 0) else (255, 255, 255)
            self.coin_particles.append([x, y, vx, vy, life, life, size, col])

    def _draw_coin_particles(self, surf):
        for p in self.coin_particles:
            x, y, vx, vy, life, max_life, size, col = p
            t = life / max(1, max_life)
            # fade by shrinking + dimming
            rad = int(max(1, size * (0.6 + 0.8 * t)))
            c = (int(col[0] * (0.6 + 0.4 * t)),
                 int(col[1] * (0.6 + 0.4 * t)),
                 int(col[2] * (0.6 + 0.4 * t)))
            pygame.draw.circle(surf, c, (int(x), int(y)), rad)

    def render(self):
        if self._surface is None:
            pygame.init()
            self._surface = pygame.Surface((self.width, self.height))
            self.clock = pygame.time.Clock()

        # Sky
        self._draw_gradient_sky(self._surface)

        # Ground with gradient
        self._draw_ground_gradient(self._surface)

        ground_surface_color = (51, 92, 53)
        pygame.draw.line(self._surface, ground_surface_color,
                         (0, self.ground_y), (self.width, self.ground_y), 2)

        # Clouds (background)
        for c in self.clouds:
            self._draw_cloud(self._surface, c[0], c[1], c[2])

        # Grass/Flowers (background layer)
        for g in self.grass:
            if g[4] == 0:
                self._draw_grass_tuft(self._surface, g[0], g[1], g[2], g[3], g[5], g[6], g[7])
        for f in self.flowers:
            if f[4] == 0:
                self._draw_flower(self._surface, f[0], f[1], f[2], f[3])

        # Pebbles
        for p in self.pebbles:
            pygame.draw.circle(self._surface, ground_surface_color, (int(p[0]), int(p[1])), 1)

        # Character
        self._draw_character(self._surface)

        # Grass/Flowers (foreground layer for occlusion)
        for g in self.grass:
            if g[4] == 1:
                self._draw_grass_tuft(self._surface, g[0], g[1], g[2], g[3], g[5], g[6], g[7])
        for f in self.flowers:
            if f[4] == 1:
                self._draw_flower(self._surface, f[0], f[1], f[2], f[3])

        # Coins + coin particles
        for idx, coin in enumerate(self.coins):
            if coin[2]:
                self._draw_coin(self._surface, coin[0], coin[1], idx)
        self._draw_coin_particles(self._surface)

        return np.transpose(np.array(pygame.surfarray.pixels3d(self._surface)), (1, 0, 2)).copy()

    def close(self):
        if self._surface is not None:
            pygame.quit()
            self._surface = None
