import gymnasium as gym
import numpy as np
import pygame


class SpaceshipEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, max_episode_steps=500):
        super().__init__()
        self.max_episode_steps = max_episode_steps

        # obs = [ship_x, ship_y, vel_x, vel_y] + coin positions
        self.max_coins = 4
        obs_dim = 4 + self.max_coins * 2 + 2  # +2 for asteroid
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # continuous thrust in XY, normalized to [-1,1]
        self.action_space = gym.spaces.Box(
            low=-np.ones(2, dtype=np.float32), high=np.ones(2, dtype=np.float32)
        )

        self.width, self.height = 224, 224
        self.scroll_speed = 12.0
        self.thrust = 0.65
        self.damping = 0.95

        # spaceship
        self.ship_w, self.ship_h = 28, 18
        self.ship_x = 40
        self.ship_y = self.height // 2
        self.vel_x, self.vel_y = 0.0, 0.0

        # coins
        self.coin_radius = 9

        # asteroids
        self.asteroid_radius = self.coin_radius * 2
        self.asteroid = None  # (x, y, active, angle, shape)
        self.collided = False

        self._surface = None
        self.clock = None
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ship_x = 40
        self.ship_y = self.height // 2
        self.vel_x = self.vel_y = 0.0
        self.steps = 0

        self.spawn_coin_sequence()

        # stars for background
        self.num_stars = 50
        self.stars = []
        for _ in range(self.num_stars):
            x = self.np_random.uniform(0, self.width)
            y = self.np_random.uniform(0, self.height)
            self.stars.append([x, y])

        return self._get_obs(), {}

    def spawn_coin_sequence(self):
        n = self.np_random.integers(2, self.max_coins + 1)
        start_x = self.width + np.random.randint(20, 60)
        start_y = np.random.randint(30, self.height - 30)
        x, y = start_x, start_y
        self.coins = []
        for _ in range(n):
            self.coins.append([x, y, True])
            x += self.coin_radius * 3
            if np.random.rand() < 0.6:
                y += self.coin_radius * 2 * np.random.choice([-1, 1])
            y = np.clip(y, 20, self.height - 20)

        # only spawn asteroid if none currently active
        if (not self.asteroid) or (not self.asteroid[2]):
            if np.random.rand() < 0.5:  # 50% chance
                for _ in range(5):  # try up to 5 times
                    ax = start_x + (n + 2) * self.coin_radius * 3 + self.np_random.uniform(-30, 30)
                    ay = np.random.randint(40, self.height - 40)

                    asteroid_rect = pygame.Rect(ax - self.asteroid_radius,
                                                ay - self.asteroid_radius,
                                                self.asteroid_radius * 2,
                                                self.asteroid_radius * 2)

                    # check overlap with coins
                    overlap = False
                    for cx, cy, active in self.coins:
                        if active:
                            coin_rect = pygame.Rect(cx - self.coin_radius,
                                                    cy - self.coin_radius,
                                                    self.coin_radius * 2,
                                                    self.coin_radius * 2)
                            if asteroid_rect.colliderect(coin_rect):
                                overlap = True
                                break

                    if not overlap:
                        # pre-generate jagged polygon points
                        num_points = 10
                        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
                        radii = self.asteroid_radius * (0.7 + 0.6 * np.random.rand(num_points))
                        shape = list(zip(angles, radii))

                        self.asteroid = [ax, ay, True, 0.0, shape]
                        break
                # if overlap every time → asteroid just won't spawn

    def step(self, action):
        action = np.clip(action, -1, 1)
        terminated = False
        truncated = False

        # thrust + physics
        self.vel_x += float(action[0]) * self.thrust
        self.vel_y += float(action[1]) * self.thrust
        self.ship_x += self.vel_x
        self.ship_y += self.vel_y
        self.vel_x *= self.damping
        self.vel_y *= self.damping

        # keep ship inside screen
        self.ship_x = np.clip(self.ship_x, 0, self.width - self.ship_w)
        self.ship_y = np.clip(self.ship_y, 0, self.height - self.ship_h)

        reward = 0.0

        # scroll asteroid
        if self.asteroid and self.asteroid[2]:
            self.asteroid[0] -= self.scroll_speed
            self.asteroid[3] += 0.02  # slow rotation
            if self.asteroid[0] < -self.asteroid_radius:
                self.asteroid[2] = False

            # check overlap penalty
            ship_rect = pygame.Rect(int(self.ship_x), int(self.ship_y),
                                    self.ship_w, self.ship_h)
            ax, ay = self.asteroid[0], self.asteroid[1]
            asteroid_rect = pygame.Rect(ax - self.asteroid_radius,
                                        ay - self.asteroid_radius,
                                        self.asteroid_radius * 2,
                                        self.asteroid_radius * 2)
            self.collided = ship_rect.colliderect(asteroid_rect)
            if self.collided:
                reward -= 2.0

        # scroll coins + stars
        for coin in self.coins:
            coin[0] -= self.scroll_speed
        for s in self.stars:
            s[0] -= self.scroll_speed * 0.5
            if s[0] < 0:
                s[0] = self.width
                s[1] = self.np_random.uniform(0, self.height)

        # respawn if no coins left
        if all(not c[2] or c[0] < -self.coin_radius for c in self.coins):
            self.spawn_coin_sequence()

        # check coin collection
        ship_rect = pygame.Rect(int(self.ship_x), int(self.ship_y),
                                self.ship_w, self.ship_h)
        for coin in self.coins:
            if coin[2]:
                cx, cy = coin[0], coin[1]
                coin_rect = pygame.Rect(cx - self.coin_radius,
                                        cy - self.coin_radius,
                                        self.coin_radius * 2,
                                        self.coin_radius * 2)
                if ship_rect.colliderect(coin_rect):
                    coin[2] = False
                    reward += 1.0

        self.steps += 1
        if self.steps >= self.max_episode_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        obs = [
            self.ship_x / self.width,
            self.ship_y / self.height,
            self.vel_x / 10.0,
            self.vel_y / 10.0,
        ]
        # coins
        for i in range(self.max_coins):
            if i < len(self.coins) and self.coins[i][2]:
                obs.append(self.coins[i][0] / self.width)
                obs.append(self.coins[i][1] / self.height)
            else:
                obs.append(-1.0)
                obs.append(-1.0)

        # asteroid
        if self.asteroid and self.asteroid[2]:
            obs.append(self.asteroid[0] / self.width)
            obs.append(self.asteroid[1] / self.height)
        else:
            obs.append(-1.0)
            obs.append(-1.0)

        return np.array(obs, dtype=np.float32)

    def _draw_ship(self, surf):
        x, y = int(self.ship_x), int(self.ship_y)
        body_color = (180, 180, 255)
        wing_color = (100, 100, 200)

        # body (nose pointing right)
        pygame.draw.polygon(
            surf, body_color,
            [
                (x, y),                              # back-top
                (x, y + self.ship_h),                # back-bottom
                (x + self.ship_w, y + self.ship_h // 2)  # nose
            ]
        )

        # wings (back-left)
        pygame.draw.line(
            surf, wing_color,
            (x + 4, y - 2), (x + 4, y + self.ship_h + 2), 2
        )

        # --- Gradient flame with flicker + width pulse ---
        speed = np.hypot(self.vel_x, self.vel_y)
        if speed > 0.1:
            base_len = self.ship_w * 0.5
            flame_len = min(base_len * 2.0, base_len + speed * 4.0)

            # add ±10% flicker to length
            flicker = 1.0 + self.np_random.uniform(-0.1, 0.1)
            flame_len *= flicker

            # flame width (baseline = 4 px, scales with speed)
            base_width = 4.0
            flame_width = base_width + speed * 0.3
            # add small jitter to width as well
            flame_width *= 1.0 + self.np_random.uniform(-0.1, 0.1)

            cx = x
            cy = y + self.ship_h // 2

            # flame layers: (color, shrink factor)
            layers = [
                ((255, 255, 120), 1.0),   # bright yellow core
                ((255, 200, 60), 0.7),    # orange
                ((255, 140, 0), 0.5),     # deep orange
            ]

            for color, f in layers:
                tip_x = cx - flame_len * f
                width = int(flame_width * f)
                pygame.draw.polygon(
                    surf, color,
                    [
                        (tip_x, cy),           # flame tip
                        (cx, cy - width),      # top attach
                        (cx, cy + width),      # bottom attach
                    ]
                )

    def _draw_asteroid(self, surf, x, y, angle, shape):
        points = []
        for a, r in shape:
            # rotate + translate
            px = x + np.cos(a + angle) * r
            py = y + np.sin(a + angle) * r
            points.append((px, py))
        pygame.draw.polygon(surf, (100, 100, 100), points)
        pygame.draw.polygon(surf, (60, 60, 60), points, 2)

    def _draw_coin(self, surf, x, y):
        pygame.draw.circle(surf, (255, 215, 0), (int(x), int(y)), self.coin_radius)
        pygame.draw.circle(surf, (200, 160, 0), (int(x), int(y)), self.coin_radius, 2)

    def render(self):
        if self._surface is None:
            pygame.init()
            self._surface = pygame.Surface((self.width, self.height))
            self.clock = pygame.time.Clock()

        self._surface.fill((0, 0, 0))  # space background
        for s in self.stars:
            pygame.draw.circle(self._surface, (255, 255, 255), (int(s[0]), int(s[1])), 1)

        self._draw_ship(self._surface)
        for coin in self.coins:
            if coin[2]:
                self._draw_coin(self._surface, coin[0], coin[1])

        if self.asteroid and self.asteroid[2]:
            self._draw_asteroid(
                self._surface,
                self.asteroid[0], self.asteroid[1],
                self.asteroid[3], self.asteroid[4]
            )

        if self.collided:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((220, 60, 60, 40))
            self._surface.blit(overlay, (0, 0))

        return np.transpose(np.array(pygame.surfarray.pixels3d(self._surface)), (1, 0, 2)).copy()

    def close(self):
        if self._surface is not None:
            pygame.quit()
            self._surface = None
