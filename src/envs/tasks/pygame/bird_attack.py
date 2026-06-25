import gymnasium as gym
import numpy as np
import pygame


class BirdAttackEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, max_episode_steps=500):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.width, self.height = 224, 224
        self.area_size = 184
        self.area_x = (self.width - self.area_size) // 2
        self.area_y = (self.height - self.area_size) // 2

        # agent
        self.agent_w, self.agent_h = 14, 8
        self.agent_x = self.width // 2 - self.agent_w // 2
        self.agent_y = self.area_y + self.area_size - 20
        self.agent_speed = 6.0
        self.reload_timer = 8

        # birds
        self.bird_w, self.bird_h = 14, 12
        self.bird_rows, self.bird_cols = 2, 8
        self.birds = []
        self.bird_dir = 1  # 1=right, -1=left
        self.bird_speed = 1.0
        self.bird_drop = 4

        # boss bird
        self.boss_spawn_step = 250
        self.boss_active = False
        self.boss_pos = None   # [x, y]
        self.boss_bullet = None  # [x, y, vx, vy]
        self.boss_fire_timer = 0
        self.boss_size = 28
        self.boss_bullet_size = 10
        self.boss_bullet_speed = 5.0
        # Boss geometry + fade
        self.boss_span = 64          # wingtip-to-wingtip (px)
        self.boss_height = 16        # vertical amplitude of the "M"
        self.boss_fadein = 0         # frames remaining while fading in

        # projectiles
        self.agent_bullets = []
        self.bird_bullets = []
        self.bullet_speed = 7

        self.max_agent_bullets = 5
        self.max_bird_bullets = 10
        max_birds = self.bird_rows * self.bird_cols
        obs_dim = (
            3                           # [agent_x, reload, num_birds]
            + max_birds                 # bird x's
            + 1                         # lowest bird y
            + self.max_agent_bullets*2  # agent bullets (x,y)×5
            + self.max_bird_bullets*2   # bird bullets  (x,y)×10
            + 1                         # stun
            + 4                         # boss bullet (x,y,vx,vy)
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # continuous horizontal thrust in [-1,1] + fire button [-1,1]
        self.action_space = gym.spaces.Box(
            low=-np.ones(2, dtype=np.float32), high=np.ones(2, dtype=np.float32)
        )

        self.score = 0
        self.steps = 0
        self.stun_timer = 0

        self._surface = None
        self.font = None
        self.clock = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_x = self.width // 2 - self.agent_w // 2
        self.reload_timer = 0
        self.agent_bullets.clear()
        self.bird_bullets.clear()
        self.score = 0
        self.steps = 0
        self.stun_timer = 0

        self.boss_active = False
        self.boss_pos = None
        self.boss_bullet = None
        self.boss_fire_timer = 0

        # init birds in grid
        self.birds = []
        start_x = self.area_x + 20
        start_y = self.area_y + 20
        spacing_x = 20
        spacing_y = 20
        for r in range(self.bird_rows):
            for c in range(self.bird_cols):
                self.birds.append([start_x + c*spacing_x,
                                start_y + r*spacing_y,
                                2])   # hp = 2
        self.bird_dir = 1

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1, 1)
        move, fire = float(action[0]), float(action[1])
        reward = 0.0
        terminated, truncated = False, False

        # move agent
        speed = self.agent_speed
        if self.stun_timer > 0:
            speed *= 0.2  # lower speed when stunned
        self.agent_x += move * speed
        self.agent_x = np.clip(self.agent_x, self.area_x,
                               self.area_x + self.area_size - self.agent_w)
        if self.stun_timer > 0:
            self.stun_timer -= 1

        # fire
        if fire > 0 and self.reload_timer == 0:
            muzzle_x = self.agent_x + self.agent_w // 2
            muzzle_y = self.agent_y - 6
            self.agent_bullets.append([muzzle_x, muzzle_y])
            self.reload_timer = 8
        if self.reload_timer > 0:
            self.reload_timer -= 1

        # move agent bullets
        for b in self.agent_bullets:
            b[1] -= self.bullet_speed
        self.agent_bullets = [b for b in self.agent_bullets if b[1] > self.area_y]

        # bird movement
        shift = self.bird_speed * self.bird_dir
        xs = [bx for bx, by, hp in self.birds]
        if xs:
            if max(xs) + self.bird_w + shift > self.area_x + self.area_size or min(xs) + shift < self.area_x:
                self.bird_dir *= -1
                for b in self.birds:
                    b[1] += self.bird_drop
            else:
                for b in self.birds:
                    b[0] += shift

        # birds shoot randomly
        for bx, by, hp in self.birds:
            if self.np_random.random() < 0.0075:
                self.bird_bullets.append([bx + self.bird_w//2, by + self.bird_h])

        # move bird bullets
        for b in self.bird_bullets:
            b[1] += self.bullet_speed
        self.bird_bullets = [b for b in self.bird_bullets if b[1] < self.area_y + self.area_size]

        new_birds = []
        for bx, by, hp in self.birds:
            for bullet in self.agent_bullets:
                if abs(bullet[0] - (bx+self.bird_w/2)) < self.bird_w/2 and \
                abs(bullet[1] - (by+self.bird_h/2)) < self.bird_h/2:
                    hp -= 1
                    self.agent_bullets.remove(bullet)
                    if hp == 0:
                        reward += 1.0
                        self.score += 1
                    break
            if hp > 0:
                new_birds.append([bx, by, hp])
        self.birds = new_birds

        # collisions: bird bullets vs agent
        ax, ay = self.agent_x, self.agent_y
        for bullet in self.bird_bullets:
            if abs(bullet[0] - (ax+self.agent_w/2)) < self.agent_w/2 and \
            abs(bullet[1] - (ay+self.agent_h/2)) < self.agent_h/2:
                reward -= 2.0
                self.bird_bullets.remove(bullet)
                self.stun_timer = 16      # 16-frame stun
                self.reload_timer = 16     # reset reload

        if not self.boss_active and self.steps >= self.boss_spawn_step:
            self.boss_active = True
            cx = self.area_x + self.area_size // 2
            cy = self.area_y + 20
            self.boss_pos = [cx, cy]      # store as CENTER (cx, cy)
            self.boss_fire_timer = 50
            self.boss_fadein = 5          # 5 frames: no move/fire

        if self.boss_active:
            if self.boss_fadein > 0:
                self.boss_fadein -= 1
            else:
                # jitter within ±10% of area, but keep full M inside bounds
                half_w = self.boss_span / 2
                half_h = self.boss_height / 2
                spawn_cx = self.area_x + self.area_size // 2
                spawn_cy = self.area_y + 20
                jx = self.np_random.uniform(-0.03, 0.03) * self.area_size
                jy = self.np_random.uniform(-0.01, 0.01) * self.area_size
                cx = np.clip(spawn_cx + jx, self.area_x + half_w, self.area_x + self.area_size - half_w)
                cy = np.clip(spawn_cy + jy, self.area_y + half_h, self.area_y + self.area_size // 2 - half_h)
                self.boss_pos[0], self.boss_pos[1] = cx, cy

                # fire
                self.boss_fire_timer -= 1
                if self.boss_fire_timer <= 0 and self.boss_bullet is None:
                    cx, cy = self.boss_pos
                    # fire from the center dip of the M (lower y)
                    origin_x = cx
                    origin_y = cy + (self.boss_height / 2)

                    ax = self.agent_x + self.agent_w / 2
                    ay = self.agent_y + self.agent_h / 2
                    dx, dy = ax - origin_x, ay - origin_y
                    n = max(1e-6, np.hypot(dx, dy))
                    vx = self.boss_bullet_speed * dx / n
                    vy = self.boss_bullet_speed * dy / n

                    self.boss_bullet = [origin_x, origin_y, vx, vy]
                    self.boss_fire_timer = 50

        # boss bullet move + bounce + collision
        if self.boss_bullet is not None:
            self.boss_bullet[0] += self.boss_bullet[2]
            self.boss_bullet[1] += self.boss_bullet[3]

            # bounce on vertical walls
            if (self.boss_bullet[0] - self.boss_bullet_size < self.area_x) or \
            (self.boss_bullet[0] + self.boss_bullet_size > self.area_x + self.area_size):
                self.boss_bullet[2] *= -1

            # despawn at bottom border (stop rendering past agent/border)
            if self.boss_bullet[1] - self.boss_bullet_size >= self.area_y + self.area_size:
                self.boss_bullet = None

            # collision
            if self.boss_bullet is not None:
                bx, by, vx, vy = self.boss_bullet
                if abs(bx - (ax+self.agent_w/2)) < (self.agent_w/2 + self.boss_bullet_size) and \
                abs(by - (ay+self.agent_h/2)) < (self.agent_h/2 + self.boss_bullet_size):
                    reward -= 2.0
                    self.stun_timer = 16
                    self.reload_timer = 16
                    self.boss_bullet = None  # bullet consumed
        
        # Bonus treat for when there are no birds left
        if len(self.birds) == 0:
            reward += 1.0

        self.steps += 1
        if self.steps >= self.max_episode_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        max_birds = self.bird_rows * self.bird_cols

        bird_xs = [bx / self.width for bx, by, hp in self.birds]
        bird_xs += [0.0] * (max_birds - len(bird_xs))

        lowest_y = max([by for bx, by, hp in self.birds], default=self.area_y) / self.height

        agent_bullets = [(x/self.width, y/self.height) for x,y in self.agent_bullets[:self.max_agent_bullets]]
        agent_bullets += [(0.0, 0.0)] * (self.max_agent_bullets - len(agent_bullets))
        agent_bullets_flat = [v for xy in agent_bullets for v in xy]

        bird_bullets = [(x/self.width, y/self.height) for x,y in self.bird_bullets[:self.max_bird_bullets]]
        bird_bullets += [(0.0, 0.0)] * (self.max_bird_bullets - len(bird_bullets))
        bird_bullets_flat = [v for xy in bird_bullets for v in xy]

        # Boss bullet (x,y,vx,vy)
        if self.boss_bullet is not None:
            bx, by, vx, vy = self.boss_bullet
            boss_xy = (np.clip(bx/self.width, 0.0, 1.0),
                    np.clip(by/self.height, 0.0, 1.0))
            s = max(1e-6, self.boss_bullet_speed)
            boss_v = (np.clip(vx/s, -1.0, 1.0),
                    np.clip(vy/s, -1.0, 1.0))
        else:
            boss_xy = (0.0, 0.0)
            boss_v  = (0.0, 0.0)

        return np.array(
            [self.agent_x / self.width,
            self.reload_timer / 5.0,
            len(self.birds) / max_birds,
            *bird_xs,
            lowest_y,
            *agent_bullets_flat,
            *bird_bullets_flat,
            self.stun_timer / 10.0,
            *boss_xy,
            *boss_v],
            dtype=np.float32
        )

    def render(self):
        if self._surface is None:
            pygame.init()
            self._surface = pygame.Surface((self.width, self.height))
            self.clock = pygame.time.Clock()
        if self.font is None:
            self.font = pygame.font.SysFont("Arial", 18, bold=True)

        self._surface.fill((0, 0, 0))
        # border
        pygame.draw.rect(self._surface, (255,255,255),
                         pygame.Rect(self.area_x, self.area_y, self.area_size, self.area_size), 2)

        # agent body
        pygame.draw.rect(self._surface, (255,140,0),
                        pygame.Rect(int(self.agent_x), int(self.agent_y),
                                    self.agent_w, self.agent_h))

        # small triangle muzzle on top-center
        mx = self.agent_x + self.agent_w // 2
        my = self.agent_y
        pygame.draw.polygon(self._surface, (255,140,0),
                            [(mx-4, my), (mx+4, my), (mx, my-6)])

        # birds (draw as "V")
        for bx, by, hp in self.birds:
            points = [(bx, by),
                    (bx+self.bird_w//2, by+self.bird_h),
                    (bx+self.bird_w, by)]
            if hp == 2:
                pygame.draw.polygon(self._surface, (200,200,200), points)
            else:
                pygame.draw.polygon(self._surface, (200,200,200), points, 2)

        # bullets
        for x,y in self.agent_bullets:
            pygame.draw.rect(self._surface, (255,140,0),
                             pygame.Rect(int(x-2), int(y-6), 4, 6))
        for x,y in self.bird_bullets:
            pygame.draw.rect(self._surface, (200,200,200),
                             pygame.Rect(int(x-2), int(y), 4, 6))
            
        # boss
        if self.boss_active and self.boss_pos is not None:
            cx, cy = self.boss_pos
            half_w = self.boss_span / 2
            half_h = self.boss_height / 2

            # "M" polyline (left tip → left peak → center dip → right peak → right tip)
            points = [
                (int(cx - half_w),         int(cy + half_h)),  # left tip (lower)
                (int(cx - half_w * 0.5),   int(cy - half_h)),  # left peak (upper)
                (int(cx),                  int(cy + half_h)),  # center dip (lower)
                (int(cx + half_w * 0.5),   int(cy - half_h)),  # right peak (upper)
                (int(cx + half_w),         int(cy + half_h)),  # right tip (lower)
            ]

            # Fade alpha: 0→255 over 5 frames
            alpha = 255 if self.boss_fadein == 0 else int(255 * (1.0 - self.boss_fadein / 5.0))

            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.lines(overlay, (200, 200, 200, alpha), False, points, 3)  # outline width=3
            self._surface.blit(overlay, (0, 0))

        # boss bullet
        if self.boss_bullet is not None:
            bx, by, vx, vy = self.boss_bullet
            pygame.draw.circle(self._surface, (200,200,200),
                            (int(bx), int(by)), self.boss_bullet_size)

        # score
        max_birds = self.bird_rows * self.bird_cols
        text_color = (80, 200, 110) if self.score >= max_birds else (255, 255, 255)
        text = self.font.render(str(self.score), True, text_color)
        self._surface.blit(text, (self.area_x+4, self.area_y-20))

        # stun
        if self.stun_timer > 0:
            factor = self.stun_timer / 16.0  # 1.0 → 0.0
        else:
            factor = 0.0
        # red flash overlay (game area only)
        if self.stun_timer > 0:
            alpha = int(140 * factor)  # max opacity ~140 (out of 255)
            overlay = pygame.Surface((self.area_size, self.area_size))
            overlay.set_alpha(alpha)
            overlay.fill((220, 30, 10))
            self._surface.blit(overlay, (self.area_x, self.area_y))

        return np.transpose(np.array(pygame.surfarray.pixels3d(self._surface)), (1,0,2)).copy()

    def close(self):
        if self._surface is not None:
            pygame.quit()
            self._surface = None
