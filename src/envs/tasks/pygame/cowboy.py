import gymnasium as gym
import numpy as np
import pygame


class CowboyEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, max_episode_steps=500):
        super().__init__()
        self.max_episode_steps = max_episode_steps

        # obs = [agent_y, agent_vel_y, obstacle_x]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        # continuous jump force normalized to [-1,1]
        self.action_space = gym.spaces.Box(low=np.array([-1.0], dtype=np.float32),
                                       high=np.array([1.0], dtype=np.float32))

        self.width, self.height = 224, 224
        self.ground_y = self.height - 40
        self.gravity = 1.0  # positive = pulls down in pygame coordinates
        self.jump_strength = 12  # pixels per step for a full jump
        self.scroll_speed = 10.0

        # cowboy sprite
        self.agent_width = 18
        self.agent_height = 30
        self.agent_x = 40

        self._surface = None
        self.clock = None

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_y = self.ground_y - self.agent_height
        self.agent_vel_y = 0.0
        self.agent_y = max(self.agent_y, self.ground_y - self.agent_height)
        self.steps = 0
        self.collided = False
        self.collision_offset = 0.0
        self.leg_pose = 0  # 0 or 1
        self.arm_pose = 0  # 0 or 1
        self.pose_timer = 0
        self.spawn_obstacle()

        # Pebbles + dust
        self.num_pebbles = 10
        self.pebbles = []
        self.dust_particles = []
        for _ in range(self.num_pebbles):
            x = self.width + self.np_random.uniform(0, self.width)
            y = self.ground_y + self.np_random.uniform(1, 4)
            self.pebbles.append([x, y])

        # Sun parameters (randomized per episode)
        self.sun_radius = int(self.np_random.integers(12, 20))  # 12–20 px
        self.sun_center = (
            self.width - int(self.np_random.integers(20, 40)),  # x offset from right edge
            int(self.np_random.integers(15, 35)),              # y offset from top
        )
        base_color = (230, 212, 100)
        jitter = self.np_random.integers(-10, 10, size=3)
        self.sun_color = tuple(
            np.clip(base_color[i] + jitter[i], 0, 255) for i in range(3)
        )

        return self._get_obs(), {}

    def spawn_obstacle(self):
        variant = self.np_random.integers(0, 4)
        base_x = self.width + 40
        if variant == 0:
            self.obstacle_rects = [pygame.Rect(base_x, self.ground_y - 25, 15, 25)]
        elif variant == 1:
            self.obstacle_rects = [pygame.Rect(base_x, self.ground_y - 40, 20, 40)]
        elif variant == 2:
            self.obstacle_rects = [
                pygame.Rect(base_x, self.ground_y - 25, 15, 25),
                pygame.Rect(base_x + 20, self.ground_y - 40, 20, 40),
            ]
        else:
            self.obstacle_rects = [
                pygame.Rect(base_x, self.ground_y - 40, 20, 40),
                pygame.Rect(base_x + 25, self.ground_y - 25, 15, 25),
            ]
        left = min(r.left for r in self.obstacle_rects)
        top = min(r.top for r in self.obstacle_rects)
        right = max(r.right for r in self.obstacle_rects)
        bottom = max(r.bottom for r in self.obstacle_rects)
        self.obstacle_hitbox = pygame.Rect(left, top, right - left, bottom - top)

    def step(self, action):
        action = float(np.clip(action[0], -1, 1))
        jumping = False
        terminated = False
        truncated = False

        if action > 0 and self.agent_y >= self.ground_y - self.agent_height - 1e-3:
            jump_impulse = max(action, 0) * self.jump_strength
            self.agent_vel_y = -jump_impulse  # negative because y=0 is top of screen
            jumping = True

        # Update vertical position
        self.agent_vel_y += self.gravity
        self.agent_y += self.agent_vel_y

        # Ground collision
        if self.agent_y > self.ground_y - self.agent_height:
            self.agent_y = self.ground_y - self.agent_height
            self.agent_vel_y = 0.0

        # Spawn dust on jump/land
        if jumping:
            for _ in range(5):
                self.dust_particles.append({
                    "x": self.agent_x + self.agent_width // 2,
                    "y": self.ground_y - 2,
                    "vx": self.np_random.uniform(-1, 1),
                    "vy": self.np_random.uniform(-2, 0),
                    "life": 20,
                })

        # Update dust particles
        for p in self.dust_particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vy"] += 0.1  # gravity on dust
            p["life"] -= 1
        self.dust_particles = [p for p in self.dust_particles if p["life"] > 0]

        # Obstacle top collision (landing on top)
        agent_rect = pygame.Rect(self.agent_x, int(self.agent_y),
                         self.agent_width, self.agent_height)
        if agent_rect.colliderect(self.obstacle_hitbox):
            # Falling downward
            if self.agent_vel_y > 0:
                # Calculate horizontal overlap
                overlap_x = min(agent_rect.right, self.obstacle_hitbox.right) - max(agent_rect.left, self.obstacle_hitbox.left)
                overlap_frac = overlap_x / self.agent_width

                # Only snap if agent is mostly above obstacle (e.g., >50% overlap)
                if overlap_frac > 0.5 and self.agent_y + self.agent_height > self.obstacle_hitbox.top:
                    self.agent_y = self.obstacle_hitbox.top - self.agent_height
                    self.agent_vel_y = 0.0

        agent_rect = pygame.Rect(self.agent_x, int(self.agent_y), self.agent_width, self.agent_height)

        # Check collision
        if agent_rect.colliderect(self.obstacle_hitbox):
            if not self.collided:
                # First frame of collision
                self.collided = True
                self.collision_offset = self.agent_x + self.agent_width - self.obstacle_hitbox.left

            for r in self.obstacle_rects:
                r.left = self.obstacle_hitbox.left + (r.left - self.obstacle_hitbox.left)
            self.obstacle_hitbox.left = self.agent_x + self.agent_width - self.collision_offset
        else:
            # Agent cleared obstacle: resume scrolling
            self.collided = False
            self.collision_offset = 0.0

        # Scroll obstacles and pebbles only if not colliding
        if not self.collided:
            # Obstacles
            self.obstacle_hitbox.move_ip(-self.scroll_speed, 0)
            for r in self.obstacle_rects:
                r.move_ip(-self.scroll_speed, 0)

            # Pebbles
            for p in self.pebbles:
                p[0] -= self.scroll_speed
                if p[0] < 0:
                    p[0] = self.width + self.np_random.uniform(0, 20)
                    p[1] = self.ground_y + self.np_random.uniform(1, 4)

        if self.obstacle_hitbox.right < 0:
            self.spawn_obstacle()
            self.collided = False

        # animate legs and arms if running
        if self.agent_y >= self.ground_y - self.agent_height - 1e-3 and not self.collided:
            self.pose_timer += 1
            if self.pose_timer % 6 == 0:
                self.leg_pose = 1 - self.leg_pose
                self.arm_pose = 1 - self.arm_pose

        self.steps += 1
        if self.steps >= self.max_episode_steps:
            truncated = True

        reward = 1.0 if not self.collided else -1.

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        obs = np.array([
            (self.agent_y - (self.ground_y - self.agent_height)) / self.height,
            self.agent_vel_y / 10.0,
            self.obstacle_hitbox.left / self.width,
        ], dtype=np.float32)
        return obs

    def _draw_cowboy(self, surf):
        x, y = self.agent_x, int(self.agent_y)
        pygame.draw.rect(surf, (139, 69, 19), (x + 6, y + 10, 6, 15))  # torso
        pygame.draw.circle(surf, (222, 184, 135), (x + 9, y + 5), 5)  # head
        pygame.draw.line(surf, (0,0,0), (x, y+1), (x+18, y+1), 2)  # hat brim
        pygame.draw.polygon(surf, (0,0,0), [(x+5,y-3),(x+13,y-3),(x+9,y-8)])  # hat crown
        # arms (animated)
        if self.arm_pose == 0:
            pygame.draw.line(surf, (0,0,0), (x+6, y+12), (x-2, y+18), 2)
            pygame.draw.line(surf, (0,0,0), (x+12, y+12), (x+20, y+18), 2)
        else:
            pygame.draw.line(surf, (0,0,0), (x+6, y+12), (x, y+20), 2)
            pygame.draw.line(surf, (0,0,0), (x+12, y+12), (x+22, y+20), 2)
        # legs (animated)
        if self.leg_pose == 0:
            pygame.draw.line(surf, (0,0,0), (x+6, y+25), (x, y+35), 2)
            pygame.draw.line(surf, (0,0,0), (x+12, y+25), (x+18, y+35), 2)
        else:
            pygame.draw.line(surf, (0,0,0), (x+6, y+25), (x-2, y+35), 2)
            pygame.draw.line(surf, (0,0,0), (x+12, y+25), (x+20, y+35), 2)

    def _draw_cactus(self, surf):
        for r in self.obstacle_rects:
            pygame.draw.rect(surf, (108, 138, 72), r)
            mid = r.top + r.height // 3
            pygame.draw.rect(surf, (108, 138, 72), (r.left - 4, mid, 4, 10))
            pygame.draw.rect(surf, (108, 138, 72), (r.right, mid - 5, 4, 10))

    def _draw_sky_gradient(self, surf):
        top_color = (150, 190, 220)   # darker top
        bottom_color = (210, 220, 235)  # lighter near horizon
        for y in range(self.height):
            t = y / self.height
            r = int(top_color[0] * (1 - t) + bottom_color[0] * t)
            g = int(top_color[1] * (1 - t) + bottom_color[1] * t)
            b = int(top_color[2] * (1 - t) + bottom_color[2] * t)
            pygame.draw.line(surf, (r, g, b), (0, y), (self.width, y))

    def _draw_ground_gradient(self, surf):
        top_color = (196, 166, 135)   # lighter at horizon
        bottom_color = (140, 110, 90)  # darker deeper down
        h = self.height - self.ground_y
        for i in range(h):
            t = i / h
            r = int(top_color[0] * (1 - t) + bottom_color[0] * t)
            g = int(top_color[1] * (1 - t) + bottom_color[1] * t)
            b = int(top_color[2] * (1 - t) + bottom_color[2] * t)
            y = self.ground_y + i
            pygame.draw.line(surf, (r, g, b), (0, y), (self.width, y))

    def _draw_dust(self, surf):
        for p in self.dust_particles:
            alpha = int(255 * (p["life"] / 20))
            dust = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(dust, (150, 120, 100, alpha), (2, 2), 2)
            surf.blit(dust, (int(p["x"]), int(p["y"])))

    def _draw_sun(self, surf):
        # Soft glow
        for r in range(self.sun_radius * 2, self.sun_radius, -4):
            alpha = max(0, 30 - (self.sun_radius * 2 - r) // 2)
            glow = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.circle(glow, (*self.sun_color, alpha), self.sun_center, r)
            surf.blit(glow, (0, 0))

        # Core
        pygame.draw.circle(surf, self.sun_color, self.sun_center, self.sun_radius)

    def render(self):
        if self._surface is None:
            pygame.init()
            self._surface = pygame.Surface((self.width, self.height))
            self.clock = pygame.time.Clock()

        # Sky gradient
        self._draw_sky_gradient(self._surface)

        # Ground gradient
        self._draw_ground_gradient(self._surface)

        # Ground line
        pygame.draw.line(
            self._surface, (115, 90, 65),
            (0, self.ground_y), (self.width, self.ground_y), 2
        )

        # Draw pebbles
        ground_surface_color = (115, 90, 65)
        for p in self.pebbles:
            pygame.draw.circle(self._surface, ground_surface_color, (int(p[0]), int(p[1])), 1)

        self._draw_cowboy(self._surface)
        self._draw_cactus(self._surface)
        self._draw_dust(self._surface)
        self._draw_sun(self._surface)

        # Add red overlay if in collision
        if self.collided:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((220, 60, 60, 20))
            self._surface.blit(overlay, (0, 0))

        return np.transpose(np.array(pygame.surfarray.pixels3d(self._surface)), (1, 0, 2)).copy()

    def close(self):
        if self._surface is not None:
            pygame.quit()
            self._surface = None
