import gymnasium as gym
import numpy as np
import pygame


class PongEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, max_episode_steps=500):
        super().__init__()
        self.max_episode_steps = max_episode_steps

        # obs = [ball_x, ball_y, ball_vx, ball_vy, paddle_y, opp_y]
        obs_dim = 6
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # continuous vertical thrust in [-1,1]
        self.action_space = gym.spaces.Box(
            low=-np.ones(1, dtype=np.float32), high=np.ones(1, dtype=np.float32)
        )

        self.width, self.height = 224, 224
        self.area_size = 184
        self.area_x = (self.width - self.area_size) // 2
        self.area_y = (self.height - self.area_size) // 2

        # paddles
        self.paddle_w, self.paddle_h = 6, 40
        self.agent_x = self.area_x + 10
        self.opp_x = self.area_x + self.area_size - 10 - self.paddle_w
        self.agent_y = self.opp_y = self.height // 2 - self.paddle_h // 2
        self.paddle_speed = 8.5
        
        # opponent
        self.reaction_delay = 0
        self.in_reaction_zone = False

        # ball
        self.ball_size = 7
        self.base_ball_speed = 8.0
        self.reset_ball()

        self.agent_score = 0
        self.opp_score = 0
        self.font = None  # will init on first render
        self.steps = 0
        self._surface = None
        self.clock = None

    def reset_ball(self):
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        angle = self.np_random.uniform(-0.5, 0.5) * np.pi
        speed = self.base_ball_speed
        direction = self.np_random.choice([-1, 1])
        self.ball_vx = direction * speed * np.cos(angle)
        self.ball_vy = speed * np.sin(angle)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_y = self.height // 2 - self.paddle_h // 2
        self.opp_y = self.height // 2 - self.paddle_h // 2
        self.reset_ball()
        self.agent_score = 0
        self.opp_score = 0
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        action = float(np.clip(action[0], -1, 1))
        terminated = False
        truncated = False
        reward = 0.0

        # agent paddle move
        self.agent_y += action * self.paddle_speed
        self.agent_y = np.clip(self.agent_y, self.area_y,
                               self.area_y + self.area_size - self.paddle_h)

        # --- Opponent AI ---
        progress = (self.ball_x - self.area_x) / self.area_size  # 0.0=agent side, 1.0=opponent side

        if progress > 0.3:
            if not self.in_reaction_zone:
                # Ball just entered reaction zone → sample new delay
                speed = np.hypot(self.ball_vx, self.ball_vy)

                # normalize ball speed to [0,1] between [min_speed, max_speed]
                min_speed, max_speed = 3.0, 8.0
                norm = np.clip((speed - min_speed) / (max_speed - min_speed), 0.0, 1.0)

                # interpolate delay range: slow → [6,8], fast → [2,4]
                slow_min, slow_max = 6, 9
                fast_min, fast_max = 2, 6
                low = int(slow_min + (fast_min - slow_min) * norm)
                high = int(slow_max + (fast_max - slow_max) * norm)

                self.reaction_delay = self.np_random.integers(low, high + 1)
                self.in_reaction_zone = True

            if self.reaction_delay > 0:
                self.reaction_delay -= 1
            else:
                # interp speed factor: 0.5 at 0.3 progress → 0.85 at 1.0
                t = (progress - 0.3) / 0.65
                speed_factor = 0.5 + 0.35 * np.clip(t, 0.0, 1.0)

                target_y = self.ball_y - self.paddle_h / 2
                if target_y > self.opp_y:
                    self.opp_y += self.paddle_speed * speed_factor
                else:
                    self.opp_y -= self.paddle_speed * speed_factor
        else:
            # ball not in opponent’s zone → reset flag
            self.in_reaction_zone = False

        # always clamp inside the play area
        self.opp_y = np.clip(self.opp_y, self.area_y,
                            self.area_y + self.area_size - self.paddle_h)

        # move ball
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # top/bottom collision
        if self.ball_y - self.ball_size < self.area_y or self.ball_y + self.ball_size > self.area_y + self.area_size:
            self.ball_vy *= -1

        # paddle collisions
        agent_rect = pygame.Rect(self.agent_x, int(self.agent_y),
                                 self.paddle_w, self.paddle_h)
        opp_rect = pygame.Rect(self.opp_x, int(self.opp_y),
                               self.paddle_w, self.paddle_h)
        ball_rect = pygame.Rect(int(self.ball_x - self.ball_size/2),
                                int(self.ball_y - self.ball_size/2),
                                self.ball_size, self.ball_size)
        if ball_rect.colliderect(agent_rect) and self.ball_vx < 0:
            self.ball_vx *= -1
        if ball_rect.colliderect(opp_rect) and self.ball_vx > 0:
            self.ball_vx *= -1

        # scoring
        if self.ball_x < self.area_x:      # opponent scores
            reward = -1.0
            self.opp_score += 1
            self.reset_ball()
        elif self.ball_x > self.area_x + self.area_size:  # agent scores
            reward = 1.0
            self.agent_score += 1
            self.reset_ball()

        self.steps += 1
        if self.steps >= self.max_episode_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        return np.array([
            self.ball_x / self.width,
            self.ball_y / self.height,
            self.ball_vx / 8.0,
            self.ball_vy / 8.0,
            self.agent_y / self.height,
            self.opp_y / self.height,
        ], dtype=np.float32)

    def render(self):
        if self._surface is None:
            pygame.init()
            self._surface = pygame.Surface((self.width, self.height))
            self.clock = pygame.time.Clock()
        if self.font is None:
            self.font = pygame.font.SysFont("Arial", 20, bold=True)

        self._surface.fill((0, 0, 0))

        # draw border
        pygame.draw.rect(
            self._surface, (255, 255, 255),
            pygame.Rect(self.area_x, self.area_y, self.area_size, self.area_size),
            2
        )

        # draw paddles
        pygame.draw.rect(
            self._surface, (255, 140, 0),
            pygame.Rect(self.agent_x, int(self.agent_y), self.paddle_w, self.paddle_h)
        )
        pygame.draw.rect(
            self._surface, (180, 180, 180),
            pygame.Rect(self.opp_x, int(self.opp_y), self.paddle_w, self.paddle_h)
        )

        # draw ball
        pygame.draw.rect(
            self._surface, (255, 255, 255),
            pygame.Rect(int(self.ball_x - self.ball_size/2),
                        int(self.ball_y - self.ball_size/2),
                        self.ball_size, self.ball_size)
        )

        # scores (above the border, centered left/right)
        agent_text = self.font.render(str(self.agent_score), True, (255, 140, 0))
        opp_text   = self.font.render(str(self.opp_score), True, (180, 180, 180))

        # x-positions roughly aligned with paddles
        agent_x = self.agent_x + self.paddle_w
        opp_x   = self.opp_x - opp_text.get_width()

        y = self.area_y - 22  # above the top border
        self._surface.blit(agent_text, (agent_x, y))
        self._surface.blit(opp_text, (opp_x, y))

        return np.transpose(np.array(pygame.surfarray.pixels3d(self._surface)), (1, 0, 2)).copy()

    def close(self):
        if self._surface is not None:
            pygame.quit()
            self._surface = None
