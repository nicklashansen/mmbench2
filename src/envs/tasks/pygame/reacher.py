import gymnasium as gym
import numpy as np
import pygame


# --- Color palette ---
C_TEAL   = (120, 160, 160)
C_PURPLE = (115,  95, 165)
C_PINK   = (160,  90, 130)
C_WHITE  = (230, 230, 230)


class ReacherEnv(gym.Env):
    """
    2-link planar reacher.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self,
                max_episode_steps: int = 200,
                target_size_m: float = 0.04,
                finger_size_m: float = 0.01,
                l1_m: float = 0.11,  # link 1 length
                l2_m: float = 0.11,  # link 2 length
                ):
        super().__init__()
        self.max_episode_steps = max_episode_steps

        # --- geometry (meters) ---
        self.L1_m, self.L2_m = float(l1_m), float(l2_m)
        self.reach_m = self.L1_m + self.L2_m

        # --- joint limits / dynamics (torque-controlled hinges) ---
        self.q1_min, self.q1_max = -np.inf, np.inf
        self.q2_min, self.q2_max = np.deg2rad(-160.0), np.deg2rad(160.0)

        self.dt = 0.02
        self.torque_scale = np.array([90, 180], dtype=np.float32) # per-joint
        self.d_lin = np.array([4.0, 38.0], dtype=np.float32) # s^-1
        self.d_quad = np.array([1.2, 0.05], dtype=np.float32) # s^-1 per |w|
        self.max_w = 30.0

        # sizes
        self.target_r_m = float(target_size_m)
        self.finger_r_m = float(finger_size_m)

        # 6D observation space: position, to_target, velocity
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        # actions = torques in [-1, 1]^2
        self.action_space = gym.spaces.Box(
            low=-np.ones(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
        )

        # --- rendering (pixels) ---
        self.width, self.height = 224, 224
        self.area_size = 184
        self.area_x = (self.width - self.area_size) // 2
        self.area_y = (self.height - self.area_size) // 2
        self.cx = self.area_x + self.area_size // 2
        self.cy = self.area_y + self.area_size // 2
        self.px_per_m = (self.area_size * 0.45) / self.reach_m

        self.link_thickness_px = 6
        self.joint_r_px = 4

        # state
        self.q1 = self.q2 = 0.0
        self.w1 = self.w2 = 0.0
        self.tx_m = self.ty_m = 0.0
        self.steps = 0

        self._surface = None
        self.clock = None

    # ------------------ Kinematics (meters) ------------------

    def _fk_m(self):
        """Forward kinematics in meters (world frame x-right, y-up)."""
        x0, y0 = 0.0, 0.0
        x1 = x0 + self.L1_m * np.cos(self.q1)
        y1 = y0 + self.L1_m * np.sin(self.q1)
        qh = self.q1 + self.q2
        x2 = x1 + self.L2_m * np.cos(qh)
        y2 = y1 + self.L2_m * np.sin(qh)
        return (x0, y0), (x1, y1), (x2, y2)

    def _sample_target(self):
        """DMControl-like target sampling: angle ~ U[0,2π], radius ~ U[0.05, 0.20] m."""
        angle = self.np_random.uniform(0, 2 * np.pi)
        r = float(self.np_random.uniform(0.05, 0.20))
        # Ensure reachable with a small margin
        r = min(r, self.reach_m - (self.finger_r_m + 0.005))
        self.tx_m = r * np.cos(angle)
        self.ty_m = r * np.sin(angle)

    # ------------------ Observations / Reward ------------------

    def _obs(self):
        (_, _), (_, _), (fx, fy) = self._fk_m()
        to_target_x = self.tx_m - fx
        to_target_y = self.ty_m - fy
        return np.array([self.q1, self.q2, to_target_x, to_target_y, self.w1, self.w2],
                dtype=np.float32)

    def _distance_m(self):
        (_, _), (_, _), (fx, fy) = self._fk_m()
        return float(np.hypot(self.tx_m - fx, self.ty_m - fy))

    def _reward(self):
        d = self._distance_m()
        return float(d <= self.target_r_m)

    # ------------------ Gym API ------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Randomize joints (like limited & rotational joints)
        self.q1 = float(self.np_random.uniform(-np.pi, np.pi))
        self.q2 = float(self.np_random.uniform(self.q2_min, self.q2_max))
        self.w1 = 0.0
        self.w2 = 0.0
        self._sample_target()
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        w = np.array([self.w1, self.w2], dtype=np.float32)
        tau = self.torque_scale * a
        drag = self.d_lin * w + self.d_quad * np.abs(w) * w
        acc = tau - drag

        w = np.clip(w + self.dt * acc, -self.max_w, self.max_w)
        self.w1, self.w2 = float(w[0]), float(w[1])
        self.q1 += self.dt * self.w1
        self.q2 += self.dt * self.w2

        # wrist joint limits (reflective clamp)
        if self.q2 > self.q2_max: self.q2 = self.q2_max; self.w2 = 0.0
        if self.q2 < self.q2_min: self.q2 = self.q2_min; self.w2 = 0.0

        reward = self._reward()
        self.steps += 1
        terminated = False
        truncated = self.steps >= self.max_episode_steps
        info = {"distance_m": self._distance_m()}
        return self._obs(), reward, terminated, truncated, info

    # ------------------ Rendering ------------------

    def _m_to_px(self, x_m, y_m):
        """Convert meters (world y-up) to screen pixels (y-down), centered in arena."""
        x_px = self.cx + x_m * self.px_per_m
        y_px = self.cy - y_m * self.px_per_m
        return int(x_px), int(y_px)

    def _make_diagonal_gradient(self, size, c0, c1, c2):
        import pygame, numpy as np
        w, h = size

        X = np.linspace(0.0, 1.0, w, dtype=np.float32)
        Y = np.linspace(0.0, 1.0, h, dtype=np.float32)
        t = (X[None, :] + Y[:, None]) * 0.5

        t1 = np.clip(t * 2.0, 0.0, 1.0)
        t2 = np.clip((t - 0.5) * 2.0, 0.0, 1.0)

        c0 = np.asarray(c0, np.float32)
        c1 = np.asarray(c1, np.float32)
        c2 = np.asarray(c2, np.float32)

        rgb = (1.0 - t1)[..., None] * c0 + t1[..., None] * c1
        rgb = (1.0 - t2)[..., None] * rgb + t2[..., None] * c2
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)           # (h, w, 3)

        surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))  # (w, h, 3)

        if getattr(self, "_surface", None) is not None:
            surf = surf.convert(self._surface)
        return surf

    def render(self):
        if self._surface is None:
            pygame.init()
            self._surface = pygame.Surface((self.width, self.height))
            self.clock = pygame.time.Clock()
            self._trail = []
            self._grad = None

        surf = self._surface

        # build gradient AFTER self._surface exists
        if self._grad is None:
            self._grad = self._make_diagonal_gradient(
                (self.width, self.height),
                C_TEAL, C_PURPLE, C_PINK
            )
        surf.blit(self._grad, (0, 0))

        # --- Arena panel ---
        panel = pygame.Surface((self.area_size, self.area_size), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 0))
        # Subtle dark fill for depth
        pygame.draw.rect(panel, (0, 0, 0, 30), panel.get_rect(), border_radius=14)
        surf.blit(panel, (self.area_x, self.area_y))

        # --- Double border ---
        pygame.draw.rect(
            surf, (60, 220, 255),
            pygame.Rect(self.area_x, self.area_y, self.area_size, self.area_size),
            width=2, border_radius=14
        )
        pygame.draw.rect(
            surf, (255, 70, 200),
            pygame.Rect(self.area_x+3, self.area_y+3, self.area_size-6, self.area_size-6),
            width=1, border_radius=12
        )

        # FK to pixels
        (x0m,y0m),(x1m,y1m),(x2m,y2m) = self._fk_m()
        x0,y0 = self._m_to_px(x0m,y0m)
        x1,y1 = self._m_to_px(x1m,y1m)
        x2,y2 = self._m_to_px(x2m,y2m)
        tx,ty = self._m_to_px(self.tx_m, self.ty_m)

        # --- Target: solid with thin outline ---
        trg_r_px = max(2, int(self.target_r_m * self.px_per_m))
        pygame.draw.circle(surf, (210, 70, 60), (tx, ty), trg_r_px)
        pygame.draw.circle(surf, (220, 80, 70), (tx, ty), trg_r_px, width=3)

        # --- Arm ---
        pygame.draw.line(self._surface, (180, 180, 180), (x0, y0), (x1, y1), self.link_thickness_px)
        pygame.draw.line(self._surface, (255, 140, 0),   (x1, y1), (x2, y2), self.link_thickness_px)
        pygame.draw.circle(self._surface, (160, 160, 160), (x0, y0), self.joint_r_px)
        pygame.draw.circle(self._surface, (255, 200, 120), (x1, y1), self.joint_r_px)

        # --- Fingertip + short trail ---
        f_r_px = max(2, int(self.finger_r_m * self.px_per_m))
        self._trail.append((x2, y2))
        if len(self._trail) > 10: self._trail.pop(0)
        for j, (px, py) in enumerate(self._trail):
            a = 12 + 12*j  # fade-in
            s = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
            pygame.draw.circle(s, (220, 220, 220, a), (px, py), f_r_px)
            surf.blit(s, (0, 0), special_flags=pygame.BLEND_PREMULTIPLIED)
        pygame.draw.circle(surf, C_WHITE, (x2, y2), f_r_px)

        return np.transpose(np.array(pygame.surfarray.pixels3d(surf)), (1, 0, 2)).copy()

    def close(self):
        if self._surface is not None:
            pygame.quit()
            self._surface = None


class ReacherEasyEnv(ReacherEnv):
    """Large target."""

    def __init__(self):
        super().__init__()


class ReacherHardEnv(ReacherEnv):
    """Small target."""

    def __init__(self):
        super().__init__(target_size_m=0.025)
