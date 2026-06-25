import gymnasium as gym
import numpy as np
import pygame


class ChaseEvadeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, max_episode_steps=500):
        super().__init__()
        self.max_episode_steps = max_episode_steps

        # Observation: [agent(4), opp(4), role(1), obstacles(6)]
        self.obs_dim = 15
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-np.ones(2, dtype=np.float32), high=np.ones(2, dtype=np.float32)
        )

        # Screen / arena
        self.width, self.height = 224, 224
        self.radius = 90
        self.center = (self.width // 2, self.height // 2 + 6)

        # Entities / gameplay
        self.char_r = 8
        self.obstacle_r = 12
        self.speed = 3.5
        self.evader_speed_scale = 0.95
        self.evader_offset = 15
        self.character_margin = 4  # keep a few px from visual edge

        # State
        self.agent_x = self.agent_y = 0.0
        self.opp_x = self.opp_y = 0.0
        self.agent_vx = self.agent_vy = 0.0
        self.opp_vx = self.opp_vy = 0.0
        self.is_agent_evader = True

        # Role-change
        self.role_change_counter = 0
        self.role_change_min_frames = 5
        self.role_change_active = False
        self.pursuer_is_agent = False

        # Obstacles
        self.num_obstacles = 3
        self.obstacle_r = 12
        self.obstacle_orbit_radius = int(self.radius * 0.45)
        self.rotation_speed = 0.01  # radians per frame
        self.base_angles = []
        self.direction = 1

        # FX
        self.flash_timer = 0
        self.flash_x = self.flash_y = 0.0
        self.flash_color = (255, 220, 50)

        # Rendering
        self._surface = None
        self.clock = None
        self.font = None
        self.font_big = None

        self.steps = 0

    # -------------------- Utils --------------------

    def _dist(self, x1, y1, x2, y2):
        return np.hypot(x1 - x2, y1 - y2)

    def _clamp_to_circle(self, x, y, r, margin=0.0):
        dx, dy = x - self.center[0], y - self.center[1]
        d = np.hypot(dx, dy)
        max_r = self.radius - r - margin
        if d > max_r:
            scale = max_r / (d + 1e-8)
            x = self.center[0] + dx * scale
            y = self.center[1] + dy * scale
        return x, y

    def _circle_collision(self, x1, y1, r1, x2, y2, r2):
        dx, dy = x2 - x1, y2 - y1
        dist = np.hypot(dx, dy)
        overlap = r1 + r2 - dist
        if overlap > 0:
            nx, ny = dx / (dist + 1e-8), dy / (dist + 1e-8)
            return True, nx, ny, overlap
        return False, 0, 0, 0

    def _resolve_obstacle_collision(self, x, y, vx, vy, r):
        for ox, oy in self.obstacles:
            hit, nx, ny, overlap = self._circle_collision(x, y, r, ox, oy, self.obstacle_r)
            if hit:
                x -= nx * (overlap + 0.5)
                y -= ny * (overlap + 0.5)
                vn = vx * nx + vy * ny
                if vn > 0:
                    vx -= 2 * vn * nx
                    vy -= 2 * vn * ny
        return x, y, vx, vy
    
    def _opp_pursue(self, target_x, target_y, opp_speed):
        """Opponent pursues a target with obstacle avoidance."""
        dx, dy = target_x - self.opp_x, target_y - self.opp_y
        dist = np.hypot(dx, dy)
        if dist > 1e-5:
            ux, uy = dx / dist, dy / dist
        else:
            ux, uy = 0.0, 0.0

        # Obstacle avoidance steering
        avoid_x, avoid_y = 0.0, 0.0
        for ox, oy in self.obstacles:
            d = np.hypot(self.opp_x - ox, self.opp_y - oy)
            if d < self.obstacle_r * 3:
                rep = max(0, (self.obstacle_r * 3 - d) / (self.obstacle_r * 3))
                avoid_x += (self.opp_x - ox) / (d + 1e-8) * rep
                avoid_y += (self.opp_y - oy) / (d + 1e-8) * rep

        ux += avoid_x
        uy += avoid_y

        # Normalize + scale
        norm = np.hypot(ux, uy)
        if norm > 1e-8:
            ux, uy = ux / norm, uy / norm
        self.opp_vx, self.opp_vy = ux * opp_speed, uy * opp_speed

    def _opp_evade(self, opp_speed):
        """Opponent evades the agent using boundary avoidance, cover, and zig-zag."""
        dx, dy = self.opp_x - self.agent_x, self.opp_y - self.agent_y
        dist = np.hypot(dx, dy)
        if dist > 1e-5:
            ux, uy = dx / dist, dy / dist
        else:
            ux, uy = 0.0, 0.0

        # Boundary avoidance (don’t get pinned against arena edge)
        cx, cy = self.center
        from_center_x, from_center_y = self.opp_x - cx, self.opp_y - cy
        d_center = np.hypot(from_center_x, from_center_y)
        if d_center > self.radius * 0.8:
            ux -= from_center_x / (d_center + 1e-8) * 0.5
            uy -= from_center_y / (d_center + 1e-8) * 0.5

        # Obstacle cover (bias toward obstacle between pursuer and evader)
        for ox, oy in self.obstacles:
            v1 = np.array([self.agent_x - ox, self.agent_y - oy])
            v2 = np.array([self.opp_x - ox, self.opp_y - oy])
            if np.dot(v1, v2) < 0:  # obstacle between them
                ux += (ox - self.opp_x) * 0.02
                uy += (oy - self.opp_y) * 0.02

        # Zig-zag when close
        if dist < self.radius * 0.4:
            jitter_dir = np.array([-dy, dx])
            jitter_dir = jitter_dir / (np.linalg.norm(jitter_dir) + 1e-8)
            jitter_strength = 0.2
            ux += jitter_dir[0] * self.np_random.choice([-1, 1]) * jitter_strength
            uy += jitter_dir[1] * self.np_random.choice([-1, 1]) * jitter_strength

        # Normalize + scale
        norm = np.hypot(ux, uy)
        if norm > 1e-8:
            ux, uy = ux / norm, uy / norm
        self.opp_vx, self.opp_vy = ux * opp_speed, uy * opp_speed

    # -------------------- Gym API --------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        ang = self.np_random.uniform(0, 2 * np.pi)
        self.agent_x = self.center[0] + (self.radius - self.evader_offset) * np.cos(ang)
        self.agent_y = self.center[1] + (self.radius - self.evader_offset) * np.sin(ang)
        self.opp_x, self.opp_y = self.center
        self.agent_vx = self.agent_vy = 0.0
        self.opp_vx = self.opp_vy = 0.0

        # Random starting role
        self.is_agent_evader = bool(self.np_random.integers(0, 2))

        self.role_change_counter = 0
        self.role_change_active = False
        self.flash_timer = 0
        self.steps = 0

        # Randomize global phase and direction
        global_phase = self.np_random.uniform(0, 2 * np.pi)
        self.direction = self.np_random.choice([-1, 1])
        self.base_angles = [
            global_phase + 2 * np.pi * i / self.num_obstacles
            for i in range(self.num_obstacles)
        ]

        # Initialize obstacle positions
        self.obstacles = []
        for ang in self.base_angles:
            ox = self.center[0] + self.obstacle_orbit_radius * np.cos(ang)
            oy = self.center[1] + self.obstacle_orbit_radius * np.sin(ang)
            self.obstacles.append((ox, oy))

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        reward = 0.0
        terminated = False
        truncated = False
        
        # Update obstacle positions (rotation)
        self.obstacles = []
        for base_ang in self.base_angles:
            ang = base_ang + self.direction * self.rotation_speed * self.steps
            ox = self.center[0] + self.obstacle_orbit_radius * np.cos(ang)
            oy = self.center[1] + self.obstacle_orbit_radius * np.sin(ang)
            self.obstacles.append((ox, oy))

        if self.role_change_active:
            self.role_change_counter += 1
            pursuer_speed = self.speed
            evader_speed = self.speed * self.evader_speed_scale

            # Pursuer -> center
            if self.pursuer_is_agent:
                dx, dy = self.center[0] - self.agent_x, self.center[1] - self.agent_y
                dist = np.hypot(dx, dy)
                if dist > 1.5:
                    ux, uy = dx / dist, dy / dist
                    self.agent_x += ux * pursuer_speed
                    self.agent_y += uy * pursuer_speed
            else:
                dx, dy = self.center[0] - self.opp_x, self.center[1] - self.opp_y
                dist = np.hypot(dx, dy)
                if dist > 1.5:
                    ux, uy = dx / dist, dy / dist
                    self.opp_x += ux * pursuer_speed
                    self.opp_y += uy * pursuer_speed

            # Evader -> outward (clamp to safe border offset)
            if self.is_agent_evader:
                dx, dy = self.agent_x - self.center[0], self.agent_y - self.center[1]
                dist = np.hypot(dx, dy)
                if dist > 1e-2:
                    ux, uy = dx / dist, dy / dist
                    self.agent_x += ux * evader_speed
                    self.agent_y += uy * evader_speed
                    self.agent_x, self.agent_y = self._clamp_to_circle(
                        self.agent_x, self.agent_y, self.char_r, margin=self.evader_offset
                    )
            else:
                dx, dy = self.opp_x - self.center[0], self.opp_y - self.center[1]
                dist = np.hypot(dx, dy)
                if dist > 1e-2:
                    ux, uy = dx / dist, dy / dist
                    self.opp_x += ux * evader_speed
                    self.opp_y += uy * evader_speed
                    self.opp_x, self.opp_y = self._clamp_to_circle(
                        self.opp_x, self.opp_y, self.char_r, margin=self.evader_offset
                    )

            # End condition: min frames AND pursuer at center
            if self.role_change_counter >= self.role_change_min_frames:
                if self.pursuer_is_agent:
                    if self._dist(self.agent_x, self.agent_y, *self.center) < 2.0:
                        self.role_change_active = False
                else:
                    if self._dist(self.opp_x, self.opp_y, *self.center) < 2.0:
                        self.role_change_active = False

        else:
            # Speed scaling
            agent_speed = self.speed * (self.evader_speed_scale if self.is_agent_evader else 1.0)
            opp_speed   = self.speed * (self.evader_speed_scale if not self.is_agent_evader else 1.0)

            # Agent move (keep off visual edge with margin)
            self.agent_vx = float(action[0]) * agent_speed
            self.agent_vy = float(action[1]) * agent_speed
            self.agent_x += self.agent_vx
            self.agent_y += self.agent_vy
            self.agent_x, self.agent_y = self._clamp_to_circle(
                self.agent_x, self.agent_y, self.char_r, margin=self.character_margin
            )
            self.agent_x, self.agent_y, self.agent_vx, self.agent_vy = self._resolve_obstacle_collision(
                self.agent_x, self.agent_y, self.agent_vx, self.agent_vy, self.char_r
            )

            # ---------------- Opponent AI ----------------
            if self.is_agent_evader:
                self._opp_pursue(self.agent_x, self.agent_y, opp_speed)
            else:
                self._opp_evade(opp_speed)

            # Apply movement + collision for opponent
            self.opp_x += self.opp_vx
            self.opp_y += self.opp_vy
            self.opp_x, self.opp_y = self._clamp_to_circle(
                self.opp_x, self.opp_y, self.char_r, margin=self.character_margin
            )
            self.opp_x, self.opp_y, self.opp_vx, self.opp_vy = self._resolve_obstacle_collision(
                self.opp_x, self.opp_y, self.opp_vx, self.opp_vy, self.char_r
            )

            # Catch check
            d = self._dist(self.agent_x, self.agent_y, self.opp_x, self.opp_y)
            if d < 2 * self.char_r:
                # store previous evader before switching
                self.prev_evader_is_agent = self.is_agent_evader
                caught_agent = self.is_agent_evader
                self.is_agent_evader = not self.is_agent_evader
                self.role_change_active = True
                self.role_change_counter = 0
                self.pursuer_is_agent = not self.is_agent_evader
                # flash effect (expand+collapse, longer for visibility)
                self.flash_x, self.flash_y = (self.agent_x + self.opp_x) / 2, (self.agent_y + self.opp_y) / 2
                self.flash_color = (255, 140, 0) if caught_agent else (70, 160, 70)
                self.flash_timer = 12

        # Reward
        d = self._dist(self.agent_x, self.agent_y, self.opp_x, self.opp_y)
        reward = ((d / self.radius) if self.is_agent_evader else -(d / self.radius)) * 0.1

        # Time
        self.steps += 1
        if self.steps >= self.max_episode_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        obs = [
            self.agent_x / self.width,
            self.agent_y / self.height,
            self.agent_vx / self.speed,
            self.agent_vy / self.speed,
            self.opp_x / self.width,
            self.opp_y / self.height,
            self.opp_vx / self.speed,
            self.opp_vy / self.speed,
            float(self.is_agent_evader),
        ]
        # Add obstacle positions (normalized)
        for ox, oy in self.obstacles:
            obs.append(ox / self.width)
            obs.append(oy / self.height)
        return np.array(obs, dtype=np.float32)

    # -------------------- Rendering helpers --------------------

    def _draw_sky(self, surf):
        """Sky gradient tinted by evader, with smooth transition on role change."""
        # Base gradient (blue)
        base_top = (130, 185, 180)
        base_bottom = (80, 170, 180)

        # Tint colors
        agent_tint = (255, 140, 0)
        opp_tint = (70, 160, 70)

        # Decide current tint
        if self.role_change_active and hasattr(self, "prev_evader_is_agent"):
            # Transitioning: interpolate old → new tint
            progress = min(1.0, self.role_change_counter / 10.0)  # ~10 frames fade
            old_tint = agent_tint if self.prev_evader_is_agent else opp_tint
            new_tint = agent_tint if self.is_agent_evader else opp_tint
            tint = [int(old*(1-progress) + new*progress) for old, new in zip(old_tint, new_tint)]
        else:
            # Constant tint
            tint = agent_tint if self.is_agent_evader else opp_tint

        # Draw gradient lines
        for i in range(self.height):
            t = i / self.height
            base = [int(base_top[j]*(1-t) + base_bottom[j]*t) for j in range(3)]
            color = [int(b*0.75 + tint[j]*0.25) for j, b in enumerate(base)]
            pygame.draw.line(surf, color, (0, i), (self.width, i))

    def _draw_arena_cylinder(self, surf):
        """Simple cylinder illusion: side rectangle + top disk + white outline."""
        cx, cy = self.center
        r = self.radius

        # Top disk fill (playable) then white outline
        pygame.draw.circle(surf, (40, 40, 45), (cx, cy), r)
        pygame.draw.circle(surf, (220, 220, 220), (cx, cy), r, 2)

    def _draw_obstacle_cylinder(self, surf, ox, oy, color=(130, 130, 130)):
        """Visual-only: small cylinder (top circle + short side). Hitbox stays circular at radius=self.obstacle_r."""
        r = self.obstacle_r
        # Side: short rectangle under the top disk (few pixels tall)
        body_h = 4
        pygame.draw.rect(surf, (90, 90, 95), pygame.Rect(int(ox - r), int(oy), 2 * r, body_h))
        # Top disk
        pygame.draw.circle(surf, color, (int(ox), int(oy)), r)

    def _draw_character(self, surf, x, y, team_color, step_count, vx, vy):
        cx, cy = int(x), int(y)
        body_col = (220, 220, 220)

        walk_phase = (step_count // 3) % 2
        offset = 1 if (abs(vx) + abs(vy)) > 0.5 else 0

        # Body
        pygame.draw.ellipse(surf, body_col, pygame.Rect(cx-6, cy-10, 12, 18))
        pygame.draw.rect(surf, team_color, pygame.Rect(cx-6, cy-3, 12, 6))

        # Feet (team-colored)
        pygame.draw.circle(surf, team_color, (cx-4, cy+8+(offset if walk_phase==0 else -offset)), 3)
        pygame.draw.circle(surf, team_color, (cx+4, cy+8+(offset if walk_phase==1 else -offset)), 3)

        # Arms angled down
        pygame.draw.ellipse(surf, body_col, pygame.Rect(cx-9, cy, 6, 4))
        pygame.draw.ellipse(surf, body_col, pygame.Rect(cx+3, cy, 6, 4))

        # Hat (close to head)
        pygame.draw.polygon(surf, team_color, [(cx, cy-14), (cx-6, cy-10), (cx+6, cy-10)])

    def _draw_flash(self, surf):
        # Expand then collapse
        if self.flash_timer <= 0:
            return
        half = 6
        if self.flash_timer > half:  # expanding
            progress = (12 - self.flash_timer) / half
        else:  # collapsing
            progress = self.flash_timer / half
        radius = int(10 + 40 * progress)
        alpha = int(255 * progress)
        flash_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.circle(flash_surface, (*self.flash_color, alpha),
                           (int(self.flash_x), int(self.flash_y)), radius, 3)
        surf.blit(flash_surface, (0, 0))
        self.flash_timer -= 1

    def _draw_countdown(self, surf):
        if not self.role_change_active:
            return
        remaining = max(0, 3 - (self.role_change_counter // 5))
        if remaining <= 0:
            return
        color = (255, 140, 0) if self.pursuer_is_agent else (70, 160, 70)

        # Outline effect (bigger font)
        txt = self.font_big.render(str(remaining), True, color)
        outline_col = (0, 0, 0)
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            t_o = self.font_big.render(str(remaining), True, outline_col)
            surf.blit(t_o, (self.center[0]-t_o.get_width()//2+dx,
                            self.center[1]-50+t_o.get_height()//2+dy))
        surf.blit(txt, (self.center[0]-txt.get_width()//2, self.center[1]-50+txt.get_height()//2))

    def _draw_status_text(self, surf):
        """Show 'orange/green is evading' in top-left with outline."""
        # Pick evader info
        if self.is_agent_evader:
            team_word, team_color = "orange", (255, 140, 0)
        else:
            team_word, team_color = "green", (0, 180, 0)

        # Split text: "<team_word> is evading"
        text_parts = [
            (team_word, team_color),
            (" is evading", (255, 255, 255)),
        ]

        x, y = 6, 3  # top-left corner
        for text, color in text_parts:
            txt_surface = self.font.render(text, True, color)

            # Outline effect
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                outline = self.font.render(text, True, (0, 0, 0))
                surf.blit(outline, (x + dx, y + dy))

            surf.blit(txt_surface, (x, y))
            x += txt_surface.get_width()  # move right for next piece

    # -------------------- Render --------------------

    def render(self):
        if self._surface is None:
            pygame.init()
            self._surface = pygame.Surface((self.width, self.height))
            self.clock = pygame.time.Clock()
        if self.font is None:
            self.font = pygame.font.SysFont("Arial", 18, bold=True)
            self.font_big = pygame.font.SysFont("Arial", 28, bold=True)

        self._surface.fill((0, 0, 0))

        # 1) Sky (behind everything)
        self._draw_sky(self._surface)

        # 2) Cylinder side (rectangle) + top disk + white outline (playable area)
        self._draw_arena_cylinder(self._surface)

        # 3) Flag at center (on the top disk)
        cx, cy = self.center
        pole_col = (180, 180, 180)
        flag_col = (255, 140, 0) if self.is_agent_evader else (70, 160, 70)
        pygame.draw.line(self._surface, pole_col, (cx, cy-12), (cx, cy+12), 2)
        pygame.draw.polygon(self._surface, flag_col, [(cx, cy-12), (cx+10, cy-8), (cx, cy-4)])

        # 4) Obstacles: simple cylinder-like visuals, same circular hitbox
        for ox, oy in self.obstacles:
            self._draw_obstacle_cylinder(self._surface, ox, oy)

        # 5) Status label
        self._draw_status_text(self._surface)

        # 6) Characters
        self._draw_character(self._surface, self.agent_x, self.agent_y,
                             (255, 140, 0), self.steps, self.agent_vx, self.agent_vy)
        self._draw_character(self._surface, self.opp_x, self.opp_y,
                             (70, 160, 70), self.steps, self.opp_vx, self.opp_vy)

        # 7) Flash + countdown
        self._draw_flash(self._surface)
        self._draw_countdown(self._surface)

        return np.transpose(np.array(pygame.surfarray.pixels3d(self._surface)), (1, 0, 2)).copy()

    def close(self):
        if self._surface is not None:
            pygame.quit()
            self._surface = None
