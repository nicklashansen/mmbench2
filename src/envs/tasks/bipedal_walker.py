import numpy as np
import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
from gymnasium.envs.box2d import bipedal_walker


bipedal_walker.VIEWPORT_W = 224
bipedal_walker.VIEWPORT_H = 224
bipedal_walker.SCALE = 15


class CustomBipedalWalker(bipedal_walker.BipedalWalker):
    """
    Custom bipedal walker environment with modifications to the rendering.
    Environments with different terrains extend this class.
    """

    def __init__(self, render_mode: str | None = None):
        super().__init__(render_mode=render_mode, hardcore=False)

    def _build_terrain(self):
        raise NotImplementedError("This method should be overridden in subclasses.")

    def _generate_terrain(self, hardcore):
        assert hardcore is False, 'Unexpectedly received hardcore=True'
        self._build_terrain()
        self._create_terrain_polys()
    
    def _create_terrain_polys(self):
        self.terrain_poly = []
        possible_ground_colors = np.array(
            [
                [102, 153, 76],
                [94, 163, 60],
                [80, 120, 58],
                [119, 125, 58],
                [181, 149, 101],
                [194, 160, 134],
                [181, 130, 91],
                [122, 176, 114],
                [128, 191, 128],
                [143, 194, 143],
                [155, 166, 123],
                [125, 87, 57],
                [133, 111, 94],
                [145, 120, 120],
                [184, 165, 128],
                [200, 180, 150],
                [96, 158, 102],
            ]
        )
        starting_ground_color = possible_ground_colors[np.random.randint(len(possible_ground_colors))]
        ground_color = starting_ground_color.copy()
        for i in range(bipedal_walker.TERRAIN_LENGTH - 1):
            poly = [
                (self.terrain_x[i], self.terrain_y[i]),
                (self.terrain_x[i + 1], self.terrain_y[i + 1]),
            ]
            self.fd_edge.shape.vertices = poly
            # ensure each color is within +- 15 of the starting color
            ground_color = np.clip(ground_color + np.random.randint(-1, 1, size=3),
                                   starting_ground_color - 15, starting_ground_color + 15)
            grass_color = (ground_color[0]-20, np.random.randint(ground_color[1], 255), ground_color[2])
            t = self.world.CreateStaticBody(fixtures=self.fd_edge)
            t.color1 = grass_color
            t.color2 = ground_color
            self.terrain.append(t)
            poly += [(poly[1][0], 0), (poly[0][0], 0)]
            self.terrain_poly.append((poly, ground_color.tolist()))
        self.terrain.reverse()

    def reset(self, *, seed=None, options=None):
        sky_rand = np.random.randint(-7, 7, size=3) + np.random.randint(-7, 7, size=1)
        cloud_rand = np.random.randint(-4, 4, size=3) + np.random.randint(-4, 4, size=1) + \
            (0.5 * sky_rand).astype(int)
        self.sky_color = np.clip(np.array([215, 215, 252]) + sky_rand, 0, 255)
        self.cloud_color = np.clip(np.array([247, 247, 247]) + cloud_rand, 0, 255)
        return super().reset(seed=seed, options=options)

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[box2d]"`'
            ) from e

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((bipedal_walker.VIEWPORT_W, bipedal_walker.VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface(
            (bipedal_walker.VIEWPORT_W + max(0.0, self.scroll) * bipedal_walker.SCALE, bipedal_walker.VIEWPORT_H)
        )

        pygame.transform.scale(self.surf, (bipedal_walker.SCALE, bipedal_walker.SCALE))

        pygame.draw.polygon(
            self.surf,
            color=self.sky_color.tolist(),
            points=[
                (self.scroll * bipedal_walker.SCALE, 0),
                (self.scroll * bipedal_walker.SCALE + bipedal_walker.VIEWPORT_W, 0),
                (self.scroll * bipedal_walker.SCALE + bipedal_walker.VIEWPORT_W, bipedal_walker.VIEWPORT_H),
                (self.scroll * bipedal_walker.SCALE, bipedal_walker.VIEWPORT_H),
            ],
        )

        for poly, x1, x2 in self.cloud_poly:
            if x2 < self.scroll / 2:
                continue
            if x1 > self.scroll / 2 + bipedal_walker.VIEWPORT_W / bipedal_walker.SCALE:
                continue
            pygame.draw.polygon(
                self.surf,
                color=self.cloud_color.tolist(),
                points=[
                    (p[0] * bipedal_walker.SCALE + self.scroll * bipedal_walker.SCALE / 2, p[1] * bipedal_walker.SCALE) for p in poly
                ],
            )
            gfxdraw.aapolygon(
                self.surf,
                [(p[0] * bipedal_walker.SCALE + self.scroll * bipedal_walker.SCALE / 2, p[1] * bipedal_walker.SCALE) for p in poly],
                (255, 255, 255),
            )
        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll:
                continue
            if poly[0][0] > self.scroll + bipedal_walker.VIEWPORT_W / bipedal_walker.SCALE:
                continue
            scaled_poly = []
            for coord in poly:
                scaled_poly.append([coord[0] * bipedal_walker.SCALE, coord[1] * bipedal_walker.SCALE])
            pygame.draw.polygon(self.surf, color=color, points=scaled_poly)
            gfxdraw.aapolygon(self.surf, scaled_poly, color)

        self.lidar_render = (self.lidar_render + 1) % 100
        i = self.lidar_render
        if i < 2 * len(self.lidar):
            single_lidar = (
                self.lidar[i]
                if i < len(self.lidar)
                else self.lidar[len(self.lidar) - i - 1]
            )
            if hasattr(single_lidar, "p1") and hasattr(single_lidar, "p2"):
                pygame.draw.line(
                    self.surf,
                    color=(255, 0, 0),
                    start_pos=(single_lidar.p1[0] * bipedal_walker.SCALE, single_lidar.p1[1] * bipedal_walker.SCALE),
                    end_pos=(single_lidar.p2[0] * bipedal_walker.SCALE, single_lidar.p2[1] * bipedal_walker.SCALE),
                    width=1,
                )

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is bipedal_walker.circleShape:
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color1,
                        center=trans * f.shape.pos * bipedal_walker.SCALE,
                        radius=f.shape.radius * bipedal_walker.SCALE,
                    )
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color2,
                        center=trans * f.shape.pos * bipedal_walker.SCALE,
                        radius=f.shape.radius * bipedal_walker.SCALE,
                    )
                else:
                    path = [trans * v * bipedal_walker.SCALE for v in f.shape.vertices]
                    if len(path) > 2:
                        pygame.draw.polygon(self.surf, color=obj.color1, points=path)
                        gfxdraw.aapolygon(self.surf, path, obj.color1)
                        path.append(path[0])
                        pygame.draw.polygon(
                            self.surf, color=obj.color2, points=path, width=1
                        )
                        gfxdraw.aapolygon(self.surf, path, obj.color2)
                    else:
                        pygame.draw.aaline(
                            self.surf,
                            start_pos=path[0],
                            end_pos=path[1],
                            color=obj.color1,
                        )

        self.surf = pygame.transform.flip(self.surf, False, True)

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (-self.scroll * bipedal_walker.SCALE, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )[:, -bipedal_walker.VIEWPORT_W:]


class BipedalWalkerFlat(CustomBipedalWalker):
    """
    Bipedal walker environment with flat terrain.
    """

    def __init__(self, render_mode: str | None = None):
        super().__init__(render_mode=render_mode)

    def _build_terrain(self):
        self.terrain_x = np.arange(0, bipedal_walker.TERRAIN_LENGTH * bipedal_walker.TERRAIN_STEP, bipedal_walker.TERRAIN_STEP)
        self.terrain_y = np.full_like(self.terrain_x, bipedal_walker.TERRAIN_HEIGHT)


class BipedalWalkerUneven(CustomBipedalWalker):
    """
    Bipedal walker environment with uneven terrain.
    """

    def __init__(self, render_mode: str | None = None):
        super().__init__(render_mode=render_mode)

    def _build_terrain(self):
        velocity = 0.
        y = bipedal_walker.TERRAIN_HEIGHT
        self.terrain_x = []
        self.terrain_y = []

        for i in range(bipedal_walker.TERRAIN_LENGTH):
            x = i * bipedal_walker.TERRAIN_STEP
            velocity = 0.8 * velocity + 0.011 * np.sign(bipedal_walker.TERRAIN_HEIGHT - y)
            if i > bipedal_walker.TERRAIN_STARTPAD:
                velocity += self.np_random.uniform(-1, 1) / bipedal_walker.SCALE
            y += velocity
            y = max(1, min(y, 5.5))
            self.terrain_x.append(x)
            self.terrain_y.append(y)


class BipedalWalkerRugged(CustomBipedalWalker):
    """
    Bipedal walker environment with rugged terrain.
    """

    def __init__(self, render_mode: str | None = None):
        super().__init__(render_mode=render_mode)

    def _build_terrain(self):
        velocity = 0.
        y = bipedal_walker.TERRAIN_HEIGHT
        self.terrain_x = []
        self.terrain_y = []

        for i in range(bipedal_walker.TERRAIN_LENGTH):
            x = i * bipedal_walker.TERRAIN_STEP
            velocity = 0.75 * velocity + 0.015 * np.sign(bipedal_walker.TERRAIN_HEIGHT - y)
            if i > bipedal_walker.TERRAIN_STARTPAD:
                velocity += 2.25 * self.np_random.uniform(-1, 1) / bipedal_walker.SCALE
            y += velocity
            y = max(1, min(y, 5.5))
            self.terrain_x.append(x)
            self.terrain_y.append(y)


class BipedalWalkerHills(CustomBipedalWalker):
    """
    Bipedal walker environment with smooth hills.
    """

    def __init__(self, render_mode: str | None = None):
        super().__init__(render_mode=render_mode)

    def _build_terrain(self, amplitude_range=(0.4, 0.8), wavelength_range=(7, 14)):
        start_y = bipedal_walker.TERRAIN_HEIGHT + np.random.uniform(-1, -0.8)
        amp = np.random.uniform(*amplitude_range)
        wavelength = np.random.uniform(*wavelength_range)
        self.terrain_x = []
        self.terrain_y = []

        for i in range(bipedal_walker.TERRAIN_LENGTH):
            x = i * bipedal_walker.TERRAIN_STEP
            y = start_y + amp * np.sin(2 * np.pi * x * (1/wavelength))
            y = max(1, min(y, 5.5))
            self.terrain_x.append(x)
            self.terrain_y.append(y)


class BipedalWalkerObstacles(CustomBipedalWalker):
    """
    Bipedal walker environment with various obstacles.
    """

    def __init__(self, render_mode: str | None = None):
        super().__init__(render_mode=render_mode)

    def _build_terrain(self):
        GRASS, STUMP, PIT, _STATES_ = range(4)
        state = GRASS
        velocity = 0.0
        y = bipedal_walker.TERRAIN_HEIGHT
        counter = bipedal_walker.TERRAIN_STARTPAD
        oneshot = False
        self.terrain = []
        self.terrain_x = []
        self.terrain_y = []

        original_y = 0
        for i in range(bipedal_walker.TERRAIN_LENGTH):
            x = i * bipedal_walker.TERRAIN_STEP
            self.terrain_x.append(x)

            if state == GRASS and not oneshot:
                velocity = 0.8 * velocity + 0.01 * np.sign(bipedal_walker.TERRAIN_HEIGHT - y)
                if i > bipedal_walker.TERRAIN_STARTPAD:
                    velocity += self.np_random.uniform(-1, 1) / bipedal_walker.SCALE
                y += velocity
                y = max(1.5, min(y, 5.5))

            elif state == PIT and oneshot:
                counter = self.np_random.integers(2, 4)
                poly = [
                    (x, y),
                    (x + bipedal_walker.TERRAIN_STEP, y),
                    (x + bipedal_walker.TERRAIN_STEP, y - 3 * bipedal_walker.TERRAIN_STEP),
                    (x, y - 3 * bipedal_walker.TERRAIN_STEP),
                ]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                self.terrain.append(t)

                self.fd_polygon.shape.vertices = [
                    (p[0] + bipedal_walker.TERRAIN_STEP * counter, p[1]) for p in poly
                ]
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                self.terrain.append(t)
                counter += 2
                original_y = y

            elif state == PIT and not oneshot:
                y = original_y
                if counter > 1:
                    y -= 3 * bipedal_walker.TERRAIN_STEP

            elif state == STUMP and oneshot:
                counter = self.np_random.integers(2, 4)
                poly = [
                    (x, y),
                    (x + counter * bipedal_walker.TERRAIN_STEP, y),
                    (x + counter * bipedal_walker.TERRAIN_STEP, y + counter * 0.75 * bipedal_walker.TERRAIN_STEP),
                    (x, y + counter * 0.75 * bipedal_walker.TERRAIN_STEP),
                ]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                self.terrain.append(t)

            oneshot = False
            self.terrain_y.append(y)
            counter -= 1
            if counter == 0:
                counter = self.np_random.integers(bipedal_walker.TERRAIN_GRASS / 2, bipedal_walker.TERRAIN_GRASS)
                if state == GRASS:
                    state = self.np_random.integers(1, _STATES_)
                    oneshot = True
                else:
                    state = GRASS
                    oneshot = True
