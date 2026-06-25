from collections import deque

import gymnasium as gym
import numpy as np


class Pixels(gym.Wrapper):
	def __init__(self, env, cfg, num_frames=1, size=None):
		super().__init__(env)
		self.cfg = cfg
		self.env = env
		if size is None:
			size = getattr(cfg, 'render_size', 128)
		self.observation_space = gym.spaces.Dict({
			'rgb': gym.spaces.Box(
				low=0, high=255, shape=(num_frames*3, size, size), dtype=np.uint8),
			'state': env.observation_space,
		})
		self._frames = deque([], maxlen=num_frames)
		self._size = size

	def _get_obs(self, is_reset=False):
		frame = self.env.render(width=self._size, height=self._size)
		if frame.shape[-1] == 3:
			frame = frame.transpose(2, 0, 1)
		num_frames = self._frames.maxlen if is_reset else 1
		for _ in range(num_frames):
			self._frames.append(frame)
		return np.concatenate(self._frames)

	def reset(self):
		state, info = self.env.reset()
		return {'state': state, 'rgb': self._get_obs(is_reset=True)}, info

	def step(self, action):
		state, reward, terminated, truncated, info = self.env.step(action)
		return {'state': state, 'rgb': self._get_obs()}, reward, terminated, truncated, info

	def close(self):
		self.env.close()

	def render(self, *args, **kwargs):
		kwargs['height'] = kwargs.get('height', self._size)
		kwargs['width'] = kwargs.get('width', self._size)
		return self.env.render(*args, **kwargs)
