import gymnasium as gym


class Render(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.cfg = cfg
		self.env = env
		self._size = cfg.render_size

	def _get_frame(self):
		frame = self.env.render(width=self._size, height=self._size).copy()
		if frame.shape[-1] == 3:
			frame = frame.transpose(2, 0, 1)
		return frame

	def reset(self):
		obs, info = self.env.reset()
		info['frame'] = self._get_frame()
		return obs, info

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action)
		info['frame'] = self._get_frame()
		return obs, reward, terminated, truncated, info

	def close(self):
		self.env.close()

	def render(self, *args, **kwargs):
		kwargs['height'] = kwargs.get('height', self._size)
		kwargs['width'] = kwargs.get('width', self._size)
		return self.env.render(*args, **kwargs)
