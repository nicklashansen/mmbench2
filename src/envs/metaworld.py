import numpy as np
import gymnasium as gym
import torch
from torchvision.transforms import functional as F
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

from envs.wrappers.timeout import Timeout
from envs.wrappers.pixels import Pixels


class MetaWorldWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		self.env.camera_name = "corner2"
		self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
		self.env._freeze_rand_vec = False

	def _extract_info(self, info):
		info = {
			'terminated': info.get('terminated', False),
			'truncated': info.get('truncated', False),
			'success': float(info.get('success', 0.)),
		}
		info['score'] = info['success']
		return info

	def reset(self, **kwargs):
		super().reset(**kwargs)
		obs, _, _, _, info = self.env.step(np.zeros(self.env.action_space.shape, dtype=np.float32))
		obs = obs.astype(np.float32)
		return obs, self._extract_info(info)

	def step(self, action):
		reward = 0
		for _ in range(2):
			obs, r, terminated, truncated, info = self.env.step(action.copy())
			reward += r
			if terminated or truncated:
				break
		obs = obs.astype(np.float32)
		info['terminated'] = terminated
		info['truncated'] = truncated
		return obs, reward, terminated, truncated, self._extract_info(info)

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def render(self, *args, **kwargs):
		h, w = kwargs.get('height', 224), kwargs.get('width', 224)
		frame = torch.from_numpy(self.env.render().copy()).permute(2, 0, 1)
		frame = frame.flip(1)
		frame = F.resize(frame, (h, w)).permute(1, 2, 0).numpy()
		return frame

	def close(self):
		self.env.close()


def make_env(cfg):
	"""
	Make Meta-World environment.
	"""
	env_id = cfg.task.split("-", 1)[-1] + "-v2-goal-observable"
	if not cfg.task.startswith('mw-') or env_id not in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE:
		raise ValueError('Unknown task:', cfg.task)
	env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_id](
		seed=cfg.seed,
		render_mode='rgb_array')
	env = MetaWorldWrapper(env, cfg)
	if cfg.obs == 'rgb':
		env = Pixels(env, cfg)
	env = Timeout(env, max_episode_steps=100)
	return env
