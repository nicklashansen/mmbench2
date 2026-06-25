import numpy as np
import gymnasium as gym
import torch
from torchvision.transforms import functional as F

from envs.tasks.bipedal_walker import BipedalWalkerFlat, BipedalWalkerUneven, \
	BipedalWalkerRugged, BipedalWalkerHills, BipedalWalkerObstacles
from envs.tasks.lunarlander import LunarLander, LunarLanderHover, LunarLanderTakeoff
from envs.wrappers.timeout import Timeout


BOX2D_TASKS = {
	'bipedal-walker-flat': BipedalWalkerFlat,
	'bipedal-walker-uneven': BipedalWalkerUneven,
	'bipedal-walker-rugged': BipedalWalkerRugged,
	'bipedal-walker-hills': BipedalWalkerHills,
	'bipedal-walker-obstacles': BipedalWalkerObstacles,
	'lunarlander-land': LunarLander,
	'lunarlander-hover': LunarLanderHover,
	'lunarlander-takeoff': LunarLanderTakeoff,
}


class Box2DWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		if cfg.obs == 'rgb':
			self.observation_space = gym.spaces.Dict({
				'rgb': gym.spaces.Box(
					low=0, high=255, shape=(3, self.cfg.render_size, self.cfg.render_size), dtype=np.uint8),
				'state': env.observation_space,
			})
		self._cumulative_reward = 0

	def _extract_info(self, info):
		info = {
			'terminated': info.get('terminated', False),
			'truncated': info.get('truncated', False),
			'success': float(info.get('success', 0.)),
		}
		if self.cfg.task.startswith('lunarlander'):
			info['score'] = np.clip(self._cumulative_reward / 500, 0, 1)
		else:
			info['score'] = np.clip(self._cumulative_reward / 250, 0, 1)
		return info

	def get_observation(self, obs):
		if self.cfg.obs == 'rgb':
			return {'state': obs, 'rgb': self.render().transpose(2, 0, 1)}
		return obs

	def reset(self):
		obs, info = self.env.reset()
		self._cumulative_reward = 0
		return self.get_observation(obs), self._extract_info(info)

	def _safe_action(self, action):
		"""
		Ensure the action is within the valid range for the environment.
		"""
		assert isinstance(action, np.ndarray), 'Action must be a numpy array.'
		assert not np.isnan(action).any(), 'Action contains NaN values.'
		action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
		return action

	def step(self, action):
		reward = 0
		action = self._safe_action(action)
		for _ in range(2):
			obs, _reward, _, truncated, info = self.env.step(action.copy())
			if self.cfg.task == 'lunarlander-land':
				_reward = np.clip(_reward, -5, 10) / 20
			if self.cfg.task.startswith('bipedal-walker'):
				_reward = np.clip(_reward, -1, 10)
			reward += _reward
			if truncated:
				break
		terminated = False
		self._cumulative_reward += reward
		info['terminated'] = terminated
		info['truncated'] = truncated
		return self.get_observation(obs), reward, terminated, truncated, self._extract_info(info)

	@property
	def unwrapped(self):
		return self.env.unwrapped
	
	def render(self, **kwargs):
		frame = self.env.render()
		h, w = self.cfg.render_size, self.cfg.render_size
		if frame.shape[0] > h or frame.shape[1] > w:
			frame = torch.from_numpy(frame).permute(2, 0, 1)
			frame = F.resize(frame, (h, w))
			frame = frame.permute(1, 2, 0).numpy()
		return frame


def make_env(cfg):
	"""
	Make Box2D environment.
	"""
	if not cfg.task in BOX2D_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	env = BOX2D_TASKS[cfg.task](render_mode='rgb_array')
	env = Box2DWrapper(env, cfg)
	env = Timeout(env, max_episode_steps=250)
	return env
