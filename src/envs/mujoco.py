import numpy as np
import gymnasium as gym
import torch
from torchvision.transforms import functional as F

from envs.wrappers.timeout import Timeout


MUJOCO_TASKS = {
	'mujoco-ant': 'Ant-v4',
	'mujoco-inverted-pendulum': 'InvertedPendulum-v4',
	'mujoco-reacher': 'Reacher-v4',
	'mujoco-pusher': 'Pusher-v4',
	'mujoco-halfcheetah': 'HalfCheetah-v4',
	'mujoco-hopper': 'Hopper-v4',
	'mujoco-walker': 'Walker2d-v4',
}


class MuJoCoWrapper(gym.Wrapper):
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
		action_dim = env.action_space.shape[0]
		self.action_space = gym.spaces.Box(
			low=np.full(action_dim, -1),
			high=np.full(action_dim, +1),
			dtype=np.float32,
		)
		self.action_scale = self.env.action_space.high
		self.action_repeat = 1 if cfg.task in {
			'mujoco-reacher', 'mujoco-pusher', 'mujoco-halfcheetah'
		} else 2
		self._cumulative_reward = 0
		self._terminated = False

	def _extract_info(self, info):
		info = {
			'terminated': info.get('terminated', False),
			'truncated': info.get('truncated', False),
			'success': float(info.get('success', 0.)),
		}
		if self.cfg.task == 'mujoco-inverted-pendulum':
			# range is [0, 1000], normalize to [0, 1]
			info['score'] = np.clip(self._cumulative_reward, 0, 1000) / 1000
		elif self.cfg.task == 'mujoco-reacher':
			# range is [-50, 0], normalize to [0, 1]
			info['score'] = 1 + np.clip(self._cumulative_reward, -50, 0) / 50
		elif self.cfg.task == 'mujoco-pusher':
			# Pusher-v4 over 100 steps: dense penalty = -dist(obj,goal) - 0.1*dist(arm,obj) - 0.001*||a||^2.
			# Cumulative range roughly [-150, 0]; expert ~[-50, -20].
			info['score'] = 1 + np.clip(self._cumulative_reward, -150, 0) / 150
		elif self.cfg.task == 'mujoco-halfcheetah':
			# range is [0, 15000], normalize to [0, 1]
			info['score'] = np.clip(self._cumulative_reward, 0, 15000) / 15000
		elif self.cfg.task in {'mujoco-ant', 'mujoco-hopper', 'mujoco-walker'}:
			# range is [0, 5000], normalize to [0, 1]
			info['score'] = np.clip(self._cumulative_reward, 0, 5000) / 5000
		else:
			raise NotImplementedError(f'Score calculation for {self.cfg.task} not implemented.')
		return info

	def get_observation(self, obs):
		if self.cfg.obs == 'rgb':
			return {'state': obs, 'rgb': self.render().transpose(2, 0, 1)}
		return obs.astype(np.float32)

	def reset(self):
		obs, info = self.env.reset()
		self._cumulative_reward = 0
		self._terminated = False
		return self.get_observation(obs), self._extract_info(info)

	def step(self, action):
		action = action * self.action_scale
		reward = 0.
		for _ in range(self.action_repeat):
			obs, _reward, terminated, truncated, info = self.env.step(action.copy())
			if 'pendulum' in self.cfg.task and (terminated or self._terminated):
				self._terminated = True
				_reward = 0.
			elif 'hopper' in self.cfg.task or 'walker' in self.cfg.task:
				_reward = max(0, _reward) if self.env.unwrapped.is_healthy else -1
			reward += _reward
		self._cumulative_reward += reward
		info['terminated'] = False
		info['truncated'] = truncated
		return self.get_observation(obs), reward, False, truncated, self._extract_info(info)

	@property
	def unwrapped(self):
		return self.env.unwrapped
	
	def render(self, **kwargs):
		frame = self.env.render().copy()
		h, w = self.cfg.render_size, self.cfg.render_size
		if frame.shape[0] > h or frame.shape[1] > w:
			frame = torch.from_numpy(frame).permute(2, 0, 1)
			frame = F.resize(frame, (h, w))
			frame = frame.permute(1, 2, 0).numpy()
		return frame


def make_env(cfg):
	"""
	Make MuJoCo environment.
	"""
	if not cfg.task in MUJOCO_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	if cfg.task in {'mujoco-ant', 'mujoco-hopper', 'mujoco-walker'}:
		env = gym.make(
			MUJOCO_TASKS[cfg.task],
			terminate_when_unhealthy=False,
			render_mode='rgb_array',
		)
	else:
		env = gym.make(
		MUJOCO_TASKS[cfg.task],
		render_mode='rgb_array',
	)
	env = MuJoCoWrapper(env, cfg)
	env = Timeout(env, max_episode_steps={
		'mujoco-reacher': 50,
		'mujoco-pusher': 100,
		'mujoco-halfcheetah': 1000,
	}.get(cfg.task, 500))
	return env
