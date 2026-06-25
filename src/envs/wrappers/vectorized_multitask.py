from copy import deepcopy

from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict


MAX_OBS_DIM = 128
MAX_ACTION_DIM = 16


class VecWrapper(gym.Wrapper):
	"""
	A simple wrapper that unifies observation and action spaces across vectorized environments.
 	This is useful for environments that have different observation/action spaces per task.
	"""
	def __init__(self, env):
		super().__init__(env)
		self.orig_observation_space = env.observation_space
		self.orig_action_space = env.action_space
		if isinstance(env.observation_space, gym.spaces.Dict):  # State + RGB
			self.observation_space['state'] = gym.spaces.Box(
				low=-np.inf, high=np.inf, shape=(MAX_OBS_DIM,), dtype=np.float32)
		elif len(env.observation_space.shape) == 1:  # State
			self.observation_space = gym.spaces.Box(
				low=-np.inf, high=np.inf, shape=(MAX_OBS_DIM,), dtype=np.float32)
		else:  # RGB
			self.observation_space = env.observation_space
		self.action_space = gym.spaces.Box(
			low=-1, high=1, shape=(MAX_ACTION_DIM,), dtype=np.float32)

	def _pad_obs(self, obs):
		if isinstance(self.observation_space, gym.spaces.Dict):  # State + RGB
			if len(obs['state'].shape) == 1 and obs['state'].shape != self.observation_space['state'].shape:
				pad = np.zeros(self.observation_space['state'].shape[0] - obs['state'].shape[0], dtype=obs['state'].dtype)
				obs['state'] = np.concatenate((obs['state'], pad), axis=-1)
		elif len(self.observation_space.shape) == 1 and obs.shape != self.observation_space.shape:
			pad = np.zeros(self.observation_space.shape[0] - obs.shape[0], dtype=obs.dtype)
			obs = np.concatenate((obs, pad), axis=-1)
		return obs

	def reset(self, **kwargs):
		obs, info = self.env.reset(**kwargs)
		return self._pad_obs(obs), info

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action[:self.orig_action_space.shape[0]])
		return self._pad_obs(obs), reward, terminated, truncated, info

	def render(self, *args, **kwargs):
		return self.env.render(*args, **kwargs).copy()

	def close(self):
		return self.env.close()


class VectorizedMultitaskWrapper(gym.Wrapper):
	"""
	Wrapper for vectorized multi-task environments.
	Each environment in the vectorized setup may have different observation and action spaces,
	and corresponds to different tasks. Creates one environment per task.
	"""

	def __init__(self, cfg, make_fn):
		self.cfg = deepcopy(cfg)
		self.cfg.task_embeddings = []  # Saves memory by not storing copies of embeddings
		
		# Create configs
		cfgs = [deepcopy(self.cfg) for _ in range(self.cfg.num_envs)]
		for i in range(len(cfgs)):
			cfgs[i].task = cfgs[i].tasks[i%self.cfg.num_tasks]
			cfgs[i].tasks = [cfgs[i].task]
			cfgs[i].num_envs = 1
			cfgs[i].child_env = True
			cfgs[i].seed = cfgs[i].seed + np.random.randint(1000)
		
		# Create environment
		wrapper = AsyncVectorEnv if self.cfg.get('env_mode', 'sync') == 'async' else SyncVectorEnv
		env_fns = [lambda c=cfgs[i]: VecWrapper(make_fn(c)) for i in range(self.cfg.num_envs)]
		self.env = wrapper(env_fns)
		super().__init__(self.env)
		
		# Define obs and action spaces
		if self.cfg.obs == 'state':
			self.observation_space = gym.spaces.Box(
				low=-np.inf, high=np.inf, shape=(MAX_OBS_DIM,), dtype=np.float32)
		else:
			self.observation_space = gym.spaces.Dict({
				'state': gym.spaces.Box(
					low=-np.inf, high=np.inf, shape=(MAX_OBS_DIM,), dtype=np.float32),
				'rgb': gym.spaces.Box(
					low=0, high=255, shape=(3, self.cfg.render_size, self.cfg.render_size), dtype=np.uint8),
			})
		self.action_space = gym.spaces.Box(
			low=-1, high=1, shape=(MAX_ACTION_DIM,), dtype=np.float32)
		assert self.action_space.shape[0] == self.env.action_space.shape[1], \
			"Action space mismatch between multitask wrapper and individual envs."		
		self._max_episode_steps = self.cfg.episode_lengths
		self.max_episode_steps = max(self._max_episode_steps)
		if self.cfg.rank == 0:
			print('Episode lengths:', self._max_episode_steps)
			print('Action dims:', self.cfg.action_dims)

	def rand_act(self):
		return torch.rand((self.cfg.num_envs, *self.action_space.shape)) * 2 - 1
	
	def _preprocess_obs(self, obs):
		if self.cfg.obs == 'state':
			obs = torch.tensor(obs, dtype=torch.float32)
		else:  # State + RGB
			obs = {
				'state': torch.tensor(obs['state'], dtype=torch.float32),
				'rgb': torch.tensor(obs['rgb'], dtype=torch.uint8),
			}
		return obs

	def reset(self):
		obs, info = self.env.reset()
		return self._preprocess_obs(obs), self._preprocess_info(info)
	
	def _preprocess_info(self, info):
		if 'final_info' in info:  # Handle final transitions
			assert 'final_observation' in info, \
				'Expected final observation in info when final_info is present.'
			fp64_to_fp32 = lambda x: x.astype(np.float32) if isinstance(x, np.ndarray) and x.dtype == np.float64 else x
			np_to_torch = lambda x: torch.from_numpy(fp64_to_fp32(x)) if isinstance(x, np.ndarray) else TensorDict({k: np_to_torch(v) for k, v in x.items()})
			info['final_observation'] = torch.stack([np_to_torch(d) for d in info['final_observation'] if d is not None])
			if self.cfg.save_video and self.cfg.get('num_demos', 0) > 0:
				info['final_frame'] = torch.stack([torch.from_numpy(d['frame']) for d in info['final_info'] if d is not None])
			keys = next(d.keys() for d in info['final_info'] if d is not None)
			pad = lambda vals: torch.from_numpy(
				np.stack([np.asarray(v, dtype=np.float32) if v is not None
					else np.full_like(next(x for x in vals if x is not None), np.nan, dtype=np.float32) for v in vals]))
			info['final_info'] = {
				k: pad([d[k] if d is not None else None for d in info['final_info']]) for k in keys}
		info['success'] = torch.tensor(info['success'], dtype=torch.float32)
		if 'frame' in info:
			info['frame'] = torch.stack([torch.from_numpy(d) for d in info['frame']])
		return info

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action.numpy())
		return self._preprocess_obs(obs), \
			   torch.tensor(reward, dtype=torch.float32), \
			   torch.tensor(terminated, dtype=torch.bool), \
			   torch.tensor(truncated, dtype=torch.bool), \
			   self._preprocess_info(info)
	
	def render(self, *args, **kwargs):
		frames = []
		for env in self.env.envs:
			frame = env.render(*args, **kwargs)
			if frame is not None:
				frames.append(torch.from_numpy(frame))
		return torch.cat(frames, dim=1) if len(frames) > 0 else None


def make_vectorized_multitask_env(cfg, make_fn):
	"""Make a vectorized multi-task environment for world-model experiments."""
	print(f'[Rank {cfg.rank}] Creating multi-task environment with tasks: {cfg.tasks}')
	env = VectorizedMultitaskWrapper(cfg, make_fn)
	return env
