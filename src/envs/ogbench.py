import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
import torch
from torchvision.transforms import functional as F
import ogbench
import ogbench.locomaze
from ogbench.online_locomotion.wrappers import GymXYWrapper

from envs.wrappers.timeout import Timeout


register(
    id='pointmaze-arena-v0',
    entry_point='ogbench.locomaze.maze:make_maze_env',
    max_episode_steps=400,
    kwargs=dict(
        loco_env_type='point',
        maze_env_type='maze',
        maze_type='arena',
    ),
)
register(
    id='pointmaze-bottleneck-v0',
    entry_point='envs.tasks.ogbench_maze:make_maze_env',
    max_episode_steps=400,
    kwargs=dict(
        loco_env_type='point',
        maze_env_type='maze',
        maze_type='bottleneck',
    ),
)
register(
    id='pointmaze-circle-v0',
    entry_point='envs.tasks.ogbench_maze:make_maze_env',
    max_episode_steps=400,
    kwargs=dict(
        loco_env_type='point',
        maze_env_type='maze',
        maze_type='circle',
    ),
)
register(
    id='pointmaze-spiral-v0',
    entry_point='envs.tasks.ogbench_maze:make_maze_env',
    max_episode_steps=400,
    kwargs=dict(
        loco_env_type='point',
        maze_env_type='maze',
        maze_type='spiral',
    ),
)
register(
    id='antmaze-arena-v0',
    entry_point='ogbench.locomaze.maze:make_maze_env',
    max_episode_steps=400,
    kwargs=dict(
        loco_env_type='ant',
        maze_env_type='maze',
        maze_type='arena',
    ),
)
register(
    id='antmaze-bottleneck-v0',
    entry_point='envs.tasks.ogbench_maze:make_maze_env',
    max_episode_steps=400,
    kwargs=dict(
        loco_env_type='ant',
        maze_env_type='maze',
        maze_type='bottleneck',
    ),
)
register(
    id='antmaze-circle-v0',
    entry_point='envs.tasks.ogbench_maze:make_maze_env',
    max_episode_steps=400,
    kwargs=dict(
        loco_env_type='ant',
        maze_env_type='maze',
        maze_type='circle',
    ),
)
register(
    id='antmaze-spiral-v0',
    entry_point='envs.tasks.ogbench_maze:make_maze_env',
    max_episode_steps=400,
    kwargs=dict(
        loco_env_type='ant',
        maze_env_type='maze',
        maze_type='spiral',
    ),
)
# below are reserved for testing
register(
    id='pointmaze-var1-v0',
    entry_point='envs.tasks.ogbench_maze:make_maze_env',
    max_episode_steps=400,
    kwargs=dict(
        loco_env_type='point',
        maze_env_type='maze',
        maze_type='var1',
    ),
)
register(
    id='pointmaze-var2-v0',
    entry_point='envs.tasks.ogbench_maze:make_maze_env',
    max_episode_steps=400,
    kwargs=dict(
        loco_env_type='point',
        maze_env_type='maze',
        maze_type='var2',
    ),
)
register(
    id='pointmaze-var3-v0',
    entry_point='envs.tasks.ogbench_maze:make_maze_env',
    max_episode_steps=400,
    kwargs=dict(
        loco_env_type='point',
        maze_env_type='maze',
        maze_type='var3',
    ),
)
register(
    id='pointmaze-var4-v0',
    entry_point='envs.tasks.ogbench_maze:make_maze_env',
    max_episode_steps=400,
    kwargs=dict(
        loco_env_type='point',
        maze_env_type='maze',
        maze_type='var4',
    ),
)
register(
    id='pointmaze-var5-v0',
    entry_point='envs.tasks.ogbench_maze:make_maze_env',
    max_episode_steps=400,
    kwargs=dict(
        loco_env_type='point',
        maze_env_type='maze',
        maze_type='var5',
    ),
)


OGBENCH_TASKS = {
	'og-ant': 'online-ant-v0',
	'og-antball': 'online-antball-v0',
	'og-point-arena': 'pointmaze-arena-v0',
	'og-point-maze': 'pointmaze-medium-v0',
	'og-point-bottleneck': 'pointmaze-bottleneck-v0',
	'og-point-circle': 'pointmaze-circle-v0',
	'og-point-spiral': 'pointmaze-spiral-v0',
	'og-ant-arena': 'antmaze-arena-v0',
	'og-ant-maze': 'antmaze-medium-v0',
	'og-ant-bottleneck': 'antmaze-bottleneck-v0',
	'og-ant-circle': 'antmaze-circle-v0',
	'og-ant-spiral': 'antmaze-spiral-v0',
	# below are reserved for testing
	'og-point-var1': 'pointmaze-var1-v0',
	'og-point-var2': 'pointmaze-var2-v0',
	'og-point-var3': 'pointmaze-var3-v0',
	'og-point-var4': 'pointmaze-var4-v0',
	'og-point-var5': 'pointmaze-var5-v0',
}


class OGBenchWrapper(gym.Wrapper):
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
			if 'point' in cfg.task:
				self.observation_space.spaces['state'] = gym.spaces.Box(
					low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
		elif 'point' in cfg.task:
			self.observation_space = gym.spaces.Box(
				low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
		self._cumulative_reward = 0
		if 'maze' in OGBENCH_TASKS[cfg.task]:
			self.action_repeat = 2 if 'ant' in cfg.task else 4
		else:
			self.action_repeat = 1

		if self.cfg.task == 'og-antball':
			# Move camera closer to the ant
			self.env.mujoco_renderer.viewer.cam.distance = 8
		elif self.cfg.task == 'og-move-cube':
			self.env.unwrapped._mode = 'data_collection'
		elif 'arena' in self.cfg.task:
			task_infos = []
			for init_i in range(2, 5):
				for init_j in range(2, 5):
					for goal_i in range(2, 5):
						for goal_j in range(2, 5):
							# Ensure init and goal are some distance apart
							if (abs(init_i - goal_i) + abs(init_j - goal_j) < 2):
								continue
							task_id = len(task_infos) + 1
							task_info = {
								'task_name': f'task{task_id}',
								'init_ij': (init_i, init_j),
								'goal_ij': (goal_i, goal_j),
								'init_xy': self.env.unwrapped.ij_to_xy((init_i, init_j)),
								'goal_xy': self.env.unwrapped.ij_to_xy((goal_i, goal_j)),
							}
							task_infos.append(task_info)
			self.env.unwrapped.task_infos = task_infos
			self.env.unwrapped.num_tasks = len(task_infos)
		elif 'maze' in self.cfg.task:
			# move goal/init in tasks 3 and 4 to make exploration a bit easier
			self.env.unwrapped.task_infos[2]['goal_ij'] = (2, 4)
			self.env.unwrapped.task_infos[2]['goal_xy'] = self.env.unwrapped.ij_to_xy((2, 4))
			self.env.unwrapped.task_infos[3]['init_ij'] = (2, 2)
			self.env.unwrapped.task_infos[3]['init_xy'] = self.env.unwrapped.ij_to_xy((2, 2))
			self.env.unwrapped.task_infos.extend([
				{'task_name': 'task6',
	 			 'init_ij': (2, 1),
				 'init_xy': self.env.unwrapped.ij_to_xy((2, 1)),
				 'goal_ij': (4, 5),
				 'goal_xy': self.env.unwrapped.ij_to_xy((4, 5))},
				{'task_name': 'task7',
	 			 'init_ij': (6, 5),
				 'init_xy': self.env.unwrapped.ij_to_xy((6, 5)),
				 'goal_ij': (4, 6),
				 'goal_xy': self.env.unwrapped.ij_to_xy((4, 6)),
				},
				{'task_name': 'task8',
	 			 'init_ij': (2, 2),
				 'init_xy': self.env.unwrapped.ij_to_xy((2, 2)),
				 'goal_ij': (4, 4),
				 'goal_xy': self.env.unwrapped.ij_to_xy((4, 4)),
				},
				{'task_name': 'task9',
	 			 'init_ij': (6, 1),
				 'init_xy': self.env.unwrapped.ij_to_xy((6, 1)),
				 'goal_ij': (2, 1),
				 'goal_xy': self.env.unwrapped.ij_to_xy((2, 1)),
				},
				{'task_name': 'task10',
	 			 'init_ij': (6, 3),
				 'init_xy': self.env.unwrapped.ij_to_xy((6, 3)),
				 'goal_ij': (3, 3),
				 'goal_xy': self.env.unwrapped.ij_to_xy((3, 3)),
				},
			])
			self.env.unwrapped.num_tasks = len(self.env.unwrapped.task_infos)

	def _extract_info(self, info):
		info = {
			'terminated': info.get('terminated', False),
			'truncated': info.get('truncated', False),
			'success': float(info.get('success', 0.)),
		}
		if self.cfg.task == 'og-ant':
			# Task has no success criterion so we use cumulative reward
			info['score'] = np.clip(self._cumulative_reward, 0, 250) / 250
		else:
			info['score'] = info['success']
		return info

	def get_observation(self, obs, info=None):
		if 'maze' in OGBENCH_TASKS[self.cfg.task]:
			assert info is not None
			xy = self.env.get_xy()
			goal_xy = self.env.get_oracle_rep()
			if 'point' in self.cfg.task:
				prev_xy = info.get('prev_qpos', xy)
				vel_xy = (xy - prev_xy)
				obs = np.concatenate([
					xy,
					vel_xy,
					goal_xy,
					xy - goal_xy,
					np.array([np.linalg.norm(xy - goal_xy)]),
				]) / 20.
			elif 'ant' in self.cfg.task:
				obs = np.concatenate([
					obs, # qpos and qvel
					xy,
					goal_xy,
					xy - goal_xy,
					np.array([np.linalg.norm(xy - goal_xy)]),
				]) / 20
		obs = obs.astype(np.float32)
		if self.cfg.obs == 'rgb':
			return {'state': obs, 'rgb': self.render().transpose(2, 0, 1)}
		return obs
	
	def get_success(self):
		if 'maze' in OGBENCH_TASKS[self.cfg.task]:
			xy = self.env.get_xy()
			goal_xy = self.env.get_oracle_rep()
			return float(np.linalg.norm(xy - goal_xy) <= self.env.unwrapped._goal_tol)
		raise NotImplementedError('Custom reward function not implemented for this task')

	def get_reward(self, info):
		if 'maze' in OGBENCH_TASKS[self.cfg.task]:
			xy = info['xy']
			goal_xy = self.env.get_oracle_rep()
			l1_dist = 0.5 * np.abs(xy - goal_xy).sum()
			l2_dist = 0.5 * np.linalg.norm(xy - goal_xy)
			if 'point' in self.cfg.task:
				vel_penalty = 0.025 * np.linalg.norm(xy - info['prev_qpos'])
			elif 'ant' in self.cfg.task:
				vel_penalty = 0.001 * np.linalg.norm(info['qvel'])
			return info['success'] - (l1_dist + l2_dist + vel_penalty) / 20.
		raise NotImplementedError('Custom reward function not implemented for this task')

	def reset(self):
		obs, info = self.env.reset()

		if self.cfg.task == 'og-antball':
			# Move goal closer to the ant
			goal_xy = np.random.uniform(low=-3, high=3, size=2)
			self.env.set_goal(goal_xy)
			
			# Recompute observation
			agent_xy, ball_xy = self.env.get_agent_ball_xy()
			qpos = self.env.data.qpos.flat.copy()
			qvel = self.env.data.qvel.flat.copy()
			obs = np.concatenate([qpos[2:-7], qpos[-5:], qvel, ball_xy - agent_xy, np.array(self.env.cur_goal_xy) - ball_xy])

		self._cumulative_reward = 0
		return self.get_observation(obs, info), self._extract_info(info)

	def step(self, action):
		reward = 0
		for _ in range(self.action_repeat):
			obs, _reward, _, truncated, info = self.env.step(action)
			reward += _reward
			if truncated:
				break
		if 'maze' in OGBENCH_TASKS[self.cfg.task]:
			info['success'] = self.get_success()
			reward = self.get_reward(info)
		self._cumulative_reward += reward
		info['terminated'] = False
		info['truncated'] = truncated
		return self.get_observation(obs, info), reward, False, truncated, self._extract_info(info)

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
	Make OGBench environment.
	"""
	if not cfg.task in OGBENCH_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	env = gym.make(OGBENCH_TASKS[cfg.task], render_mode='rgb_array', height=cfg.render_size, width=cfg.render_size)
	if cfg.task == 'og-ant':
		env = GymXYWrapper(env, resample_interval=100)
	env = Timeout(env, max_episode_steps={
			'og-ant': 1000,
			'og-antball': 200,
			'og-move-cube': 200,
		}.get(cfg.task, 400)
	)
	env = OGBenchWrapper(env, cfg)
	return env
