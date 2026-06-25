import warnings
warnings.filterwarnings('ignore')

import gymnasium as gym

from envs.wrappers.vectorized_multitask import make_vectorized_multitask_env
from envs.wrappers.render import Render
from envs.dmcontrol import make_env as make_dm_control_env
from envs.maniskill import make_env as make_maniskill_env
from envs.metaworld import make_env as make_metaworld_env
from envs.mujoco import make_env as make_mujoco_env
from envs.box2d import make_env as make_box2d_env
from envs.robodesk import make_env as make_robodesk_env
from envs.ogbench import make_env as make_ogbench_env
from envs.pygame import make_env as make_pygame_env
try:
	from envs.atari import make_env as make_atari_env
except ImportError as _atari_err:  # ale_py is optional; only the Atari domain needs it
	def make_atari_env(cfg, _err=_atari_err):
		raise ImportError(
			"The Atari domain requires `ale_py` (install ale_py>=0.10 for continuous-action "
			f"Atari). Original import error: {_err}"
		)


def make_env(cfg):
	"""
	Make an environment for world-model experiments.
	"""
	gym.logger.set_level(40)
	if not cfg.child_env:
		env = make_vectorized_multitask_env(cfg, make_env)
	else:
		env = None
		for fn in [
			make_dm_control_env, make_maniskill_env, make_metaworld_env,
			make_mujoco_env, make_box2d_env, make_robodesk_env,
			make_ogbench_env, make_pygame_env, make_atari_env,
		]:
			try:
				env = fn(cfg)
				break
			except ValueError as e:
				if 'Unknown task' in str(e):
					continue
				else:
					raise e
		if env is None:
			raise ValueError(f'Failed to make environment "{cfg.task}": please verify that dependencies are installed and that the task exists.')
		assert cfg.num_envs == 1 or cfg.get('obs', 'state') == 'state', \
			'Vectorized environments only support state observations.'
		if cfg.save_video and cfg.get('num_demos', 0) > 0:
			env = Render(env, cfg)
		print(f'[Rank {cfg.rank}] Created env for task {cfg.task}')
	try: # Dict
		cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
	except: # Box
		cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
	cfg.action_dim = env.action_space.shape[0]
	cfg.episode_length = env.max_episode_steps
	return env
