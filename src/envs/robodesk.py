import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import numpy as np
import gymnasium as gym
from dm_control import mujoco
import robodesk


ROBODESK_TASKS = {
	"rd-open-slide": dict(
		env="open_slide",
		max_episode_steps=100,
	),
	"rd-open-drawer": dict(
		env="open_drawer",
		max_episode_steps=100,
	),
	"rd-stack": dict(
		env="stack",
		max_episode_steps=100,
	),
	"rd-upright-block-off-table": dict(
		env="upright_block_off_table",
		max_episode_steps=100,
	),
	"rd-flat-block-in-bin": dict(
		env="flat_block_in_bin",
		max_episode_steps=100,
	),
	"rd-lift-upright-block": dict(
		env="lift_upright_block",
		max_episode_steps=100,
	),
	"rd-lift-ball": dict(
		env="lift_ball",
		max_episode_steps=100,
	),
	"rd-ball-off-table": dict(
		env="ball_off_table",
		max_episode_steps=100,
	),
	"rd-ball-in-bin": dict(
		env="ball_in_bin",
		max_episode_steps=100,
	),
	"rd-push-red": dict(
		env="push_red",
		max_episode_steps=100,
	),
	"rd-push-green": dict(
		env="push_green",
		max_episode_steps=100,
	),
	"rd-push-blue": dict(
		env="push_blue",
		max_episode_steps=100,
	),
}


class RoboDeskWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		obs_dim = sum(space.shape[0] for k, space in env.observation_space.spaces.items() if k != 'image')
		if self.cfg.obs == 'state':
			self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
		elif self.cfg.obs == 'rgb':
			self.observation_space = gym.spaces.Dict({
				'rgb': gym.spaces.Box(
					low=0, high=255, shape=(3, self.cfg.render_size, self.cfg.render_size), dtype=np.uint8),
				'state': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
			})
		self.action_space = env.action_space
		self.max_episode_steps = env.episode_length

		def render(mode='rgb_array', resize=True):
			# _get_obs calls render with resize=True
			assert mode == 'rgb_array', "Only 'rgb_array' mode is supported"
			if resize and self.cfg.obs != 'rgb':  # Skip rendering
				return None
			params = {'distance': 1.4, 'azimuth': 90, 'elevation': -60,
					'crop_box': (16.75, 25.0, 105.0, 88.75), 'size': self.cfg.render_size}
			camera = mujoco.Camera(
				physics=self.env.physics, height=params['size'],
				width=params['size'], camera_id=-1)
			camera._render_camera.distance = params['distance']
			camera._render_camera.azimuth = params['azimuth']
			camera._render_camera.elevation = params['elevation']
			camera._render_camera.lookat[:] = [0, 0.535, 1.1]
			image = camera.render(depth=False, segmentation=False)
			camera._scene.free()
			return image
		
		self.env.render = render

	def _extract_info(self, info):
		success = self.env.reward_functions[ROBODESK_TASKS[self.cfg.task]['env']]('success')
		info = {
			'terminated': info.get('terminated', False),
			'truncated': info.get('truncated', False),
			'success': float(success),
		}
		info['score'] = info['success']
		return info

	def _flatten(self, obs):
		return np.concatenate([obs[k].flatten() for k in self.env.observation_space.spaces if k != 'image'], dtype=np.float32)
	
	def get_observation(self, obs):
		if self.cfg.obs == 'rgb':
			return {'state': self._flatten(obs), 'rgb': self.render().copy().transpose(2, 0, 1)}
		return self._flatten(obs)

	def reset(self, **kwargs):
		self.env.reset()
		obs, _, _, info = self.env.step(np.zeros(self.env.action_space.shape, dtype=np.float32))
		return self.get_observation(obs), self._extract_info(info)

	def step(self, action):
		obs, reward, truncated, info = self.env.step(action.copy())
		info['truncated'] = truncated
		return self.get_observation(obs), reward, False, truncated, self._extract_info(info)

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def render(self, *args, **kwargs):
		return self.env.render(resize=False)

	def close(self):
		self.env.close()


def make_env(cfg):
	"""
	Make RoboDesk environment.
	"""
	if cfg.task not in ROBODESK_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	env = robodesk.RoboDesk(
		task=ROBODESK_TASKS[cfg.task]['env'],
		reward='dense',
		action_repeat=10,
		episode_length=10*ROBODESK_TASKS[cfg.task]['max_episode_steps']+1,
		image_size=cfg.render_size if cfg.obs == 'rgb' else 1,
	)
	env = RoboDeskWrapper(env, cfg)
	return env
