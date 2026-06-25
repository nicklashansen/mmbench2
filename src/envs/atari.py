import warnings
warnings.filterwarnings('ignore')

import gymnasium as gym
import numpy as np
import ale_py

from envs.wrappers.timeout import Timeout


ATARI_TASKS = {
	'atari-alien': dict(
        env='ALE/Alien-v5',
        min_return=0,
        max_return=1000,
	),
	'atari-assault': dict(
        env='ALE/Assault-v5',
        min_return=0,
        max_return=1000,
	),
	'atari-asterix': dict(
        env='ALE/Asterix-v5',
        min_return=0,
        max_return=5000,
	),
	'atari-atlantis': dict(
        env='ALE/Atlantis-v5',
        min_return=0,
        max_return=50000,
	),
	'atari-bank-heist': dict(
        env='ALE/BankHeist-v5',
        min_return=0,
        max_return=1000,
	),
	'atari-battle-zone': dict(
        env='ALE/BattleZone-v5',
        min_return=0,
        max_return=15000,
	),
	'atari-beamrider': dict(
        env='ALE/BeamRider-v5',
        min_return=0,
        max_return=2000,
	),
	'atari-berzerk': dict(
        env='ALE/Berzerk-v5',
        min_return=0,
        max_return=500,
	),
	'atari-bowling': dict(
        env='ALE/Bowling-v5',
        min_return=0,
        max_return=100,
	),
	'atari-boxing': dict(
        env='ALE/Boxing-v5',
        min_return=-100,
        max_return=100,
	),
	'atari-chopper-command': dict(
        env='ALE/ChopperCommand-v5',
        min_return=0,
        max_return=1000,
	),
	'atari-crazy-climber': dict(
        env='ALE/CrazyClimber-v5',
		min_return=0,
		max_return=20000,
	),
	'atari-double-dunk': dict(
        env='ALE/DoubleDunk-v5',
		min_return=0,
		max_return=25,
	),
	'atari-enduro': dict(
        env='ALE/Enduro-v5',
		min_return=0,
		max_return=2500,
	),
	'atari-fishing-derby': dict(
        env='ALE/FishingDerby-v5',
		min_return=0,
		max_return=80,
	),
	'atari-gopher': dict(
        env='ALE/Gopher-v5',
        min_return=0,
        max_return=2000,
	),
	'atari-gravitar': dict(
        env='ALE/Gravitar-v5',
        min_return=0,
        max_return=5000,
	),
	'atari-ice-hockey': dict(
        env='ALE/IceHockey-v5',
        min_return=-15,
        max_return=15,
	),
	'atari-jamesbond': dict(
        env='ALE/Jamesbond-v5',
        min_return=0,
        max_return=1000,
	),
	'atari-kangaroo': dict(
        env='ALE/Kangaroo-v5',
        min_return=0,
        max_return=10000,
	),
	'atari-krull': dict(
        env='ALE/Krull-v5',
        min_return=0,
        max_return=10000,
	),
	'atari-ms-pacman': dict(
        env='ALE/MsPacman-v5',
        min_return=0,
        max_return=5000,
	),
	'atari-name-this-game': dict(
        env='ALE/NameThisGame-v5',
        min_return=0,
        max_return=3000,
	),
	'atari-phoenix': dict(
        env='ALE/Phoenix-v5',
        min_return=0,
        max_return=1000,
	),
	'atari-pong': dict(
        env='ALE/Pong-v5',
        min_return=-21,
        max_return=21,
	),
	'atari-riverraid': dict(
        env='ALE/Riverraid-v5',
        min_return=0,
        max_return=5000,
	),
	'atari-road-runner': dict(
        env='ALE/RoadRunner-v5',
        min_return=0,
        max_return=50000,
	),
	'atari-robotank': dict(
        env='ALE/Robotank-v5',
        min_return=0,
        max_return=50,
	),
	'atari-seaquest': dict(
        env='ALE/Seaquest-v5',
        min_return=0,
        max_return=5000,
	),
	'atari-space-invaders': dict(
        env='ALE/SpaceInvaders-v5',
        min_return=0,
        max_return=2000,
	),
	'atari-tennis': dict(
        env='ALE/Tennis-v5',
        min_return=-21,
        max_return=21,
	),
	'atari-tutankham': dict(
        env='ALE/Tutankham-v5',
        min_return=0,
        max_return=100,
	),
	'atari-upndown': dict(
        env='ALE/UpNDown-v5',
        min_return=0,
        max_return=5000,
	),
	'atari-wizard-of-wor': dict(
        env='ALE/WizardOfWor-v5',
        min_return=0,
        max_return=5000,
	),
	'atari-yars-revenge': dict(
        env='ALE/YarsRevenge-v5',
		min_return=0,
		max_return=20000,
	),
}


class AtariWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		if cfg.obs == 'rgb':
			self.observation_space = gym.spaces.Dict({
				'rgb': gym.spaces.Box(
					low=0, high=255, shape=(3, self.cfg.render_size, self.cfg.render_size), dtype=np.uint8),
				'state': gym.spaces.Box(
					low=-np.inf, high=np.inf, shape=(128,), dtype=np.float32)
			})
		else:
			self.observation_space = gym.spaces.Box(
				low=-np.inf, high=np.inf, shape=(128,), dtype=np.float32)
		# Actions are radius, theta, and fire, where first two are the parameters of polar coordinates.
		self.action_space = gym.spaces.Box(
			low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
		self._cumulative_reward = 0
		self._min_return = ATARI_TASKS[cfg.task]['min_return']
		self._max_return = ATARI_TASKS[cfg.task]['max_return']
		self._canvas = np.zeros((224, 224, 3), dtype=np.uint8)

	def _extract_info(self, info):
		info = {
			'terminated': info.get('terminated', False),
			'truncated': info.get('truncated', False),
			'success': float(info.get('success', 0.)),
		}
		info['score'] = np.clip(
			(self._cumulative_reward - self._min_return) / (self._max_return - self._min_return), 0, 1)
		return info
	
	def get_observation(self, obs):
		if self.cfg.obs == 'rgb':
			return {'state': obs, 'rgb': self.render().transpose(2, 0, 1)}
		return obs.astype(np.float32) / 255.

	def reset(self):
		obs, info = self.env.reset()
		self._cumulative_reward = 0
		return self.get_observation(obs), self._extract_info(info)

	def _map_action(self, action):
		# Map action from [-1, 1] to the Atari action space.
		low, high = self.env.action_space.low, self.env.action_space.high
		return action * (high - low) / 2 + (high + low) / 2

	def step(self, action):
		action = self._map_action(action)
		obs, reward, terminated, truncated, info = self.env.step(action)
		terminated = False
		self._cumulative_reward += reward
		info['terminated'] = terminated
		info['truncated'] = truncated
		return self.get_observation(obs), reward, terminated, truncated, self._extract_info(info)

	@property
	def unwrapped(self):
		return self.env.unwrapped
	
	def render(self, **kwargs):
		frame = self.env.render()  # (210, 160, 3)
		h, w = self.cfg.render_size, self.cfg.render_size
		h_start = (h - frame.shape[0]) // 2
		w_start = (w - frame.shape[1]) // 2
		self._canvas[h_start:h_start + frame.shape[0], w_start:w_start + frame.shape[1]] = frame
		return self._canvas.copy()


def make_env(cfg):
	"""
	Make Atari environment.
	"""
	if not cfg.task in ATARI_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	env = gym.make(
		ATARI_TASKS[cfg.task]['env'],
		obs_type='ram',
		continuous=True,
		repeat_action_probability=0,
		render_mode='rgb_array',
	)
	env = AtariWrapper(env, cfg)
	env = Timeout(env, max_episode_steps=ATARI_TASKS[cfg.task].get('max_episode_steps', 1_000))
	return env
