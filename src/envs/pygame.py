import numpy as np
import gymnasium as gym

import envs.tasks.pygame as games


PYGAME_TASKS = {
	'pygame-cowboy': games.CowboyEnv,
	'pygame-coinrun': games.CoinRunEnv,
	'pygame-spaceship': games.SpaceshipEnv,
	'pygame-pong': games.PongEnv,
	'pygame-bird-attack': games.BirdAttackEnv,
	'pygame-highway': games.HighwayEnv,
	'pygame-landing': games.LandingEnv,
	'pygame-air-hockey': games.AirHockeyEnv,
	'pygame-rocket-collect': games.RocketCollectEnv,
	'pygame-chase-evade': games.ChaseEvadeEnv,
	'pygame-coconut-dodge': games.CoconutDodgeEnv,
	'pygame-cartpole-balance': games.CartpoleBalanceEnv,
	'pygame-cartpole-swingup': games.CartpoleSwingupEnv,
	'pygame-cartpole-balance-sparse': games.CartpoleBalanceSparseEnv,
	'pygame-cartpole-swingup-sparse': games.CartpoleSwingupSparseEnv,
	'pygame-cartpole-tremor': games.CartpoleTremorEnv,
	'pygame-point-maze-var1': games.PointMazeVariant1Env,
	'pygame-point-maze-var2': games.PointMazeVariant2Env,
	'pygame-point-maze-var3': games.PointMazeVariant3Env,
	# below are reserved for testing
	'pygame-point-maze-var4': games.PointMazeVariant4Env,
	'pygame-reacher-easy': games.ReacherEasyEnv,
	'pygame-reacher-hard': games.ReacherHardEnv,
	# MiniArcade tasks
	'pygame-foraging': games.ForagingEnv,
	'pygame-orchard-gardener': games.OrchardGardenerEnv,
	'pygame-frozen-lake-6x6': games.FrozenLake6x6Env,
	'pygame-frozen-lake-5x5': games.FrozenLake5x5Env,
	'pygame-frozen-lake-4x4': games.FrozenLake4x4Env,
	'pygame-potion-recipe1': games.PotionRecipe1Env,
	'pygame-potion-recipe2': games.PotionRecipe2Env,
	'pygame-potion-recipe3': games.PotionRecipe3Env,
	'pygame-potion-recipe4': games.PotionRecipe4Env,
	'pygame-potion-recipe5': games.PotionRecipe5Env,
	'pygame-potion-recipe6': games.PotionRecipe6Env,
	'pygame-potion-recipe7': games.PotionRecipe7Env,
	'pygame-potion-recipe8': games.PotionRecipe8Env,
	'pygame-potion-recipe9': games.PotionRecipe9Env,
	'pygame-potion-recipe10': games.PotionRecipe10Env,
	'pygame-dungeon-explorer1': games.DungeonExplorer1Env,
	'pygame-dungeon-explorer2': games.DungeonExplorer2Env,
	'pygame-dungeon-explorer3': games.DungeonExplorer3Env,
	'pygame-dungeon-explorer4': games.DungeonExplorer4Env,
	'pygame-dungeon-explorer5': games.DungeonExplorer5Env,
	'pygame-dungeon-explorer6': games.DungeonExplorer6Env,
	'pygame-dungeon-explorer7': games.DungeonExplorer7Env,
	'pygame-dungeon-explorer8': games.DungeonExplorer8Env,
	'pygame-dungeon-explorer9': games.DungeonExplorer9Env,
	'pygame-dungeon-explorer10': games.DungeonExplorer10Env,
	'pygame-firefly': games.FireflyEnv,
	'pygame-plankton': games.PlanktonEnv,
	'pygame-whirlpool': games.WhirlpoolEnv,
}


class PygameWrapper(gym.Wrapper):
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
		if 'cowboy' in self.cfg.task:
			info['score'] = np.clip(self._cumulative_reward / 500, 0, 1)
		elif 'coinrun' in self.cfg.task:
			info['score'] = np.clip(self._cumulative_reward / 50, 0, 1)
		elif 'spaceship' in self.cfg.task:
			info['score'] = np.clip(self._cumulative_reward / 60, 0, 1)
		elif 'pong' in self.cfg.task:  # convert [-3, 3] to [0, 1]
			info['score'] = np.clip((self._cumulative_reward + 3) / 6, 0, 1)
		elif 'bird-attack' in self.cfg.task:
			info['score'] = np.clip(self._cumulative_reward / 16, 0, 1)
		elif 'highway' in self.cfg.task:
			info['score'] = np.clip(self._cumulative_reward / 10, 0, 1)
		elif 'landing' in self.cfg.task:
			info['score'] = np.clip(self._cumulative_reward / 200, 0, 1)
		elif 'air-hockey' in self.cfg.task:
			info['score'] = np.clip(self._cumulative_reward / 10, 0, 1)
		elif 'rocket-collect' in self.cfg.task:
			info['score'] = np.clip(self._cumulative_reward / 25, 0, 1)
		elif 'chase-evade' in self.cfg.task:  # convert [-50, 50] to [0, 1]
			info['score'] = np.clip((self._cumulative_reward + 50) / 100, 0, 1)
		elif 'coconut-dodge' in self.cfg.task:  # convert [-15, 0] to [0, 1]
			info['score'] = np.clip((self._cumulative_reward + 15) / 15, 0, 1)
		elif 'cartpole' in self.cfg.task:
			info['score'] = np.clip(self._cumulative_reward / 500, 0, 1)
		elif 'point-maze' in self.cfg.task:
			info['score'] = info['success']
		elif 'reacher' in self.cfg.task:
			info['score'] = np.clip(self._cumulative_reward / 200, 0, 1)
		elif 'foraging' in self.cfg.task:
			info['score'] = np.clip(self._cumulative_reward / 100, 0, 1)
		elif 'potion-recipe' in self.cfg.task:
			info['score'] = info['success']
		elif 'dungeon-explorer' in self.cfg.task:
			info['score'] = info['success']
		elif 'orchard-gardener' in self.cfg.task:
			# 500 steps, 5 trees with shaped reward + harvest(+0.3) + deliver(+1.5).
			info['score'] = np.clip(self._cumulative_reward / 30, 0, 1)
		elif 'frozen-lake' in self.cfg.task:
			# +1 per beacon (1 for 4x4, 2 for 5x5/6x6) plus minor shaping.
			n_flags = 1 if '4x4' in self.cfg.task else 2
			info['score'] = np.clip(self._cumulative_reward / (n_flags + 0.5), 0, 1)
		elif 'firefly' in self.cfg.task:
			# 6 captures × +1.0 reward.
			info['score'] = np.clip(self._cumulative_reward / 6, 0, 1)
		elif 'plankton' in self.cfg.task:
			# Mix of small (+1.0) and big (+5.0) motes; ~10 covers a typical good run.
			info['score'] = np.clip(self._cumulative_reward / 10, 0, 1)
		elif 'whirlpool' in self.cfg.task:
			# 7 fish × +1.0 catch reward.
			info['score'] = np.clip(self._cumulative_reward / 7, 0, 1)
		else:
			info['score'] = 0
		return info

	def get_observation(self, obs):
		if self.cfg.obs == 'rgb':
			return {'state': obs, 'rgb': self.render().transpose(2, 0, 1)}
		return obs

	def reset(self):
		obs, info = self.env.reset()
		self._cumulative_reward = 0
		return self.get_observation(obs), self._extract_info(info)

	def step(self, action):
		obs, reward, _, truncated, info = self.env.step(action.copy())
		terminated = False
		self._cumulative_reward += reward
		info['terminated'] = terminated
		info['truncated'] = truncated
		return self.get_observation(obs), reward, terminated, truncated, self._extract_info(info)

	@property
	def unwrapped(self):
		return self.env.unwrapped
	
	def render(self, **kwargs):
		return self.env.render()


def make_env(cfg):
	"""
	Make Pygame (MiniArcade) environment.
	"""
	if not cfg.task in PYGAME_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	env = PYGAME_TASKS[cfg.task]()
	env = PygameWrapper(env, cfg)
	return env
