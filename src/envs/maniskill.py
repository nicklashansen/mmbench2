import warnings
warnings.filterwarnings('ignore')

import gymnasium as gym
import numpy as np
import mani_skill.envs
from mani_skill.utils.wrappers import CPUGymWrapper

import envs.tasks.maniskill
from envs.wrappers.timeout import Timeout


MANISKILL_TASKS = {
	"ms-ant-walk": dict(
		env="MS-AntWalk-v1",
		control_mode=None,
		max_episode_steps=500,
	),
	"ms-ant-run": dict(
		env="MS-AntRun-v1",
		control_mode=None,
		max_episode_steps=500,
	),
	"ms-cartpole-balance": dict(
		env="MS-CartpoleBalance-v1",
		control_mode=None,
		max_episode_steps=500,
	),
	"ms-cartpole-swingup": dict(
		env="MS-CartpoleSwingUp-v1",
		control_mode=None,
		max_episode_steps=500,
	),
	"ms-cartpole-balance-sparse": dict(
		env="MS-CartpoleBalanceSparse-v1",
		control_mode=None,
		max_episode_steps=500,
	),
	"ms-cartpole-swingup-sparse": dict(
		env="MS-CartpoleSwingUpSparse-v1",
		control_mode=None,
		max_episode_steps=500,
	),
	"ms-hopper-stand": dict(
		env="MS-HopperStand-v1",
		control_mode=None,
		max_episode_steps=250,
	),
	"ms-hopper-hop": dict(
		env="MS-HopperHop-v1",
		control_mode=None,
		max_episode_steps=250,
	),
    "ms-pick-cube": dict(
        env="PickCube-v1",
        control_mode="pd_ee_delta_pos",
		max_episode_steps=25,
    ),
	"ms-pick-cube-eepose": dict(
        env="PickCube-v1",
        control_mode="pd_ee_delta_pose",
		max_episode_steps=25,
    ),
	"ms-pick-cube-so": dict(
        env="PickCubeSO100-v1",
        control_mode="pd_joint_delta_pos",
		max_episode_steps=25,
    ),
	"ms-poke-cube": dict(
        env="PokeCube-v1",
        control_mode="pd_ee_delta_pose",
		max_episode_steps=25,
    ),
	"ms-push-cube": dict(
        env="PushCube-v1",
        control_mode="pd_ee_delta_pose",
		max_episode_steps=25,
    ),
	"ms-pull-cube": dict(
        env="PullCube-v1",
        control_mode="pd_ee_delta_pose",
		max_episode_steps=25,
    ),
	"ms-pull-cube-tool": dict(
        env="PullCubeTool-v1",
        control_mode="pd_ee_delta_pose",
		max_episode_steps=50,
    ),
	"ms-stack-cube": dict(
        env="StackCube-v1",
        control_mode="pd_ee_delta_pose",
		max_episode_steps=25,
    ),
	"ms-place-sphere": dict(
        env="PlaceSphere-v1",
        control_mode="pd_ee_delta_pos",
		max_episode_steps=25,
    ),
	"ms-lift-peg": dict(
        env="LiftPegUpright-v1",
        control_mode="pd_ee_delta_pose",
		max_episode_steps=25,
	),
	"ms-pick-apple": dict(
        env="PickSingleYCB-v1",
        control_mode="pd_ee_delta_pose",
		max_episode_steps=25,
		model_id="013_apple",
	),
	"ms-pick-banana": dict(
        env="PickSingleYCB-v1",
        control_mode="pd_ee_delta_pose",
		max_episode_steps=25,
		model_id="011_banana",
	),
	"ms-pick-can": dict(
        env="PickSingleYCB-v1",
        control_mode="pd_ee_delta_pose",
		max_episode_steps=25,
		model_id="005_tomato_soup_can",
	),
	"ms-pick-fork": dict(
        env="PickSingleYCB-v1",
        control_mode="pd_ee_delta_pose",
		max_episode_steps=25,
		model_id="030_fork",
	),
	"ms-pick-hammer": dict(
        env="PickSingleYCB-v1",
        control_mode="pd_ee_delta_pose",
		max_episode_steps=25,
		model_id="048_hammer",
	),
	"ms-pick-knife": dict(
        env="PickSingleYCB-v1",
        control_mode="pd_ee_delta_pose",
		max_episode_steps=25,
		model_id="032_knife",
	),
	"ms-pick-mug": dict(
        env="PickSingleYCB-v1",
        control_mode="pd_ee_delta_pose",
		max_episode_steps=25,
		model_id="025_mug",
	),
	"ms-pick-orange": dict(
        env="PickSingleYCB-v1",
        control_mode="pd_ee_delta_pose",
		max_episode_steps=25,
		model_id="017_orange",
	),
	"ms-pick-screwdriver": dict(
        env="PickSingleYCB-v1",
        control_mode="pd_ee_delta_pose",
		max_episode_steps=25,
		model_id="044_flat_screwdriver",
	),
	"ms-pick-spoon": dict(
        env="PickSingleYCB-v1",
        control_mode="pd_ee_delta_pose",
		max_episode_steps=25,
		model_id="031_spoon",
	),
	"ms-pick-tennis-ball": dict(
        env="PickSingleYCB-v1",
        control_mode="pd_ee_delta_pose",
		max_episode_steps=25,
		model_id="056_tennis_ball",
	),
	"ms-pick-baseball": dict(
        env="PickSingleYCB-v1",
        control_mode="pd_ee_delta_pose",
		max_episode_steps=25,
		model_id="055_baseball",
	),
	"ms-pick-sponge": dict(
        env="PickSingleYCB-v1",
        control_mode="pd_ee_delta_pose",
		max_episode_steps=25,
		model_id="026_sponge",
	),
	"ms-pick-cube-xarm6": dict(
        env="PickCubexArm6-v1",
        control_mode="pd_ee_delta_pos",
		max_episode_steps=25,
	),
	"ms-reach": dict(
        env="Reach-v1",
        control_mode="pd_ee_delta_pos",
		max_episode_steps=25,
    ),
	"ms-reach-eepose": dict(
        env="Reach-v1",
        control_mode="pd_ee_delta_pose",
		max_episode_steps=25,
    ),
	"ms-reach-xarm6": dict(
        env="ReachxArm6-v1",
        control_mode="pd_ee_delta_pos",
		max_episode_steps=25,
    ),
	"ms-anymal-reach": dict(
		env="AnymalC-Reach-v1",
		control_mode="pd_joint_delta_pos",
		max_episode_steps=100,
	),
	# below: held-out tasks (test split)
	"ms-push-apple": dict(
        env="PushYCB-v1",
        control_mode="pd_ee_delta_pose",
		max_episode_steps=25,
		model_id="013_apple",
	),
	"ms-push-pear": dict(
        env="PushYCB-v1",
        control_mode="pd_ee_delta_pose",
		max_episode_steps=25,
		model_id="016_pear",
	),
	"ms-push-rubiks-cube": dict(
        env="PushYCB-v1",
        control_mode="pd_ee_delta_pose",
		max_episode_steps=25,
		model_id="077_rubiks_cube",
	),
	"ms-push-can": dict(
        env="PushYCB-v1",
        control_mode="pd_ee_delta_pose",
		max_episode_steps=25,
		model_id="005_tomato_soup_can",
		spawn_height=0.08,
	),
	"ms-push-sponge": dict(
        env="PushYCB-v1",
        control_mode="pd_ee_delta_pose",
		max_episode_steps=25,
		model_id="026_sponge",
	),
	"ms-push-banana": dict(
        env="PushYCB-v1",
        control_mode="pd_ee_delta_pose",
		max_episode_steps=25,
		model_id="011_banana",
		spawn_height=0.03,
	),
	"ms-push-screwdriver": dict(
        env="PushYCB-v1",
        control_mode="pd_ee_delta_pose",
		max_episode_steps=25,
		model_id="044_flat_screwdriver",
	),
	"ms-pick-rubiks-cube": dict(
        env="PickSingleYCB-v1",
        control_mode="pd_ee_delta_pose",
		max_episode_steps=25,
		model_id="077_rubiks_cube",
	),
	"ms-pick-cup": dict(
        env="PickSingleYCB-v1",
        control_mode="pd_ee_delta_pose",
		max_episode_steps=25,
		model_id="065-a_cups",
	),
	"ms-pick-golf-ball": dict(
        env="PickSingleYCB-v1",
        control_mode="pd_ee_delta_pose",
		max_episode_steps=25,
		model_id="058_golf_ball",
	),
	"ms-pick-soccer-ball": dict(
        env="PickSingleYCB-v1",
        control_mode="pd_ee_delta_pose",
		max_episode_steps=25,
		model_id="053_mini_soccer_ball",
	),
	"ms-pick-wood-block": dict(
        env="PickSingleYCB-v1",
        control_mode="pd_ee_delta_pose",
		max_episode_steps=25,
		model_id="036_wood_block",
	),
}


class ManiSkillWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		if self.cfg.obs == 'state':
			self.observation_space = self.env.single_observation_space
		else:
			self.observation_space = gym.spaces.Dict({
				'rgb': gym.spaces.Box(
					low=0, high=255, shape=(3, self.cfg.render_size, self.cfg.render_size), dtype=np.uint8),
				'state': env.observation_space,
			})
		self.action_space = self.env.single_action_space
		self._cumulative_reward = 0
		if self.cfg.task.startswith('ms-pick-') and MANISKILL_TASKS[cfg.task]['env'] == 'PickSingleYCB-v1':
			model_id = MANISKILL_TASKS[cfg.task]['model_id']
			env.unwrapped.all_model_ids = [model_id]
			env.reset(options=dict(reconfigure=True))
		if self.cfg.task.startswith('ms-push-') and self.cfg.task != 'ms-push-cube':
			model_id = MANISKILL_TASKS[cfg.task]['model_id']
			env.unwrapped.model_id = model_id
			if 'spawn_height' in MANISKILL_TASKS[cfg.task]:
				env.unwrapped.spawn_height = MANISKILL_TASKS[cfg.task]['spawn_height']
			env.reset(options=dict(reconfigure=True))

	def _extract_info(self, info):
		info = {
			'terminated': info.get('terminated', False),
			'truncated': info.get('truncated', False),
			'success': float(info.get('success', 0.)),
		}
		if 'cartpole' in self.cfg.task:
			info['score'] = self._cumulative_reward/1000
		elif 'hopper' in self.cfg.task:
			info['score'] = self._cumulative_reward/600
		elif 'ant' in self.cfg.task:
			# MS-AntWalk / MS-AntRun: continuous locomotion with no success
			# criterion (analogous to mujoco-ant). Normalize cumulative reward
			# over a 500-step episode.
			info['score'] = np.clip(self._cumulative_reward, 0, 1000) / 1000
		else:
			info['score'] = info['success']
		return info

	def get_observation(self, obs):
		if self.cfg.obs == 'state':
			return obs
		return {'state': obs, 'rgb': self.env.render().transpose(2, 0, 1)}

	def reset(self):
		obs, info = self.env.reset()
		self._cumulative_reward = 0
		return self.get_observation(obs), self._extract_info(info)
	
	def step(self, action):
		reward = 0
		for _ in range(2):
			obs, r, terminated, truncated, info = self.env.step(action)
			reward += r
			done = terminated or truncated
			if done:
				break
		self._cumulative_reward += reward
		return self.get_observation(obs), reward, terminated, truncated, self._extract_info(info)

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def render(self, *args, **kwargs):
		return self.env.render()
	

def make_env(cfg):
	"""
	Make ManiSkill3 environment.
	"""
	if cfg.task not in MANISKILL_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	task_cfg = MANISKILL_TASKS[cfg.task]
	env = gym.make(
		task_cfg['env'],
		obs_mode='state',
		control_mode=task_cfg['control_mode'],
		num_envs=1,
		render_mode='rgb_array',
		sensor_configs=dict(width=cfg.render_size, height=cfg.render_size),
		human_render_camera_configs=dict(width=cfg.render_size, height=cfg.render_size),
		reconfiguration_freq=None,
		sim_backend='auto',
	)
	env = CPUGymWrapper(env, ignore_terminations=True)
	env = ManiSkillWrapper(env, cfg)
	env = Timeout(env, max_episode_steps=task_cfg['max_episode_steps'])
	return env
