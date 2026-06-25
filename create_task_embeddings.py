import os
from collections import OrderedDict
import json

import torch
from transformers import CLIPTokenizer, CLIPTextModel


def save_dict_as_json(data, filepath):
	# Convert any torch.Tensors to Python scalars/lists
	def convert(obj):
		if isinstance(obj, torch.Tensor):
			return obj.tolist()
		if isinstance(obj, dict):
			return {k: convert(v) for k, v in obj.items()}
		if isinstance(obj, list):
			return [convert(i) for i in obj]
		return obj

	with open(filepath, 'w') as f:
		json.dump(convert(data), f, indent=2)


# Embodiment descriptions
EMB_WALKER = '2D walker with 6 controllable joints across 2 legs'
EMB_CHEETAH = '2D cheetah with 6 controllable joints across 2 legs'
EMB_GIRAFFE = '2D giraffe with 7 controllable joints across 2 legs and the neck'
EMB_SPINNER = '2D spinner with 5 controllable joints across 5 arms'
EMB_SPINNER_FOUR_ARMS = '2D spinner with 4 controllable joints across 4 arms'
EMB_JUMPER = '2D jumper with 2 controllable joints across 2 legs'
EMB_REACHER = '2D reacher with 2 controllable joints'
EMB_REACHER3 = '2D reacher with 3 controllable joints'
EMB_ACROBOT = '2D acrobot with 1 controllable joint, 2 links total'
EMB_PENDULUM = '2D pendulum with 1 controllable joint'
EMB_CARTPOLE = '2D cartpole with 1 controllable joint (the cart) and a pole'
EMB_CUP_CATCH = '2D cup that moves in any 2D direction and has a ball attached by string'
EMB_FINGER = '2D finger with 2 controllable joints'
EMB_FISH = '3D fish with 5 controllable joints'
EMB_HOPPER = '2D hopper with 4 controllable joints, 1 leg'
EMB_QUADRUPED = '3D quadruped ("ant") with 12 controllable joints across 4 legs'
EMB_METAWORLD = 'Sawyer robot with a 2-finger gripper and delta end-effector position control'
EMB_MANISKILL_CARTPOLE = 'Very simple cartpole with 1 controllable joint (the cart)'
EMB_MANISKILL_HOPPER = 'Classic 2D hopper with 4 controllable joints'
EMB_MANISKILL_ANT = 'Quadruped (ant) with 12 controllable joints across 4 legs'
EMB_MANISKILL_FRANKA_POS = 'Franka robot with a 2-finger gripper and delta end-effector position control'
EMB_MANISKILL_FRANKA_POSE = 'Franka robot with a 2-finger gripper and delta end-effector 6D-pose control'
EMB_MANISKILL_SO100_POS = 'SO100 robot with a 2-finger gripper and delta joint position control'
EMB_MANISKILL_WIDOWX_POS = 'WidowX robot with a 2-finger gripper and delta joint position control'
EMB_MANISKILL_XARM6_POS = 'xArm6 robot with a 2-finger gripper and delta end-effector position control'
EMB_MANISKILL_ANYMAL = 'Quadruped (Anymal robot) with 12 controllable joints across 4 legs'
EMB_MUJOCO_ANT = '3D MuJoCo ant (quadruped) with 8 controllable joints across 4 legs'
EMB_MUJOCO_INVERTED_PENDULUM = '2D inverted pendulum problem with a controllable cart'
EMB_MUJOCO_REACHER = '2D MuJoCo reacher with 2 controllable joints'
EMB_MUJOCO_PUSHER = 'Simple multi-jointed robot arm with a 2-finger gripper and joint torque control'
EMB_MUJOCO_HALFCHEETAH = '2D MuJoCo half-cheetah with 6 controllable joints'
EMB_MUJOCO_HOPPER = '2D MuJoCo hopper with 3 controllable joints'
EMB_MUJOCO_WALKER = '2D MuJoCo walker with 6 controllable joints across its 2 legs'
EMB_LUNARLANDER = '2D lunar lander with 2 controllable thrusters'
EMB_BIPEDAL_WALKER = '2D cartoonish bipedal walker with 4 controllable joints across 2 legs'
EMB_ROBODESK_FRANKA_POSE = 'Franka robot with a 2-finger gripper and delta joint position control'
EMB_OGBENCH_POINTMASS = 'Simple 2D point mass with 2 controllable actions: x and y velocity'
EMB_OGBENCH_ANT = '3D ant (quadruped) with 8 controllable joints across 4 legs'
EMB_PYGAME_SCROLLER = '2D side-scroller game with 1 controllable action: jump'
EMB_PYGAME_SPACESHIP = '2D side-scroller controlling a spaceship that moves in X and Y directions'
EMB_PYGAME_PONG = '2D pong game with 1 controllable action: paddle movement'
EMB_PYGAME_SHOOTER = '2D shooter game with 2 controllable actions: move and fire'
EMB_PYGAME_HIGHWAY = '2D side-scroller controlling a vehicle that moves in X and Y directions'
EMB_PYGAME_HELICOPTER = '2D helicopter that moves in X and Y directions'
EMB_PYGAME_AIR_HOCKEY = '2D air hockey game with 2 controllable actions: paddle X and Y movement'
EMB_PYGAME_ROCKET = '2D space rocket with 2 controllable actions: main engine and side thruster'
EMB_PYGAME_CHASE_EVADE = '2D chase and evade game with X and Y movement'
EMB_PYGAME_COCONUT_DODGE = '2D minigame with a character that moves laterally'
EMB_PYGAME_POINTMASS = '2D point mass with 2 controllable actions: x and y velocity'
EMB_PYGAME_CARTPOLE = '2D cartpole with 1 controllable joint (the cart), implemented in python'
EMB_PYGAME_REACHER = '2D reacher with 2 controllable joints'
EMB_PYGAME_FORAGING = '2D foraging environment with 2 controllable actions: move in X and Y directions'
EMB_PYGAME_ORCHARD_GARDENER = '2D orchard gardener environment with 2 controllable actions: move in X and Y directions'
EMB_PYGAME_FROZEN_LAKE = '2D frozen lake environment with 2 controllable actions: move in X and Y directions'
EMB_PYGAME_POTION = '2D potion making game with 2 controllable actions: move in X and Y directions'
EMB_PYGAME_WHIRLPOOL = '2D fishing game with a central whirlpool hazard and 2 controllable thruster actions'
EMB_PYGAME_DUNGEON = '2D dungeon exploration with 2 controllable actions: move in X and Y directions'
EMB_ATARI = 'Atari game with 3 controllable actions: radius, theta, and fire'


# Task descriptions
TASKS = OrderedDict({
	'walker-stand': {
		'embodiment': EMB_WALKER,
		'instruction': 'Stand upright and maintain balance',
		'action_dim': 6,
		'max_episode_steps': 500,
	},
	'walker-walk': {
		'embodiment': EMB_WALKER,
		'instruction': 'Walk forward at moderate speed',
		'action_dim': 6,
		'max_episode_steps': 500,
	},
	'walker-run': {
		'embodiment': EMB_WALKER,
		'instruction': 'Run forward as fast as possible',
		'action_dim': 6,
		'max_episode_steps': 500,
	},
	'cheetah-run': {
		'embodiment': EMB_CHEETAH,
		'instruction': 'Run forward quickly',
		'action_dim': 6,
		'max_episode_steps': 500,
	},
	'reacher-easy': {
		'embodiment': EMB_REACHER,
		'instruction': 'Reach for the large red target',
		'action_dim': 2,
		'max_episode_steps': 500,
	},
	'reacher-hard': {
		'embodiment': EMB_REACHER,
		'instruction': 'Reach for the small red target',
		'action_dim': 2,
		'max_episode_steps': 500,
	},
	'acrobot-swingup': {
		'embodiment': EMB_ACROBOT,
		'instruction': 'Swing up to a vertical position',
		'action_dim': 1,
		'max_episode_steps': 500,
	},
	'pendulum-swingup': {
		'embodiment': EMB_PENDULUM,
		'instruction': 'Swing up to an upright position',
		'action_dim': 1,
		'max_episode_steps': 500,
	},
	'cartpole-balance': {
		'embodiment': EMB_CARTPOLE,
		'instruction': 'Balance the pole upright',
		'action_dim': 1,
		'max_episode_steps': 500,
	},
	'cartpole-balance-sparse': {
		'embodiment': EMB_CARTPOLE,
		'instruction': 'Balance the pole upright',
		'action_dim': 1,
		'max_episode_steps': 500,
	},
	'cartpole-swingup': {
		'embodiment': EMB_CARTPOLE,
		'instruction': 'Swing the pole up to a vertical position',
		'action_dim': 1,
		'max_episode_steps': 500,
	},
	'cartpole-swingup-sparse': {
		'embodiment': EMB_CARTPOLE,
		'instruction': 'Swing the pole up to a vertical position',
		'action_dim': 1,
		'max_episode_steps': 500,
	},
	'cup-catch': {
		'embodiment': EMB_CUP_CATCH,
		'instruction': 'Swing the ball into the cup',
		'action_dim': 2,
		'max_episode_steps': 500,
	},
	# Visual variant of cup-catch: identical physics/control as the base task,
	# only the skybox/floor textures differ. Same embodiment, instruction,
	# action_dim, and episode length so lang_emb / act_mask are interchangeable.
	'cup-catch-var1': {
		'embodiment': EMB_CUP_CATCH,
		'instruction': 'Swing the ball into the cup',
		'action_dim': 2,
		'max_episode_steps': 500,
	},
	'finger-spin': {
		'embodiment': EMB_FINGER,
		'instruction': 'Spin the cylindrical object quickly',
		'action_dim': 2,
		'max_episode_steps': 500,
	},
	'finger-turn-easy': {
		'embodiment': EMB_FINGER,
		'instruction': 'Turn the cylindrical object to the target position',
		'action_dim': 2,
		'max_episode_steps': 500,
	},
	# Visual variant of finger-turn-easy: identical physics/control as the base
	# task, only the skybox/floor/cylinder textures differ.
	'finger-turn-easy-var1': {
		'embodiment': EMB_FINGER,
		'instruction': 'Turn the cylindrical object to the target position',
		'action_dim': 2,
		'max_episode_steps': 500,
	},
	'finger-turn-hard': {
		'embodiment': EMB_FINGER,
		'instruction': 'Turn the cylindrical object to the target position',
		'action_dim': 2,
		'max_episode_steps': 500,
	},
	'fish-swim': {
		'embodiment': EMB_FISH,
		'instruction': 'Swim to the target location',
		'action_dim': 5,
		'max_episode_steps': 500,
	},
	'hopper-stand': {
		'embodiment': EMB_HOPPER,
		'instruction': 'Stand upright and balance on its leg',
		'action_dim': 4,
		'max_episode_steps': 500,
	},
	'hopper-hop': {
		'embodiment': EMB_HOPPER,
		'instruction': 'Hop forward as fast as possible',
		'action_dim': 4,
		'max_episode_steps': 500,
	},
	'quadruped-walk': {
		'embodiment': EMB_QUADRUPED,
		'instruction': 'Walk at a steady pace using all 4 legs',
		'action_dim': 12,
		'max_episode_steps': 500,
	},
	'quadruped-run': {
		'embodiment': EMB_QUADRUPED,
		'instruction': 'Run as fast as possible using all 4 legs',
		'action_dim': 12,
		'max_episode_steps': 500,
	},
	'walker-walk-backward': {
		'embodiment': EMB_WALKER,
		'instruction': 'Walk backward',
		'action_dim': 6,
		'max_episode_steps': 500,
	},
	'walker-run-backward': {
		'embodiment': EMB_WALKER,
		'instruction': 'Run backward quickly',
		'action_dim': 6,
		'max_episode_steps': 500,
	},
	'cheetah-run-backward': {
		'embodiment': EMB_CHEETAH,
		'instruction': 'Run backward as fast as possible',
		'action_dim': 6,
		'max_episode_steps': 500,
	},
	'cheetah-run-front': {
		'embodiment': EMB_CHEETAH,
		'instruction': 'Run forward quickly only on its front leg',
		'action_dim': 6,
		'max_episode_steps': 500,
	},
	'cheetah-run-back': {
		'embodiment': EMB_CHEETAH,
		'instruction': 'Run forward only on its hind leg',
		'action_dim': 6,
		'max_episode_steps': 500,
	},
	'cheetah-jump': {
		'embodiment': EMB_CHEETAH,
		'instruction': 'Jump as high as possible',
		'action_dim': 6,
		'max_episode_steps': 500,
	},
	'hopper-hop-backward': {
		'embodiment': EMB_HOPPER,
		'instruction': 'Hop backward on its leg',
		'action_dim': 4,
		'max_episode_steps': 500,
	},
	'reacher-three-easy': {
		'embodiment': EMB_REACHER3,
		'instruction': 'Reach for the large red target',
		'action_dim': 3,
		'max_episode_steps': 500,
	},
	'reacher-three-hard': {
		'embodiment': EMB_REACHER3,
		'instruction': 'Reach for the small red target',
		'action_dim': 3,
		'max_episode_steps': 500,
	},
	'cup-spin': {
		'embodiment': EMB_CUP_CATCH,
		'instruction': 'Spin the ball around the cup',
		'action_dim': 2,
		'max_episode_steps': 500,
	},
	'pendulum-spin': {
		'embodiment': EMB_PENDULUM,
		'instruction': 'Spin the pendulum quickly',
		'action_dim': 1,
		'max_episode_steps': 500,
	},
	'jumper-jump': {
		'embodiment': EMB_JUMPER,
		'instruction': 'Jump as high as possible',
		'action_dim': 2,
		'max_episode_steps': 500,
	},
	'spinner-spin': {
		'embodiment': EMB_SPINNER,
		'instruction': 'Spin forward',
		'action_dim': 5,
		'max_episode_steps': 500,
	},
	'spinner-spin-backward': {
		'embodiment': EMB_SPINNER,
		'instruction': 'Spin backward',
		'action_dim': 5,
		'max_episode_steps': 500,
	},
	'spinner-jump': {
		'embodiment': EMB_SPINNER,
		'instruction': 'Jump as high as possible',
		'action_dim': 5,
		'max_episode_steps': 500,
	},
	'giraffe-run': {
		'embodiment': EMB_GIRAFFE,
		'instruction': 'Run forward at high speed',
		'action_dim': 7,
		'max_episode_steps': 500,
	},
	'mw-assembly': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Pick up the hollow object by its handle and place it over the peg',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-basketball': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Pick up the basketball and place it into the hoop',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-button-press-topdown': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Press the button from above',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-button-press-topdown-wall': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Press the button from above, avoiding the wall',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-button-press': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Press the button from the side',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-button-press-wall': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Press the button from the side, avoiding the wall',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-coffee-button': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Press the button on the coffee machine',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-coffee-pull': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Pull the coffee cup to the target position',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-coffee-push': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Push the coffee cup to the target position',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-dial-turn': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Turn the dial',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-disassemble': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Pick up the hollow object by its handle and remove it from the peg',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-door-open': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Open the door',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-door-close': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Close the door',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-drawer-close': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Close the drawer',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-drawer-open': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Open the drawer',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-faucet-open': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Open the faucet by turning the handle',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-faucet-close': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Close the faucet by turning the handle',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-hammer': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Hammer the peg into the hole',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-handle-press-side': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Press down the handle',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-handle-press': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Press down the handle',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-handle-pull-side': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Pull up the handle',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-handle-pull': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Pull up the handle',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-lever-pull': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Pull the lever up',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-peg-insert-side': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Insert the peg into the hole',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-peg-unplug-side': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Unplug the peg from the hole',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-pick-out-of-hole': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Pick the object out of the hole',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-pick-place': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Pick up the object and move it to the target position',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-pick-place-wall': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Pick up the object and move it to the target position, avoiding the wall',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-plate-slide': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Slide the hockey puck into the goal',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-plate-slide-side': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Slide the hockey puck into the goal',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-plate-slide-back': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Slide the hockey puck into the goal',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-plate-slide-back-side': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Slide the hockey puck into the goal',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-push-back': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Push the object to the target position',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-push': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Push the object to the green target',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-push-wall': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Push the object around the wall and to the target position',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-reach': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Reach for the target position',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-reach-wall': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Reach for the target position, avoiding the wall',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-soccer': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Push the soccer ball into the goal',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-stick-push': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Push the thermo bottle to the target position using the stick',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-stick-pull': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Pull the thermo bottle to the target position using the stick',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-sweep-into': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Place the object at the target position',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-sweep': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Pick up the object and place it at the target position',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-window-open': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Open the window',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-window-close': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Close the window',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-bin-picking': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Pick up the object from the red bin and place it in the blue bin',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-box-close': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Close the box',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-door-lock': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Lock the door',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-door-unlock': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Unlock the door',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'mw-hand-insert': {
		'embodiment': EMB_METAWORLD,
		'instruction': 'Move the object to the target position',
		'action_dim': 4,
		'max_episode_steps': 100,
	},
	'ms-ant-walk': {
		'embodiment': EMB_MANISKILL_ANT,
		'instruction': 'Walk along the X-axis at 0.5 m/s',
		'action_dim': 8,
		'max_episode_steps': 500,
	},
	'ms-ant-run': {
		'embodiment': EMB_MANISKILL_ANT,
		'instruction': 'Run along the X-axis at 4 m/s',
		'action_dim': 8,
		'max_episode_steps': 500,
	},
	'ms-cartpole-balance': {
		'embodiment': EMB_MANISKILL_CARTPOLE,
		'instruction': 'Balance the pole upright',
		'action_dim': 1,
		'max_episode_steps': 500,
	},
	'ms-cartpole-swingup': {
		'embodiment': EMB_MANISKILL_CARTPOLE,
		'instruction': 'Swing the pole up to a vertical position',
		'action_dim': 1,
		'max_episode_steps': 500,
	},
	'ms-cartpole-balance-sparse': {
		'embodiment': EMB_MANISKILL_CARTPOLE,
		'instruction': 'Balance the pole such that it is upright',
		'action_dim': 1,
		'max_episode_steps': 500,
	},
	'ms-cartpole-swingup-sparse': {
		'embodiment': EMB_MANISKILL_CARTPOLE,
		'instruction': 'Swing up the pole to an upright position',
		'action_dim': 1,
		'max_episode_steps': 500,
	},
	'ms-hopper-stand': {
		'embodiment': EMB_MANISKILL_HOPPER,
		'instruction': 'Stand upright',
		'action_dim': 4,
		'max_episode_steps': 250,
		'discount_factor': 0.99,
	},
	'ms-hopper-hop': {
		'embodiment': EMB_MANISKILL_HOPPER,
		'instruction': 'Move forward quickly by hopping',
		'action_dim': 4,
		'max_episode_steps': 250,
		'discount_factor': 0.99,
	},
	'ms-pick-cube': {
		'embodiment': EMB_MANISKILL_FRANKA_POS,
		'instruction': 'Pick up the red cube and move it to the green target position',
		'action_dim': 4,
		'max_episode_steps': 25,
	},
	'ms-pick-cube-eepose': {
		'embodiment': EMB_MANISKILL_FRANKA_POSE,
		'instruction': 'Pick up the red cube and move it to the green target position',
		'action_dim': 7,
		'max_episode_steps': 25,
	},
	'ms-pick-cube-so': {
		'embodiment': EMB_MANISKILL_SO100_POS,
		'instruction': 'Pick up the red cube and move it to the green target position',
		'action_dim': 6,
		'max_episode_steps': 25,
	},
	'ms-poke-cube': {
		'embodiment': EMB_MANISKILL_FRANKA_POSE,
		'instruction': 'Push the red cube to the target using the blue peg',
		'action_dim': 7,
		'max_episode_steps': 25,
	},
	'ms-push-cube': {
		'embodiment': EMB_MANISKILL_FRANKA_POSE,
		'instruction': 'Push the small cube to the target',
		'action_dim': 7,
		'max_episode_steps': 25,
	},
	'ms-pull-cube': {
		'embodiment': EMB_MANISKILL_FRANKA_POSE,
		'instruction': 'Pull the small cube to the target using the gripper',
		'action_dim': 7,
		'max_episode_steps': 25,
	},
	'ms-pull-cube-tool': {
		'embodiment': EMB_MANISKILL_FRANKA_POSE,
		'instruction': 'Pull the small cube to the target using the tool',
		'action_dim': 7,
		'max_episode_steps': 50,
	},
	'ms-stack-cube': {
		'embodiment': EMB_MANISKILL_FRANKA_POSE,
		'instruction': 'Stack the red cube on top of the green cube',
		'action_dim': 7,
		'max_episode_steps': 25,
	},
	'ms-place-sphere': {
		'embodiment': EMB_MANISKILL_FRANKA_POS,
		'instruction': 'Pick up the ball and put it in the tray',
		'action_dim': 4,
		'max_episode_steps': 25,
	},
	'ms-lift-peg': {
		'embodiment': EMB_MANISKILL_FRANKA_POSE,
		'instruction': 'Lift peg and place it upright on the table',
		'action_dim': 7,
		'max_episode_steps': 25,
	},
	'ms-pick-apple': {
		'embodiment': EMB_MANISKILL_FRANKA_POSE,
		'instruction': 'Pick up the apple and move it to the target position',
		'action_dim': 7,
		'max_episode_steps': 25,
	},
	'ms-pick-banana': {
		'embodiment': EMB_MANISKILL_FRANKA_POSE,
		'instruction': 'Pick up the banana and move it to the target position',
		'action_dim': 7,
		'max_episode_steps': 25,
	},
	'ms-pick-baseball': {
		'embodiment': EMB_MANISKILL_FRANKA_POSE,
		'instruction': 'Pick up the baseball and move it to the target position',
		'action_dim': 7,
		'max_episode_steps': 25,
	},
	'ms-pick-can': {
		'embodiment': EMB_MANISKILL_FRANKA_POSE,
		'instruction': 'Pick up the can and move it to the target position',
		'action_dim': 7,
		'max_episode_steps': 25,
	},
	'ms-pick-hammer': {
		'embodiment': EMB_MANISKILL_FRANKA_POSE,
		'instruction': 'Pick up the hammer and move it to the target position',
		'action_dim': 7,
		'max_episode_steps': 25,
	},
	'ms-pick-fork': {
		'embodiment': EMB_MANISKILL_FRANKA_POSE,
		'instruction': 'Pick up the fork and move it to the target position',
		'action_dim': 7,
		'max_episode_steps': 25,
	},
	'ms-pick-knife': {
		'embodiment': EMB_MANISKILL_FRANKA_POSE,
		'instruction': 'Pick up the knife and move it to the target position',
		'action_dim': 7,
		'max_episode_steps': 25,
	},
	'ms-pick-mug': {
		'embodiment': EMB_MANISKILL_FRANKA_POSE,
		'instruction': 'Pick up the mug and move it to the target position',
		'action_dim': 7,
		'max_episode_steps': 25,
	},
	'ms-pick-orange': {
		'embodiment': EMB_MANISKILL_FRANKA_POSE,
		'instruction': 'Pick up the orange and move it to the target position',
		'action_dim': 7,
		'max_episode_steps': 25,
	},
	'ms-pick-screwdriver': {
		'embodiment': EMB_MANISKILL_FRANKA_POSE,
		'instruction': 'Pick up the screwdriver and move it to the target position',
		'action_dim': 7,
		'max_episode_steps': 25,
	},
	'ms-pick-sponge': {
		'embodiment': EMB_MANISKILL_FRANKA_POSE,
		'instruction': 'Pick up the sponge and move it to the target position',
		'action_dim': 7,
		'max_episode_steps': 25,
	},
	'ms-pick-spoon': {
		'embodiment': EMB_MANISKILL_FRANKA_POSE,
		'instruction': 'Pick up the spoon and move it to the target position',
		'action_dim': 7,
		'max_episode_steps': 25,
	},
	'ms-pick-tennis-ball': {
		'embodiment': EMB_MANISKILL_FRANKA_POSE,
		'instruction': 'Pick up the tennis ball and move it to the target position',
		'action_dim': 7,
		'max_episode_steps': 25,
	},
	'ms-pick-cube-xarm6': {
		'embodiment': EMB_MANISKILL_XARM6_POS,
		'instruction': 'Pick up the red cube and move it to the green target position',
		'action_dim': 4,
		'max_episode_steps': 25,
	},
	'ms-reach': {
		'embodiment': EMB_MANISKILL_FRANKA_POS,
		'instruction': 'Reach target position with end-effector',
		'action_dim': 4,
		'max_episode_steps': 25,
	},
	'ms-reach-eepose': {
		'embodiment': EMB_MANISKILL_FRANKA_POSE,
		'instruction': 'Reach the target position with the end-effector',
		'action_dim': 7,
		'max_episode_steps': 25,
	},
	'ms-reach-xarm6': {
		'embodiment': EMB_MANISKILL_XARM6_POS,
		'instruction': 'Reach the target position with the gripper',
		'action_dim': 4,
		'max_episode_steps': 25,
	},
	'ms-anymal-reach': {
		'embodiment': EMB_MANISKILL_ANYMAL,
		'instruction': 'Reach target',
		'action_dim': 12,
		'max_episode_steps': 100,
	},
	'mujoco-ant': {
		'embodiment': EMB_MUJOCO_ANT,
		'instruction': 'Run as fast as possible',
		'action_dim': 8,
		'max_episode_steps': 500,
		'discount_factor': 0.99,
	},
	'mujoco-inverted-pendulum': {
		'embodiment': EMB_MUJOCO_INVERTED_PENDULUM,
		'instruction': 'Balance the inverted pendulum upright',
		'action_dim': 1,
		'max_episode_steps': 500,
		'discount_factor': 0.99,
	},
	'mujoco-reacher': {
		'embodiment': EMB_MUJOCO_REACHER,
		'instruction': 'Reach the target position with the end-effector',
		'action_dim': 2,
		'max_episode_steps': 50,
	},
	'mujoco-halfcheetah': {
		'embodiment': EMB_MUJOCO_HALFCHEETAH,
		'instruction': 'Run forward as fast as possible',
		'action_dim': 6,
		'max_episode_steps': 1000,
		'discount_factor': 0.99,
	},
	'mujoco-hopper': {
		'embodiment': EMB_MUJOCO_HOPPER,
		'instruction': 'Hopper hops forward at high speed',
		'action_dim': 3,
		'max_episode_steps': 500,
		'discount_factor': 0.99,
	},
	'mujoco-walker': {
		'embodiment': EMB_MUJOCO_WALKER,
		'instruction': 'Walk forward at high speed',
		'action_dim': 6,
		'max_episode_steps': 500,
		'discount_factor': 0.99,
	},
	'bipedal-walker-flat': {
		'embodiment': EMB_BIPEDAL_WALKER,
		'instruction': 'Walk forward quickly on flat terrain',
		'action_dim': 4,
		'max_episode_steps': 250,
		'discount_factor': 0.99,
	},
	'bipedal-walker-uneven': {
		'embodiment': EMB_BIPEDAL_WALKER,
		'instruction': 'Walk forward as fast as possible on uneven terrain',
		'action_dim': 4,
		'max_episode_steps': 250,
		'discount_factor': 0.99,
	},
	'bipedal-walker-rugged': {
		'embodiment': EMB_BIPEDAL_WALKER,
		'instruction': 'Walk forward at high speed on rugged terrain',
		'action_dim': 4,
		'max_episode_steps': 250,
		'discount_factor': 0.99,
	},
	'bipedal-walker-hills': {
		'embodiment': EMB_BIPEDAL_WALKER,
		'instruction': 'Traverse the hills as fast as possible',
		'action_dim': 4,
		'max_episode_steps': 250,
		'discount_factor': 0.99,
	},
	'bipedal-walker-obstacles': {
		'embodiment': EMB_BIPEDAL_WALKER,
		'instruction': 'Traverse the various obstacles quickly',
		'action_dim': 4,
		'max_episode_steps': 250,
		'discount_factor': 0.99,
	},
	'lunarlander-land': {
		'embodiment': EMB_LUNARLANDER,
		'instruction': 'Land safely on the landing pad',
		'action_dim': 2,
		'max_episode_steps': 250,
		'discount_factor': 0.99,
	},
	'lunarlander-hover': {
		'embodiment': EMB_LUNARLANDER,
		'instruction': 'Hover safely around the target position',
		'action_dim': 2,
		'max_episode_steps': 250,
		'discount_factor': 0.99,
	},
	'lunarlander-takeoff': {
		'embodiment': EMB_LUNARLANDER,
		'instruction': 'Take off and fly to the target position',
		'action_dim': 2,
		'max_episode_steps': 250,
		'discount_factor': 0.99,
	},
	'rd-open-slide': {
		'embodiment': EMB_ROBODESK_FRANKA_POSE,
		'instruction': 'Open the sliding door',
		'action_dim': 5,
		'max_episode_steps': 100,
	},
	'rd-open-drawer': {
		'embodiment': EMB_ROBODESK_FRANKA_POSE,
		'instruction': 'Open the drawer',
		'action_dim': 5,
		'max_episode_steps': 100,
	},
	'rd-flat-block-in-bin': {
		'embodiment': EMB_ROBODESK_FRANKA_POSE,
		'instruction': 'Put the flat block into the bin',
		'action_dim': 5,
		'max_episode_steps': 100,
	},
	'rd-push-red': {
		'embodiment': EMB_ROBODESK_FRANKA_POSE,
		'instruction': 'Push the red button',
		'action_dim': 5,
		'max_episode_steps': 100,
	},
	'rd-push-green': {
		'embodiment': EMB_ROBODESK_FRANKA_POSE,
		'instruction': 'Please push the green button',
		'action_dim': 5,
		'max_episode_steps': 100,
	},
	'rd-push-blue': {
		'embodiment': EMB_ROBODESK_FRANKA_POSE,
		'instruction': 'Push the blue button',
		'action_dim': 5,
		'max_episode_steps': 100,
	},
	'og-ant': {
		'embodiment': EMB_OGBENCH_ANT,
		'instruction': 'Run at high speed',
		'action_dim': 8,
		'max_episode_steps': 1000,
		'discount_factor': 0.99,
	},
	'og-antball': {
		'embodiment': EMB_OGBENCH_ANT,
		'instruction': 'Push the soccer ball to the goal location',
		'action_dim': 8,
		'max_episode_steps': 200,
	},
	'og-point-arena': {
		'embodiment': EMB_OGBENCH_POINTMASS,
		'instruction': 'Move the point mass to the red goal location',
		'action_dim': 2,
		'max_episode_steps': 100,
	},
	'og-point-maze': {
		'embodiment': EMB_OGBENCH_POINTMASS,
		'instruction': 'Navigate through a maze to reach the red goal',
		'action_dim': 2,
		'max_episode_steps': 100,
	},
	'og-point-bottleneck': {
		'embodiment': EMB_OGBENCH_POINTMASS,
		'instruction': 'Navigate the point mass to the red goal, avoiding obstacles',
		'action_dim': 2,
		'max_episode_steps': 100,
	},
	'og-point-circle': {
		'embodiment': EMB_OGBENCH_POINTMASS,
		'instruction': 'Navigate the point mass around a circular maze to reach a red goal',
		'action_dim': 2,
		'max_episode_steps': 100,
	},
	'og-point-spiral': {
		'embodiment': EMB_OGBENCH_POINTMASS,
		'instruction': 'Move to the red goal location by navigating through a spiral maze',
		'action_dim': 2,
		'max_episode_steps': 100,
	},
	'og-ant-arena': {
		'embodiment': EMB_OGBENCH_ANT,
		'instruction': 'Move the ant to the red goal',
		'action_dim': 8,
		'max_episode_steps': 200,
	},
	'og-ant-maze': {
		'embodiment': EMB_OGBENCH_ANT,
		'instruction': 'Navigate through a maze to reach the red goal',
		'action_dim': 8,
		'max_episode_steps': 200,
	},
	'og-ant-bottleneck': {
		'embodiment': EMB_OGBENCH_ANT,
		'instruction': 'Navigate to the red goal location, avoiding obstacles',
		'action_dim': 8,
		'max_episode_steps': 200,
	},
	'og-ant-circle': {
		'embodiment': EMB_OGBENCH_ANT,
		'instruction': 'Reach the red goal location by navigating through the circular maze',
		'action_dim': 8,
		'max_episode_steps': 200,
	},
	'og-ant-spiral': {
		'embodiment': EMB_OGBENCH_ANT,
		'instruction': 'The ant must traverse a spiral maze to reach a goal marked in red',
		'action_dim': 8,
		'max_episode_steps': 200,
	},
	'pygame-cowboy': {
		'embodiment': EMB_PYGAME_SCROLLER,
		'instruction': 'Jump to avoid obstacles',
		'action_dim': 1,
		'max_episode_steps': 500,
	},
	'pygame-coinrun': {
		'embodiment': EMB_PYGAME_SCROLLER,
		'instruction': 'Collect as many coins as possible',
		'action_dim': 1,
		'max_episode_steps': 500,
	},
	'pygame-spaceship': {
		'embodiment': EMB_PYGAME_SPACESHIP,
		'instruction': 'Collect coins while avoiding asteroids',
		'action_dim': 2,
		'max_episode_steps': 500,
	},
	'pygame-pong': {
		'embodiment': EMB_PYGAME_PONG,
		'instruction': 'Score against an AI opponent',
		'action_dim': 1,
		'max_episode_steps': 500,
	},
	'pygame-bird-attack': {
		'embodiment': EMB_PYGAME_SHOOTER,
		'instruction': 'Eliminate all enemies in a space-invaders style game',
		'action_dim': 2,
		'max_episode_steps': 500,
	},
	'pygame-highway': {
		'embodiment': EMB_PYGAME_HIGHWAY,
		'instruction': 'Avoid collision with other vehicles',
		'action_dim': 2,
		'max_episode_steps': 500,
	},
	'pygame-landing': {
		'embodiment': EMB_PYGAME_HELICOPTER,
		'instruction': 'Land the helicopter on the moving ship',
		'action_dim': 2,
		'max_episode_steps': 200,
	},
	'pygame-air-hockey': {
		'embodiment': EMB_PYGAME_AIR_HOCKEY,
		'instruction': 'Score points against the opponent',
		'action_dim': 2,
		'max_episode_steps': 500,
	},
	'pygame-rocket-collect': {
		'embodiment': EMB_PYGAME_ROCKET,
		'instruction': 'Collect coins while avoiding asteroids',
		'action_dim': 2,
		'max_episode_steps': 500,
	},
	'pygame-chase-evade': {
		'embodiment': EMB_PYGAME_CHASE_EVADE,
		'instruction': 'Take turns being the chaser or evader',
		'action_dim': 2,
		'max_episode_steps': 500,
	},
	'pygame-coconut-dodge': {
		'embodiment': EMB_PYGAME_COCONUT_DODGE,
		'instruction': 'Dodge falling coconuts',
		'action_dim': 1,
		'max_episode_steps': 500,
	},
	'pygame-cartpole-balance': {
		'embodiment': EMB_PYGAME_CARTPOLE,
		'instruction': 'Balance the pole',
		'action_dim': 1,
		'max_episode_steps': 500,
	},
	'pygame-cartpole-balance-sparse': {
		'embodiment': EMB_PYGAME_CARTPOLE,
		'instruction': 'Balance the pole',
		'action_dim': 1,
		'max_episode_steps': 500,
	},
	'pygame-cartpole-swingup': {
		'embodiment': EMB_PYGAME_CARTPOLE,
		'instruction': 'Swing up the pole and balance it',
		'action_dim': 1,
		'max_episode_steps': 500,
	},
	'pygame-cartpole-swingup-sparse': {
		'embodiment': EMB_PYGAME_CARTPOLE,
		'instruction': 'Swing up the pole and balance it',
		'action_dim': 1,
		'max_episode_steps': 500,
	},
	'pygame-cartpole-tremor': {
		'embodiment': EMB_PYGAME_CARTPOLE,
		'instruction': 'Balance the pole amidst disturbances',
		'action_dim': 1,
		'max_episode_steps': 500,
	},
	'pygame-point-maze-var1': {
		'embodiment': EMB_PYGAME_POINTMASS,
		'instruction': 'Navigate to the target location',
		'action_dim': 2,
		'max_episode_steps': 200,
	},
	'pygame-point-maze-var2': {
		'embodiment': EMB_PYGAME_POINTMASS,
		'instruction': 'Navigate to the green target',
		'action_dim': 2,
		'max_episode_steps': 200,
	},
	'pygame-point-maze-var3': {
		'embodiment': EMB_PYGAME_POINTMASS,
		'instruction': 'Go to the green target location',
		'action_dim': 2,
		'max_episode_steps': 200,
	},
	'atari-alien': {
		'embodiment': EMB_ATARI,
		'instruction': 'Escape the maze and eliminate aliens',
		'action_dim': 3,
		'max_episode_steps': 1000,
		'discount_factor': 0.99,
	},
	'atari-assault': {
		'embodiment': EMB_ATARI,
		'instruction': 'Destroy enemy bases and dodge projectiles',
		'action_dim': 3,
		'max_episode_steps': 1000,
		'discount_factor': 0.99,
	},
	'atari-asterix': {
		'embodiment': EMB_ATARI,
		'instruction': 'Collect food and avoid Roman guards',
		'action_dim': 3,
		'max_episode_steps': 1000,
		'discount_factor': 0.99,
	},
	'atari-atlantis': {
		'embodiment': EMB_ATARI,
		'instruction': 'Defend the city from invading aircraft',
		'action_dim': 3,
		'max_episode_steps': 1000,
		'discount_factor': 0.99,
	},
	'atari-bank-heist': {
		'embodiment': EMB_ATARI,
		'instruction': 'Rob banks and escape the police',
		'action_dim': 3,
		'max_episode_steps': 1000,
		'discount_factor': 0.99,
	},
	'atari-battle-zone': {
		'embodiment': EMB_ATARI,
		'instruction': 'Destroy enemy tanks from your tank',
		'action_dim': 3,
		'max_episode_steps': 1000,
		'discount_factor': 0.99,
	},
	'atari-beamrider': {
		'embodiment': EMB_ATARI,
		'instruction': 'Shoot alien ships and avoid obstacles',
		'action_dim': 3,
		'max_episode_steps': 1000,
		'discount_factor': 0.99,
	},
	'atari-boxing': {
		'embodiment': EMB_ATARI,
		'instruction': 'Punch the opponent and score points in a boxing match',
		'action_dim': 3,
		'max_episode_steps': 1000,
		'discount_factor': 0.99,
	},
	'atari-chopper-command': {
		'embodiment': EMB_ATARI,
		'instruction': 'Protect trucks by shooting enemy aircraft',
		'action_dim': 3,
		'max_episode_steps': 1000,
		'discount_factor': 0.99,
	},
	'atari-crazy-climber': {
		'embodiment': EMB_ATARI,
		'instruction': 'Climb the skyscraper and avoid falling hazards',
		'action_dim': 3,
		'max_episode_steps': 1000,
		'discount_factor': 0.99,
	},
	'atari-double-dunk': {
		'embodiment': EMB_ATARI,
		'instruction': 'Score baskets in half-court basketball',
		'action_dim': 3,
		'max_episode_steps': 1000,
		'discount_factor': 0.99,
	},
	'atari-gopher': {
		'embodiment': EMB_ATARI,
		'instruction': 'Protect crops by blocking the gopher',
		'action_dim': 3,
		'max_episode_steps': 1000,
		'discount_factor': 0.99,
	},
	'atari-ice-hockey': {
		'embodiment': EMB_ATARI,
		'instruction': 'Score goals in a game of ice hockey',
		'action_dim': 3,
		'max_episode_steps': 1000,
		'discount_factor': 0.99,
	},
	'atari-jamesbond': {
		'embodiment': EMB_ATARI,
		'instruction': 'Rescue hostages and destroy enemy bases',
		'action_dim': 3,
		'max_episode_steps': 1000,
		'discount_factor': 0.99,
	},
	'atari-kangaroo': {
		'embodiment': EMB_ATARI,
		'instruction': 'Rescue your child while avoiding monkeys',
		'action_dim': 3,
		'max_episode_steps': 1000,
		'discount_factor': 0.99,
	},
	'atari-krull': {
		'embodiment': EMB_ATARI,
		'instruction': 'Defeat enemies and rescue the princess',
		'action_dim': 3,
		'max_episode_steps': 1000,
		'discount_factor': 0.99,
	},
	'atari-ms-pacman': {
		'embodiment': EMB_ATARI,
		'instruction': 'Eat pellets and avoid ghosts in a game of Ms. Pacman',
		'action_dim': 3,
		'max_episode_steps': 1000,
		'discount_factor': 0.99,
	},
	'atari-name-this-game': {
		'embodiment': EMB_ATARI,
		'instruction': 'Collect treasure and avoid sea creatures',
		'action_dim': 3,
		'max_episode_steps': 1000,
		'discount_factor': 0.99,
	},
	'atari-phoenix': {
		'embodiment': EMB_ATARI,
		'instruction': 'Shoot alien birds and destroy the mothership',
		'action_dim': 3,
		'max_episode_steps': 1000,
		'discount_factor': 0.99,
	},
	'atari-pong': {
		'embodiment': EMB_ATARI,
		'instruction': 'Score against an opponent in a game of Pong',
		'action_dim': 3,
		'max_episode_steps': 1000,
		'discount_factor': 0.99,
	},
	'atari-road-runner': {
		'embodiment': EMB_ATARI,
		'instruction': 'Collect bird seed and avoid the coyote',
		'action_dim': 3,
		'max_episode_steps': 1000,
		'discount_factor': 0.99,
	},
	'atari-robotank': {
		'embodiment': EMB_ATARI,
		'instruction': 'Eliminate enemy tanks in battle',
		'action_dim': 3,
		'max_episode_steps': 1000,
		'discount_factor': 0.99,
	},
	'atari-seaquest': {
		'embodiment': EMB_ATARI,
		'instruction': 'Rescue divers and fight underwater enemies',
		'action_dim': 3,
		'max_episode_steps': 1000,
		'discount_factor': 0.99,
	},
	'atari-space-invaders': {
		'embodiment': EMB_ATARI,
		'instruction': 'Shoot alien waves before they land in a game of Space Invaders',
		'action_dim': 3,
		'max_episode_steps': 1000,
		'discount_factor': 0.99,
	},
	'atari-tutankham': {
		'embodiment': EMB_ATARI,
		'instruction': 'Collect treasures and avoid tomb enemies',
		'action_dim': 3,
		'max_episode_steps': 1000,
		'discount_factor': 0.99,
	},
	'atari-upndown': {
		'embodiment': EMB_ATARI,
		'instruction': 'Collect flags and avoid crashing cars',
		'action_dim': 3,
		'max_episode_steps': 1000,
		'discount_factor': 0.99,
	},
	'atari-yars-revenge': {
		'embodiment': EMB_ATARI,
		'instruction': 'Destroy the Qotile while avoiding its weapon',
		'action_dim': 3,
		'max_episode_steps': 1000,
		'discount_factor': 0.99,
	},
	# below are reserved for testing
	'cartpole-balance-two-poles-sparse': {
		'embodiment': EMB_CARTPOLE,
		'instruction': 'Balance the pole upright',
		'action_dim': 1,
		'max_episode_steps': 500,
	},
	'cartpole-balance-long-sparse': {
		'embodiment': EMB_CARTPOLE,
		'instruction': 'Balance the pole upright',
		'action_dim': 1,
		'max_episode_steps': 500,
	},
	'walker-stand-incline': {
		'embodiment': EMB_WALKER,
		'instruction': 'Stand upright and maintain balance',
		'action_dim': 6,
		'max_episode_steps': 500,
	},
	'walker-walk-incline': {
		'embodiment': EMB_WALKER,
		'instruction': 'Walk forward at moderate speed',
		'action_dim': 6,
		'max_episode_steps': 500,
	},
	'walker-run-incline': {
		'embodiment': EMB_WALKER,
		'instruction': 'Run forward as fast as possible',
		'action_dim': 6,
		'max_episode_steps': 500,
	},
	'walker-arabesque': {
		'embodiment': EMB_WALKER,
		'instruction': 'Stand on one leg with the other leg extended backward',
		'action_dim': 6,
		'max_episode_steps': 500,
	},
	'walker-lie-down': {
		'embodiment': EMB_WALKER,
		'instruction': 'Lie down on the ground',
		'action_dim': 6,
		'max_episode_steps': 500,
	},
	'walker-legs-up': {
		'embodiment': EMB_WALKER,
		'instruction': 'Lie down with legs up',
		'action_dim': 6,
		'max_episode_steps': 500,
	},
	'walker-headstand': {
		'embodiment': EMB_WALKER,
		'instruction': 'Stand on your head',
		'action_dim': 6,
		'max_episode_steps': 500,
	},
	'spinner-spin-four': {
		'embodiment': EMB_SPINNER_FOUR_ARMS,
		'instruction': 'Spin forward',
		'action_dim': 4,
		'max_episode_steps': 500,
	},
	'spinner-spin-backward-four': {
		'embodiment': EMB_SPINNER_FOUR_ARMS,
		'instruction': 'Spin backward',
		'action_dim': 4,
		'max_episode_steps': 500,
	},
	'spinner-jump-four': {
		'embodiment': EMB_SPINNER_FOUR_ARMS,
		'instruction': 'Jump as high as possible',
		'action_dim': 4,
		'max_episode_steps': 500,
	},
	'ms-push-apple': {
		'embodiment': EMB_MANISKILL_FRANKA_POSE,
		'instruction': 'Push the apple to the target',
		'action_dim': 7,
		'max_episode_steps': 25,
	},
	'ms-push-pear': {
		'embodiment': EMB_MANISKILL_FRANKA_POSE,
		'instruction': 'Push the pear to the target',
		'action_dim': 7,
		'max_episode_steps': 25,
	},
	'ms-push-rubiks-cube': {
		'embodiment': EMB_MANISKILL_FRANKA_POSE,
		'instruction': 'Push the rubiks cube to the target',
		'action_dim': 7,
		'max_episode_steps': 25,
	},
	'ms-push-can': {
		'embodiment': EMB_MANISKILL_FRANKA_POSE,
		'instruction': 'Push the can to the target',
		'action_dim': 7,
		'max_episode_steps': 25,
	},
	'ms-push-sponge': {
		'embodiment': EMB_MANISKILL_FRANKA_POSE,
		'instruction': 'Push the sponge to the target',
		'action_dim': 7,
		'max_episode_steps': 25,
	},
	'ms-push-banana': {
		'embodiment': EMB_MANISKILL_FRANKA_POSE,
		'instruction': 'Push the banana to the target',
		'action_dim': 7,
		'max_episode_steps': 25,
	},
	'ms-push-screwdriver': {
		'embodiment': EMB_MANISKILL_FRANKA_POSE,
		'instruction': 'Push the screwdriver to the target',
		'action_dim': 7,
		'max_episode_steps': 25,
	},
	'ms-pick-rubiks-cube': {
		'embodiment': EMB_MANISKILL_FRANKA_POSE,
		'instruction': 'Pick up the rubiks cube and move it to the target position',
		'action_dim': 7,
		'max_episode_steps': 25,
	},
	'ms-pick-cup': {
		'embodiment': EMB_MANISKILL_FRANKA_POSE,
		'instruction': 'Pick up the cup and move it to the target position',
		'action_dim': 7,
		'max_episode_steps': 25,
	},
	'ms-pick-golf-ball': {
		'embodiment': EMB_MANISKILL_FRANKA_POSE,
		'instruction': 'Pick up the golf ball and move it to the target position',
		'action_dim': 7,
		'max_episode_steps': 25,
	},
	'ms-pick-soccer-ball': {
		'embodiment': EMB_MANISKILL_FRANKA_POSE,
		'instruction': 'Pick up the soccer ball and move it to the target position',
		'action_dim': 7,
		'max_episode_steps': 25,
	},
	'og-point-var1': {
		'embodiment': EMB_OGBENCH_POINTMASS,
		'instruction': 'Navigate the point mass to the goal, avoiding obstacles',
		'action_dim': 2,
		'max_episode_steps': 100,
	},
	'og-point-var2': {
		'embodiment': EMB_OGBENCH_POINTMASS,
		'instruction': 'Navigate the point mass around a maze to reach a red goal',
		'action_dim': 2,
		'max_episode_steps': 100,
	},
	'pygame-point-maze-var4': {
		'embodiment': EMB_PYGAME_POINTMASS,
		'instruction': 'Navigate to the target location',
		'action_dim': 2,
		'max_episode_steps': 200,
	},
	'pygame-point-maze-var5': {
		'embodiment': EMB_PYGAME_POINTMASS,
		'instruction': 'Go to the target location',
		'action_dim': 2,
		'max_episode_steps': 200,
	},
	'pygame-point-maze-var6': {
		'embodiment': EMB_PYGAME_POINTMASS,
		'instruction': 'Navigate to the target',
		'action_dim': 2,
		'max_episode_steps': 200,
	},
	'pygame-point-maze-var7': {
		'embodiment': EMB_PYGAME_POINTMASS,
		'instruction': 'Go to the green target',
		'action_dim': 2,
		'max_episode_steps': 200,
	},
	'pygame-point-maze-var8': {
		'embodiment': EMB_PYGAME_POINTMASS,
		'instruction': 'Go to the target',
		'action_dim': 2,
		'max_episode_steps': 200,
	},
	'pygame-reacher-easy': {
		'embodiment': EMB_PYGAME_REACHER,
		'instruction': 'Reach for the red target',
		'action_dim': 2,
		'max_episode_steps': 200,
	},
	'pygame-reacher-hard': {
		'embodiment': EMB_PYGAME_REACHER,
		'instruction': 'Reach for the small red target',
		'action_dim': 2,
		'max_episode_steps': 200,
	},
	'pygame-reacher-var1': {
		'embodiment': EMB_PYGAME_REACHER,
		'instruction': 'Reach for the red target',
		'action_dim': 2,
		'max_episode_steps': 200,
	},
	'pygame-reacher-var2': {
		'embodiment': EMB_PYGAME_REACHER,
		'instruction': 'Reach for the red target',
		'action_dim': 2,
		'max_episode_steps': 200,
	},
	# MiniArcade tasks
	'pygame-foraging': {
		'embodiment': EMB_PYGAME_FORAGING,
		'instruction': 'Collect logs and bring them back to the home base',
		'action_dim': 2,
		'max_episode_steps': 500,
	},
	'pygame-orchard-gardener': {
		'embodiment': EMB_PYGAME_ORCHARD_GARDENER,
		'instruction': 'Pick fruits from trees return them to home',
		'action_dim': 2,
		'max_episode_steps': 500,
	},
	'pygame-frozen-lake-6x6': {
		'embodiment': EMB_PYGAME_FROZEN_LAKE,
		'instruction': 'Pick up flags on a 6x6 grid while avoiding holes in the ice',
		'action_dim': 2,
		'max_episode_steps': 500,
	},
	'pygame-frozen-lake-5x5': {
		'embodiment': EMB_PYGAME_FROZEN_LAKE,
		'instruction': 'Pick up flags on a 5x5 grid while avoiding holes in the ice',
		'action_dim': 2,
		'max_episode_steps': 500,
	},
	'pygame-frozen-lake-4x4': {
		'embodiment': EMB_PYGAME_FROZEN_LAKE,
		'instruction': 'Pick up flags on a 4x4 grid while avoiding holes in the ice',
		'action_dim': 2,
		'max_episode_steps': 500,
	},
	'pygame-potion-recipe1': {
		'embodiment': EMB_PYGAME_POTION,
		'instruction': 'Collect ingredients and brew a potion according to the recipe',
		'action_dim': 2,
		'max_episode_steps': 250,
	},
	'pygame-potion-recipe2': {
		'embodiment': EMB_PYGAME_POTION,
		'instruction': 'Collect ingredients and brew a potion according to the recipe',
		'action_dim': 2,
		'max_episode_steps': 250,
	},
	'pygame-potion-recipe3': {
		'embodiment': EMB_PYGAME_POTION,
		'instruction': 'Collect ingredients and brew a potion according to the recipe',
		'action_dim': 2,
		'max_episode_steps': 250,
	},
	'pygame-potion-recipe4': {
		'embodiment': EMB_PYGAME_POTION,
		'instruction': 'Collect ingredients and brew a potion according to the recipe',
		'action_dim': 2,
		'max_episode_steps': 250,
	},
	'pygame-potion-recipe5': {
		'embodiment': EMB_PYGAME_POTION,
		'instruction': 'Collect ingredients and brew a potion according to the recipe',
		'action_dim': 2,
		'max_episode_steps': 250,
	},
	'pygame-potion-recipe6': {
		'embodiment': EMB_PYGAME_POTION,
		'instruction': 'Collect ingredients and brew a potion according to the recipe',
		'action_dim': 2,
		'max_episode_steps': 250,
	},
	'pygame-potion-recipe7': {
		'embodiment': EMB_PYGAME_POTION,
		'instruction': 'Collect ingredients and brew a potion according to the recipe',
		'action_dim': 2,
		'max_episode_steps': 250,
	},
	'pygame-potion-recipe8': {
		'embodiment': EMB_PYGAME_POTION,
		'instruction': 'Collect ingredients and brew a potion according to the recipe',
		'action_dim': 2,
		'max_episode_steps': 250,
	},
	'pygame-potion-recipe9': {
		'embodiment': EMB_PYGAME_POTION,
		'instruction': 'Collect ingredients and brew a potion according to the recipe',
		'action_dim': 2,
		'max_episode_steps': 250,
	},
	'pygame-potion-recipe10': {
		'embodiment': EMB_PYGAME_POTION,
		'instruction': 'Collect ingredients and brew a potion according to the recipe',
		'action_dim': 2,
		'max_episode_steps': 250,
	},
	'pygame-dungeon-explorer1': {
		'embodiment': EMB_PYGAME_DUNGEON,
		'instruction': 'Navigate the dungeon to find a treasure',
		'action_dim': 2,
		'max_episode_steps': 250,
	},
	'pygame-dungeon-explorer2': {
		'embodiment': EMB_PYGAME_DUNGEON,
		'instruction': 'Navigate the dungeon to find a treasure',
		'action_dim': 2,
		'max_episode_steps': 250,
	},
	'pygame-dungeon-explorer3': {
		'embodiment': EMB_PYGAME_DUNGEON,
		'instruction': 'Navigate the dungeon to find a treasure',
		'action_dim': 2,
		'max_episode_steps': 250,
	},
	'pygame-dungeon-explorer4': {
		'embodiment': EMB_PYGAME_DUNGEON,
		'instruction': 'Navigate the dungeon to find a treasure',
		'action_dim': 2,
		'max_episode_steps': 250,
	},
	'pygame-dungeon-explorer5': {
		'embodiment': EMB_PYGAME_DUNGEON,
		'instruction': 'Navigate the dungeon to find a treasure',
		'action_dim': 2,
		'max_episode_steps': 250,
	},
	'pygame-dungeon-explorer6': {
		'embodiment': EMB_PYGAME_DUNGEON,
		'instruction': 'Navigate the dungeon to find a treasure',
		'action_dim': 2,
		'max_episode_steps': 250,
	},
	'pygame-dungeon-explorer7': {
		'embodiment': EMB_PYGAME_DUNGEON,
		'instruction': 'Navigate the dungeon to find a treasure',
		'action_dim': 2,
		'max_episode_steps': 250,
	},
	'pygame-dungeon-explorer8': {
		'embodiment': EMB_PYGAME_DUNGEON,
		'instruction': 'Navigate the dungeon to find a treasure',
		'action_dim': 2,
		'max_episode_steps': 250,
	},
	'pygame-dungeon-explorer9': {
		'embodiment': EMB_PYGAME_DUNGEON,
		'instruction': 'Navigate the dungeon to find a treasure',
		'action_dim': 2,
		'max_episode_steps': 250,
	},
	'pygame-dungeon-explorer10': {
		'embodiment': EMB_PYGAME_DUNGEON,
		'instruction': 'Navigate the dungeon to find a treasure',
		'action_dim': 2,
		'max_episode_steps': 250,
	},
	'pygame-firefly': {
		'embodiment': EMB_PYGAME_POTION,
		'instruction': 'Catch fireflies',
		'action_dim': 2,
		'max_episode_steps': 500,
	},
	'pygame-plankton': {
		'embodiment': EMB_PYGAME_POTION,
		'instruction': 'Catch plankton',
		'action_dim': 2,
		'max_episode_steps': 500,
	},
	'pygame-whirlpool': {
		'embodiment': EMB_PYGAME_WHIRLPOOL,
		'instruction': 'Catch the fish while avoiding the whirlpool at the center',
		'action_dim': 2,
		'max_episode_steps': 500,
	},
})


tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
text_model.eval()
text_model.to('cuda')

print(f'Found {len(TASKS)} tasks. Creating text embeddings...')
for task_name, task_info in TASKS.items():
	text = f'{task_info["embodiment"]}. {task_info["instruction"]}.'

	inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
	inputs = {k: v.to('cuda') for k, v in inputs.items()}

	with torch.no_grad():
		text_embeddings = text_model(**inputs).last_hidden_state

	task_info['text_embedding'] = text_embeddings.mean(dim=1).squeeze().cpu().numpy().tolist()

FILEPATH = './tasks.json'
assert os.path.exists(os.path.dirname(FILEPATH)), f'Directory does not exist: {os.path.dirname(FILEPATH)}'
save_dict_as_json(TASKS, FILEPATH)
print(f'Saved task embeddings of dim {len(TASKS["walker-stand"]["text_embedding"])}.')
