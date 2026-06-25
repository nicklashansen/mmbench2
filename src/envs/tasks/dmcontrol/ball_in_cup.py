import collections
import os

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import ball_in_cup
from dm_control.suite import common
from dm_control.utils import rewards
from dm_control.utils import io as resources
import numpy as np

_TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dmcontrol')

_DIST_TARGET = 0.5
_TARGET_SPEED = 6.

_DEFAULT_TIME_LIMIT = 20  # (seconds)
_CONTROL_TIMESTEP = .02   # (seconds)


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return resources.GetResource(os.path.join(_TASKS_DIR, 'ball_in_cup.xml')), common.ASSETS


@ball_in_cup.SUITE.add('custom')
def spin(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Ball-in-Cup Spin task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = CustomBallInCup(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


# ---------------------------------------------------------------------------
# Visual variant assets (var1): dark-gray skybox + muted floor.
#
# The dm_control common XML includes (`./common/skybox.xml` and
# `./common/materials.xml`) are normally resolved via `common.ASSETS`. For the
# *_var1 visual variant we pass a copy of that dict with these two entries
# overridden — the host XML and every other asset (visual.xml lighting,
# self/effector/target colors, etc.) stay byte-identical to stock so dynamics
# are unchanged.
# ---------------------------------------------------------------------------

_VAR1_DARK_SKYBOX_XML = b"""<mujoco>
  <asset>
    <texture name="skybox" type="skybox" builtin="flat" rgb1=".2 .2 .2" rgb2=".2 .2 .2"
             width="800" height="800"/>
  </asset>
</mujoco>"""

# Floor checker tiles are a purple gradient (#524466 → #6b5c85). The
# `decoration` material (used for the cylindrical anchors that the finger /
# cup are mounted on) is recolored to a complementary purple #6b578a.
# All other materials (self / effector / target / eye / site) keep their
# stock colors so the cup, ball, finger, and target geometry look identical.
_VAR1_MUTED_MATERIALS_XML = b"""<mujoco>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".32 .27 .40" rgb2=".42 .36 .52"
             width="300" height="300" mark="edge" markrgb=".42 .36 .52"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
    <material name="self" rgba=".7 .5 .3 1"/>
    <material name="self_default" rgba=".7 .5 .3 1"/>
    <material name="self_highlight" rgba="0 .5 .3 1"/>
    <material name="effector" rgba=".7 .4 .2 1"/>
    <material name="effector_default" rgba=".7 .4 .2 1"/>
    <material name="effector_highlight" rgba="0 .5 .3 1"/>
    <material name="decoration" rgba=".42 .34 .54 1"/>
    <material name="eye" rgba="0 .2 1 1"/>
    <material name="target" rgba=".6 .3 .3 1"/>
    <material name="target_default" rgba=".6 .3 .3 1"/>
    <material name="target_highlight" rgba=".6 .3 .3 .4"/>
    <material name="site" rgba=".5 .5 .5 .3"/>
  </asset>
</mujoco>"""


def _var1_assets():
  """dm_control common.ASSETS with skybox and materials swapped to var1."""
  assets = dict(common.ASSETS)
  assets['./common/skybox.xml'] = _VAR1_DARK_SKYBOX_XML
  assets['./common/materials.xml'] = _VAR1_MUTED_MATERIALS_XML
  return assets


@ball_in_cup.SUITE.add('custom')
def catch_var1(time_limit=_DEFAULT_TIME_LIMIT, random=None,
               environment_kwargs=None):
  """Stock Ball-in-Cup Catch dynamics with a dark-gray skybox + muted floor.

  Visual variant: identical physics, control, and reward to upstream
  `ball_in_cup.catch` — only the skybox and floor textures differ.
  """
  physics = ball_in_cup.Physics.from_xml_string(
      common.read_model('ball_in_cup.xml'), _var1_assets())
  task = ball_in_cup.BallInCup(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


class Physics(mujoco.Physics):
  """Physics with additional features for the Ball-in-Cup domain."""

  def ball_to_target(self):
    """Returns the vector from the ball to the target."""
    target = self.named.data.site_xpos['target', ['x', 'z']]
    ball = self.named.data.xpos['ball', ['x', 'z']]
    return target - ball

  def in_target(self):
    """Returns 1 if the ball is in the target, 0 otherwise."""
    ball_to_target = abs(self.ball_to_target())
    target_size = self.named.model.site_size['target', [0, 2]]
    ball_size = self.named.model.geom_size['ball', 0]
    return float(all(ball_to_target < target_size - ball_size))


class CustomBallInCup(ball_in_cup.BallInCup):
  """Custom Ball-in-Cup tasks."""

  def initialize_episode(self, physics):
    # Find a collision-free random initial position of the ball.
    penetrating = True
    valid_pos = False
    init_out_of_target = self.random.uniform() < 0.1
    while penetrating or not valid_pos:
      physics.named.data.qpos['ball_x'] = self.random.uniform(-.2, .2)
      physics.named.data.qpos['ball_z'] = self.random.uniform(.2, .5)
      physics.after_reset()
      penetrating = physics.data.ncon > 0
      valid_pos = bool(physics.in_target()) or init_out_of_target
    base.Task.initialize_episode(self, physics)

  def get_observation(self, physics):
    """Returns an observation of the state."""
    obs = collections.OrderedDict()
    obs['position'] = physics.position()
    obs['velocity'] = physics.velocity()
    return obs

  def get_reward(self, physics):
    dist = np.linalg.norm(physics.ball_to_target())
    ball_vel_x = abs(physics.named.data.qvel['ball_x'])
    ball_vel_z = abs(physics.named.data.qvel['ball_z'])
    ball_vel = np.linalg.norm([ball_vel_x, ball_vel_z])

    # reward: spin around target (maximize distance to target + ball velocity)
    dist_reward = rewards.tolerance(dist,
                                    bounds=(_DIST_TARGET, float('inf')),
                                    margin=_DIST_TARGET/2,
                                    value_at_margin=0.5,
                                    sigmoid='linear')
    not_in_target = 1 - physics.in_target()
    vel_reward = rewards.tolerance(ball_vel,
                                   bounds=(_TARGET_SPEED, float('inf')),
                                   margin=_TARGET_SPEED/2,
                                   value_at_margin=0.5,
                                   sigmoid='linear')
    spin_reward = not_in_target * (dist_reward + 2*vel_reward) / 3
    return spin_reward
