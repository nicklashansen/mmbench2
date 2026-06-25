"""Custom Finger tasks (visual variants only — dynamics unchanged)."""

from dm_control.rl import control
from dm_control.suite import common
from dm_control.suite import finger

# Reuse the var1 dark-skybox / muted-floor asset overrides defined alongside
# ball_in_cup so all dmcontrol *_var1 variants share the same look.
from envs.tasks.dmcontrol.ball_in_cup import _var1_assets

_DEFAULT_TIME_LIMIT = 20  # (seconds)
_CONTROL_TIMESTEP = .02   # (seconds)
_EASY_TARGET_SIZE = 0.07  # mirrors stock dm_control.suite.finger


@finger.SUITE.add('custom')
def turn_easy_var1(time_limit=_DEFAULT_TIME_LIMIT, random=None,
                   environment_kwargs=None):
  """Stock Finger turn_easy dynamics with a dark-gray skybox + muted floor.

  Visual variant: identical physics, control, and reward to upstream
  `finger.turn_easy` — only the skybox and floor textures differ.
  """
  physics = finger.Physics.from_xml_string(
      common.read_model('finger.xml'), _var1_assets())
  task = finger.Turn(target_radius=_EASY_TARGET_SIZE, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)
