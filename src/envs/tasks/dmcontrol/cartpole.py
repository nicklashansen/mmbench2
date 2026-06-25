import os

from dm_control.rl import control
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.suite import cartpole
from dm_control.suite.cartpole import Balance, _make_model, Physics
from dm_control.utils import io as resources

_TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dmcontrol')

_DEFAULT_TIME_LIMIT = 10
SUITE = containers.TaggedTasks()


def get_model_and_assets(num_poles=1):
  """Returns a tuple containing the model XML string and a dict of assets."""
  return _make_model(num_poles), common.ASSETS


def get_long_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return resources.GetResource(os.path.join(_TASKS_DIR, 'cartpole.xml')), common.ASSETS


@cartpole.SUITE.add('custom')
def balance_two_poles(time_limit=_DEFAULT_TIME_LIMIT, random=None,
              environment_kwargs=None):
  """Returns the Cartpole Balance task with two poles."""
  physics = Physics.from_xml_string(*get_model_and_assets(num_poles=2))
  task = Balance(swing_up=False, sparse=False, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@cartpole.SUITE.add('custom')
def balance_two_poles_sparse(time_limit=_DEFAULT_TIME_LIMIT, random=None,
              environment_kwargs=None):
  """Returns the Cartpole Balance Sparse task with two poles."""
  physics = Physics.from_xml_string(*get_model_and_assets(num_poles=2))
  task = Balance(swing_up=False, sparse=True, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@cartpole.SUITE.add('custom')
def balance_long(time_limit=_DEFAULT_TIME_LIMIT, random=None,
              environment_kwargs=None):
  """Returns the Cartpole Balance task with a long pole."""
  physics = Physics.from_xml_string(*get_long_model_and_assets())
  task = Balance(swing_up=False, sparse=False, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@cartpole.SUITE.add('custom')
def balance_long_sparse(time_limit=_DEFAULT_TIME_LIMIT, random=None,
              environment_kwargs=None):
  """Returns the Cartpole Balance Sparse task with a long pole."""
  physics = Physics.from_xml_string(*get_long_model_and_assets())
  task = Balance(swing_up=False, sparse=True, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)
