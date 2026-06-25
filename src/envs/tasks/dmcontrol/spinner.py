import os

from dm_control.rl import control
from dm_control.suite import common
from dm_control.suite import cheetah
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import io as resources

_TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dmcontrol')

_SPINNER_SPIN_SPEED = 8.0
_SPINNER_JUMP_HEIGHT = 1.5

SUITE = containers.TaggedTasks()


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return resources.GetResource(os.path.join(_TASKS_DIR, 'spinner.xml')), common.ASSETS


def get_four_arms_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return resources.GetResource(os.path.join(_TASKS_DIR, 'spinner_four_arms.xml')), common.ASSETS


@SUITE.add('custom')
def spin(time_limit=cheetah._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Spin task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Spinner(goal='spin', move_speed=0.8*cheetah._RUN_SPEED, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               **environment_kwargs)


@SUITE.add('custom')
def spin_backward(time_limit=cheetah._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Spin Backward task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Spinner(goal='spin-backward', move_speed=0.8*cheetah._RUN_SPEED, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               **environment_kwargs)


@SUITE.add('custom')
def jump(time_limit=cheetah._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Jump task."""
    physics = cheetah.Physics.from_xml_string(*get_model_and_assets())
    task = Spinner(goal='jump', move_speed=0, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               **environment_kwargs)


@SUITE.add('custom')
def spin_four(time_limit=cheetah._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Spin task but with 4 arms."""
    physics = Physics.from_xml_string(*get_four_arms_model_and_assets())
    task = Spinner(goal='spin', move_speed=0.8*cheetah._RUN_SPEED, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               **environment_kwargs)

@SUITE.add('custom')
def spin_backward_four(time_limit=cheetah._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Spin Backward task but with 4 arms."""
    physics = Physics.from_xml_string(*get_four_arms_model_and_assets())
    task = Spinner(goal='spin-backward', move_speed=0.8*cheetah._RUN_SPEED, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               **environment_kwargs)


@SUITE.add('custom')
def jump_four(time_limit=cheetah._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Jump task but with 4 arms."""
    physics = cheetah.Physics.from_xml_string(*get_four_arms_model_and_assets())
    task = Spinner(goal='jump', move_speed=0, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               **environment_kwargs)


class Physics(cheetah.Physics):
    """Physics simulation with additional features for the Spinner domain."""

    def angmomentum(self):
        """Returns the angular momentum of torso of the Spinner about Y axis."""
        return self.named.data.subtree_angmom['torso'][1]


class Spinner(cheetah.Cheetah):
    """Spinner tasks."""
    
    def __init__(self, goal='run', move_speed=0, random=None):
        super().__init__(random)
        self._goal = goal
        self._move_speed = move_speed
    
    def _spin_reward(self, physics, forward=True):
        angmom_reward = rewards.tolerance(
                            (1. if forward else -1.) * physics.angmomentum(),
                            bounds=(_SPINNER_SPIN_SPEED, float('inf')),
                            margin=_SPINNER_SPIN_SPEED,
                            value_at_margin=0,
                            sigmoid='linear')
        horizontal_speed_reward = rewards.tolerance(
                            (1. if forward else -1.) * physics.speed(),
                            bounds=(self._move_speed, float('inf')),
                            margin=self._move_speed,
                            value_at_margin=0,
                            sigmoid='linear')
        spin_reward = (2*angmom_reward + 3*horizontal_speed_reward) / 5
        return spin_reward
    
    def _jump_reward(self, physics):
        torso_height = physics.named.data.xpos['torso', 'z']
        height_reward = rewards.tolerance(torso_height,
                            bounds=(_SPINNER_JUMP_HEIGHT, float('inf')),
                            margin=_SPINNER_JUMP_HEIGHT/2)
        return height_reward

    def get_reward(self, physics):
        if self._goal == 'spin':
            return self._spin_reward(physics, forward=True)
        elif self._goal == 'spin-backward':
            return self._spin_reward(physics, forward=False)
        elif self._goal == 'jump':
            return self._jump_reward(physics)
        else:
            raise NotImplementedError(f'Goal {self._goal} is not implemented.')
