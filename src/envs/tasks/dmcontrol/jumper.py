import os

from dm_control.rl import control
from dm_control.suite import common
from dm_control.suite import cheetah
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import io as resources

_TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dmcontrol')

_JUMPER_JUMP_HEIGHT = 1.5

SUITE = containers.TaggedTasks()


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return resources.GetResource(os.path.join(_TASKS_DIR, 'jumper.xml')), common.ASSETS


@SUITE.add('custom')
def jump_forward(time_limit=cheetah._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Jump Forward task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Jumper(goal='jump-forward', move_speed=0.8*cheetah._RUN_SPEED, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               **environment_kwargs)


@SUITE.add('custom')
def jump_backward(time_limit=cheetah._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Jump Backward task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Jumper(goal='jump-backward', move_speed=0.8*cheetah._RUN_SPEED, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               **environment_kwargs)


@SUITE.add('custom')
def jump(time_limit=cheetah._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Jump task."""
    physics = cheetah.Physics.from_xml_string(*get_model_and_assets())
    task = Jumper(goal='jump', move_speed=0, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               **environment_kwargs)


class Physics(cheetah.Physics):
    """Physics simulation with additional features for the Jumper domain."""

    def angmomentum(self):
        """Returns the angular momentum of torso of the Jumper about Y axis."""
        return self.named.data.subtree_angmom['torso'][1]


class Jumper(cheetah.Cheetah):
    """Jumper tasks."""
    
    def __init__(self, goal='run', move_speed=0, random=None):
        super().__init__(random)
        self._goal = goal
        self._move_speed = move_speed
    
    def _jump_horizontal_reward(self, physics, forward=True):
        horizontal_speed_reward = rewards.tolerance(
                            (1. if forward else -1.) * physics.speed(),
                            bounds=(self._move_speed, float('inf')),
                            margin=self._move_speed,
                            value_at_margin=0,
                            sigmoid='linear')
        height_reward = self._height_reward(physics)
        return (3*horizontal_speed_reward + 2*height_reward) / 5
    
    def _height_reward(self, physics):
        torso_height = physics.named.data.xpos['torso', 'z']
        height_reward = rewards.tolerance(torso_height,
                            bounds=(_JUMPER_JUMP_HEIGHT, float('inf')),
                            margin=_JUMPER_JUMP_HEIGHT/2)
        return height_reward

    def get_reward(self, physics):
        if self._goal == 'jump-forward':
            return self._jump_horizontal_reward(physics, forward=True)
        elif self._goal == 'spin-backward':
            return self._jump_horizontal_reward(physics, forward=False)
        elif self._goal == 'jump':
            return self._height_reward(physics)
        else:
            raise NotImplementedError(f'Goal {self._goal} is not implemented.')
