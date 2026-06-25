import os

from dm_control.rl import control
from dm_control.suite import common
from dm_control.suite import cheetah
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import io as resources

_TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dmcontrol')

_GIRAFFE_HEAD_HEIGHT = 0.5

SUITE = containers.TaggedTasks()


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return resources.GetResource(os.path.join(_TASKS_DIR, 'giraffe.xml')), common.ASSETS


@SUITE.add('custom')
def run(time_limit=cheetah._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Run task."""
    physics = cheetah.Physics.from_xml_string(*get_model_and_assets())
    task = Giraffe(goal='run', move_speed=cheetah._RUN_SPEED, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               **environment_kwargs)


class Giraffe(cheetah.Cheetah):
    """Giraffe tasks."""
    
    def __init__(self, goal='run', move_speed=0, random=None):
        super().__init__(random)
        self._goal = goal
        self._move_speed = move_speed

    def _run_reward(self, physics):
        head_height = physics.named.data.xpos['head', 'z']
        head_up_reward = rewards.tolerance(head_height,
                            bounds=(_GIRAFFE_HEAD_HEIGHT, float('inf')),
                            margin=_GIRAFFE_HEAD_HEIGHT/2)                           
        horizontal_speed_reward = rewards.tolerance(physics.speed(),
                            bounds=(self._move_speed, float('inf')),
                            margin=self._move_speed,
                            value_at_margin=0,
                            sigmoid='linear')
        return head_up_reward * (3*horizontal_speed_reward + 1) / 4

    def get_reward(self, physics):
        if self._goal == 'run':
            return self._run_reward(physics)
        else:
            raise NotImplementedError(f'Goal {self._goal} is not implemented.')
