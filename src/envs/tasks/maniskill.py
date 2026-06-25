from typing import Any, Dict

import sapien
import torch

from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.envs.tasks.control.cartpole import CartpoleBalanceEnv, CartpoleSwingUpEnv


@register_env("PickCubexArm6-v1")
class PickCubeXarm6Env(PickCubeEnv):
    """
    A variant of the PickCube environment for the xArm6 robot.
    """

    def __init__(self, *args, robot_uids=None, **kwargs):
        super().__init__(*args, robot_uids="xarm6_robotiq", **kwargs)


@register_env("Reach-v1", max_episode_steps=50)
class ReachEnv(PickCubeEnv):

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = (
                torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            goal_xyz[:, 0] += self.cube_spawn_center[0]
            goal_xyz[:, 1] += self.cube_spawn_center[1]
            goal_xyz[:, 2] = torch.rand((b)) * self.max_goal_height
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            tcp_pose=self.agent.tcp_pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                tcp_to_goal_pos=self.goal_site.pose.p - self.agent.tcp_pose.p,
            )
        return obs

    def evaluate(self):
        robot_at_goal = (
            torch.linalg.norm(self.goal_site.pose.p - self.agent.tcp_pose.p, axis=1)
            <= self.goal_thresh
        )
        return {
            "success": robot_at_goal,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.agent.tcp_pose.p, axis=1
        )
        return 5*(1 + info["success"].float() - torch.tanh(5*tcp_to_goal_dist))


@register_env("ReachxArm6-v1", max_episode_steps=50)
class ReachXarm6Env(ReachEnv):
    """
    A variant of the Reach environment for the xArm6 robot.
    """

    def __init__(self, *args, robot_uids=None, **kwargs):
        super().__init__(*args, robot_uids="xarm6_robotiq", **kwargs)


@register_env("MS-CartpoleBalanceSparse-v1", max_episode_steps=1000)
class CartpoleBalanceSparseEnv(CartpoleBalanceEnv):

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        upright = (self.pole_angle_cosine + 1) / 2
        return (upright > 0.85).float()


@register_env("MS-CartpoleSwingUpSparse-v1", max_episode_steps=1000)
class CartpoleSwingUpSparseEnv(CartpoleSwingUpEnv):

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        upright = (self.pole_angle_cosine + 1) / 2
        return (upright > 0.85).float()


@register_env("PushYCB-v1", max_episode_steps=50, asset_download_ids=["ycb"])
class PushYCBEnv(PushCubeEnv):

    cube_half_size = 0.04
    model_id = "037_scissors"

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self._objs = []
        model_ids = [self.model_id]
        for i, model_id in enumerate(model_ids):
            builder = actors.get_actor_builder(
                self.scene,
                id=f"ycb:{model_id}",
            )
            builder.initial_pose = sapien.Pose(p=[0, 0, 0])
            builder.set_scene_idxs([i])
            self._objs.append(builder.build(name=f"{model_id}-{i}"))
            self.remove_from_state_dict_registry(self._objs[-1])
        self.obj = Actor.merge(self._objs, name="cube")
        self.add_to_state_dict_registry(self.obj)

        # we specify add_collisions=False as we only use this as a visual for videos and do not want it to affect the actual physics
        # we finally specify the body_type to be "kinematic" so that the object stays in place
        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, 1e-3]),
        )
