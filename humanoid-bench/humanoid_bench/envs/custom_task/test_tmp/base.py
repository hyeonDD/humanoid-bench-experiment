import os

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium.spaces import Box
from dm_control.utils import rewards

from humanoid_bench.tasks import Task


# Height of head above which stand reward is 1.
_STAND_HEIGHT = 1.65
_CRAWL_HEIGHT = 0.8

# Horizontal speeds above which move reward is 1.
# TODO WALK SPEED 선택
_WALK_SPEED = 1 # ClimbingUpwards, Hurdle
"""
_WALK_SPEED = 0.5 # Pole
"""
_RUN_SPEED = 5

class CustomTask(Task):
    # TODO g1 robot 추가할지 말지 선택
    qpos0_robot = {
        "h1": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0",
        "h1hand": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
        "h1touch": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",        
        # ClimbingUPwards, Hurdle 의 g1 bot
        # "g1": "0 0 0.75 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1.57 0 0 0 0 0 0 0 0 0 0 0 1.57 0 0 0 0 0 0 0"
    }

    # TODO Hurdle 의 move speed는 RUN SPEED (5)임
    _move_speed = _WALK_SPEED
    """
    _move_speed = _RUN_SPEED # Hurdle
    """

    # TODO Hurdle의 카메라는 cam_hurdle이고 나머지는 cam_default임
    """
    camera_name = "cam_default"
    camera_name = "cam_hurdle"
    """
    success_bar = 700 # 이건 3개다 똑같음

    # TODO htarget값 선택
    htarget_low = np.array([-2.5, -2.5, 0.3])
    htarget_high = np.array([50.0, 2.5, 1.8])
    # ClimbingUpwards, Hurdle의 htraget값
    """
    htarget_low = np.array([-1.0, -1.0, 0.8])
    htarget_high = np.array([1000.0, 1.0, 2.0])
    """

    # TODO ClimbingUpwards, Hurdle의 __init__ 에는 G1bot일때 _CRAWL_HEIGHT를 0.6
    def __init__(self, robot=None, env=None, **kwargs):
        super().__init__(robot, env, **kwargs)
        if robot.__class__.__name__ == "G1":
            global _STAND_HEIGHT
            _STAND_HEIGHT = 1.28
    """
    def __init__(self, robot=None, env=None, **kwargs):
        super().__init__(robot, env, **kwargs)
        if robot.__class__.__name__ == "G1":
            global _STAND_HEIGHT, _CRAWL_HEIGHT
            _STAND_HEIGHT = 1.28
            _CRAWL_HEIGHT = 0.6
    """

    # observation_space 3개다 동일
    @property
    def observation_space(self):
        return Box(
            low=-np.inf, high=np.inf, shape=(self.robot.dof * 2 - 1,), dtype=np.float64
        )

    def get_reward(self):
        # TODO stand보상을 구할때 standing값을 ClimbingUpwards의 stading으로?
        # Pole과 Hurdle은 stading을 구하는값이 동일
        standing = rewards.tolerance(
            self.robot.head_height(),
            bounds=(_STAND_HEIGHT, float("inf")),
            margin=_STAND_HEIGHT / 4,
        )
        """
        # ClimbingUpwards임
        standing = rewards.tolerance(
            self.robot.head_height() - self.robot.left_foot_height(),
            bounds=(1.2, float("inf")),
            margin=0.45,
        ) * rewards.tolerance(
            self.robot.head_height() - self.robot.right_foot_height(),
            bounds=(1.2, float("inf")),
            margin=0.45,
        )
        """

        # TODO uprgiht의 bounds 선택하기
        upright = rewards.tolerance(
            self.robot.torso_upright(),
            # ClimbingUpwards는 0.5, Hurdle은 0.8
            bounds=(0.9, float("inf")),
            sigmoid="linear",
            margin=1.9,
            value_at_margin=0,
        )

        # stand_reward 3개 동일
        stand_reward = standing * upright

        # small_control 3개 동일
        small_control = rewards.tolerance( 
            self.robot.actuator_forces(),
            margin=10,
            value_at_margin=0,
            sigmoid="quadratic",
        ).mean()
        small_control = (4 + small_control) / 5

        com_velocity = self.robot.center_of_mass_velocity()[0]
        move = rewards.tolerance(
            com_velocity,
            bounds=(self._move_speed, float("inf")),
            # TODO move speed 선택
            # Pole 0.5 ClimbingUpwards 1, Hurdle 5
            margin=self._move_speed,
            value_at_margin=0,
            sigmoid="linear",
        )
        move = (5 * move + 1) / 6

        # TODO collision 계산을 3개task 한번에 할 수 있도록 구현해야함
        all_geoms_id = self._env.named.data.geom_xpos.axes.row.names

        collision_discount = 1
        for pair in self._env.data.contact.geom:
            if (
                any(["pole_r" in all_geoms_id[p_val] for p_val in pair])
                and 0 not in pair
            ):  #
                collision_discount = 0.1
                break

        # TODO reward 3개task 한번에 할 수 있도록 구현해야함
        reward = (
            0.5 * (small_control * stand_reward) + 0.5 * move
        ) * collision_discount
        return reward, {
            "stand_reward": stand_reward,
            "small_control": small_control,
            "move": move,
            "standing": standing,
            "upright": upright,
            "collision_discount": collision_discount,
        }

    # TODO 종료 3개 task에 적절하게 변경 필요
    def get_terminated(self):
        # Pole
        """
        return self._env.data.qpos[2] < 0.5, {}
        # Hurdle
        return self._env.data.qpos[2] < 0.2, {}
        # ClimbingUPwards
        return self.robot.torso_upright() < 0.1, {}
        """