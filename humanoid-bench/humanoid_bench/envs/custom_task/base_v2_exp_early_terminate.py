import os

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium.spaces import Box
from dm_control.utils import rewards

from humanoid_bench.tasks import Task


# Height of head above which stand reward is 1.
_STAND_HEIGHT = 1.65

# Horizontal speeds above which move reward is 1.
# pole=0.5, clime(walk)=1, hurdle=5
_WALK_SPEED = 1


class BaseEarlyTerminate(Task):
    qpos0_robot = {
        "h1": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0",
        # "h1hand": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
        # "h1touch": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
    }
    _move_speed = _WALK_SPEED
    success_bar = 700

    htarget_low = np.array([-2.5, -2.5, 0.3])
    htarget_high = np.array([50.0, 2.5, 1.8])

    def __init__(self, robot=None, env=None, **kwargs):
        super().__init__(robot, env, **kwargs)
        # if robot.__class__.__name__ == "G1":
        #     global _STAND_HEIGHT
        #     _STAND_HEIGHT = 1.28

    @property
    def observation_space(self):
        return Box(
            low=-np.inf, high=np.inf, shape=(self.robot.dof * 2 - 1,), dtype=np.float64
        )

    def get_reward(self):
        # hurdle : _move_speed = _RUN_SPEED

        # hurdle, pole에서는 bounds=_STAND_HEIGHT, margin=_STAND_HEIGHT / 4
        standing = rewards.tolerance(
            self.robot.head_height() - self.robot.left_foot_height(),
            bounds=(1.2, float("inf")),
            margin=0.45,
        ) * rewards.tolerance(
            self.robot.head_height() - self.robot.right_foot_height(),
            bounds=(1.2, float("inf")),
            margin=0.45,
        )

        # climb에서는 bounds=0.5, pole=0.9
        upright = rewards.tolerance(
            self.robot.torso_upright(),
            bounds=(0.5, float("inf")),
            sigmoid="linear",
            margin=1.9,
            value_at_margin=0,
        )
        stand_reward = standing * upright
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
            margin=self._move_speed,
            value_at_margin=0,
            sigmoid="linear",
        )
        move = (5 * move + 1) / 6

        all_geoms_id = self._env.named.data.geom_xpos.axes.row.names

        # any에 충돌로 보상을 줄일 물체 이름은 collision_으로 시작
        collision_discount = 1
        for pair in self._env.data.contact.geom:
            if (
                any(["collision" in all_geoms_id[p_val] for p_val in pair])
                and 0 not in pair
            ):  #
                collision_discount = 0.1
                break

        # 비중도 고려
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

    def get_terminated(self):
        # pole : return self._env.data.qpos[2] < 0.5, {}
        # hurdle : return self._env.data.qpos[2] < 0.2, {}
        return self.robot.torso_upright() < 0.3, {}
