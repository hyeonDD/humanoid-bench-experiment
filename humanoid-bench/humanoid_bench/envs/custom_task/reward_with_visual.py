import os
import numpy as np
import mujoco
import torch
from ultralytics import YOLO
from gymnasium.spaces import Box
from dm_control.utils import rewards
import cv2

from humanoid_bench.tasks import Task

# YOLOv8 분류 모델 불러오기
model = YOLO("../../../classifier/task_classifier_model/yolov8n-cls.pt").model
model.load_state_dict(
    torch.load(
        "../../../classifier/task_classifier_model/yolov8_model_epoch_10_acc_97.66.pt"
    )
)

# Height of head above which stand reward is 1.
_STAND_HEIGHT = 1.65

# Horizontal speeds above which move reward is 1.
_HURDLE_SPEED = 5
_POLE_SPEED = 0.5
_SLIDE_SPEED = 1


def predict_image_label(model, img_tensor):
    """
    모델과 이미지 텐서를 입력받아 예측된 레이블을 반환하는 함수.
    Parameters:
    model: 훈련된 YOLOv8 분류 모델
    img_tensor: 전처리된 이미지 텐서 (1, C, H, W 형태)

    Returns:
    predicted_label: 예측된 레이블
    """
    model.eval()  # 모델을 평가 모드로 설정
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)  # 예측된 클래스 인덱스 가져오기
        predicted_label = predicted.item()  # 텐서를 정수로 변환
    return predicted_label


class BaseWithTask(Task):
    qpos0_robot = {
        "h1": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0",
        "h1hand": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
        "h1touch": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
        "g1": "0 0 0.75 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1.57 0 0 0 0 0 0 0 0 0 0 0 1.57 0 0 0 0 0 0 0",
    }
    _move_speed = _SLIDE_SPEED
    success_bar = 700

    htarget_low = np.array([-2.5, -2.5, 0.3])
    htarget_high = np.array([50.0, 2.5, 1.8])

    def __init__(self, robot=None, env=None, **kwargs):
        super().__init__(robot, env, **kwargs)
        if robot.__class__.__name__ == "G1":
            global _STAND_HEIGHT
            _STAND_HEIGHT = 1.28

        self.task_label = -1

    @property
    def observation_space(self):
        return Box(
            low=-np.inf, high=np.inf, shape=(self.robot.dof * 2 - 1,), dtype=np.float64
        )

    def render(self):
        """
        Mujoco 환경에서 이미지를 가져와서 YOLO 모델에 맞는 텐서 형식으로 변환하는 함수
        """
        # Mujoco에서 이미지 가져오기 (224,224 사이즈로 변환)
        image = self._env.mujoco_renderer.render(
            self._env.render_mode, self._env.camera_id, self._env.camera_name
        )

        # 이미지 크기 조정 (224x224)
        image_resized = cv2.resize(image, (224, 224))

        # 이미지 텐서로 변환 (C, H, W 형식으로 변환 및 차원 추가)
        img_tensor = torch.tensor(image_resized.transpose(2, 0, 1)).unsqueeze(0).float()
        return img_tensor

    def predict_task(self):
        """
        이미지 예측 함수
        """
        img_tensor = self.render()
        return predict_image_label(model, img_tensor)

    def get_reward(self):
        """
        YOLO 모델을 이용해 예측된 task에 따라 보상 함수를 적용하는 함수.
        """
        self.task_label = self.predict_task()
        # task에 따른 보상함수 적용
        if self.task_label == 0:  # Hurdle
            self._move_speed = _HURDLE_SPEED
            reward, info = self.get_hurdle_reward()
        elif self.task_label == 1:  # pole
            self._move_speed = _POLE_SPEED
            self.htarget_low = np.array([-1.0, -1.0, 0.8])
            self.htarget_high = np.array([1000.0, 1.0, 2.0])
            reward, info = self.get_pole_reward()
        elif self.task_label == 2:  # ramp/slide
            reward, info = self.get_slide_reward()
        else:
            reward, info = 0, {}

        return reward, info

    def get_terminated(self):
        """
        YOLO 모델을 이용해 예측된 task에 따라 종료 조건을 적용하는 함수.
        """
        # task에 따른 종료조건 적용
        if self.task_label == 0:  # Hurdle
            return self._env.data.qpos[2] < 0.2, {}
        elif self.task_label == 1:  # pole
            return self._env.data.qpos[2] < 0.5, {}
        elif self.task_label == 2:  # ramp/slide
            return self.robot.torso_upright() < 0.1, {}
        else:
            return False, {}

    def compute_rewards(self, task_lb, bounds_val, speed):
        """
        Hurdle, Pole, Ramp(Slide) 모두 공통적인 보상 계산식을 가지고있음
        공통적인 보상 계산을 처리하는 함수
        """
        if task_lb == 2:
            standing = rewards.tolerance(
                self.robot.head_height() - self.robot.left_foot_height(),
                bounds=(1.2, float("inf")),
                margin=0.45,
            ) * rewards.tolerance(
                self.robot.head_height() - self.robot.right_foot_height(),
                bounds=(1.2, float("inf")),
                margin=0.45,
            )
        else:
            standing = rewards.tolerance(
                self.robot.head_height(),
                bounds=(_STAND_HEIGHT, float("inf")),
                margin=_STAND_HEIGHT / 4,
            )

        upright = rewards.tolerance(
            self.robot.torso_upright(),
            bounds=(bounds_val, float("inf")),
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
            bounds=(speed, float("inf")),
            margin=speed,
            value_at_margin=0,
            sigmoid="linear",
        )
        move = (5 * move + 1) / 6

        return standing, upright, small_control, stand_reward, move

    def get_hurdle_reward(self):

        standing, upright, small_control, stand_reward, move = self.compute_rewards(
            0, 0.8, _HURDLE_SPEED
        )
        collision_discount = 1
        all_geoms_id = self._env.named.data.geom_xpos.axes.row.names

        for pair in self._env.data.contact.geom:
            if (
                any(["collision" in all_geoms_id[p_val] for p_val in pair])
                and 0 not in pair
            ):  #
                collision_discount = 0.1
                break

        reward = small_control * stand_reward * move * collision_discount

        return reward, {
            "stand_reward": stand_reward,
            "small_control": small_control,
            "move": move,
            "standing": standing,
            "upright": upright,
            "wall_collision_discount": collision_discount,
        }

    def get_pole_reward(self):
        standing, upright, small_control, stand_reward, move = self.compute_rewards(
            1, 0.9, _POLE_SPEED
        )

        all_geoms_id = self._env.named.data.geom_xpos.axes.row.names

        collision_discount = 1
        for pair in self._env.data.contact.geom:
            if (
                any(["collision" in all_geoms_id[p_val] for p_val in pair])
                and 0 not in pair
            ):  #
                collision_discount = 0.1
                break

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

    def get_slide_reward(self):
        standing, upright, small_control, stand_reward, move = self.compute_rewards(
            2, 0.5, _SLIDE_SPEED
        )
        return stand_reward * small_control * move, {  # small_control *
            "stand_reward": stand_reward,
            "small_control": small_control,
            "move": move,
            "standing": standing,
            "upright": upright,
        }
