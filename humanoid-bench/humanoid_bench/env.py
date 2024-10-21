import os

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium.envs import register
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from dm_control.mujoco import index
from dm_control.mujoco.engine import NamedIndexStructs
from dm_control.utils import rewards

from humanoid_bench.dmc_wrapper import MjDataWrapper, MjModelWrapper

from .wrappers import (
    SingleReachWrapper,
    DoubleReachAbsoluteWrapper,
    DoubleReachRelativeWrapper,
    BlockedHandsLocoWrapper,
    ObservationWrapper,
)

from .robots import H1, H1Hand, H1SimpleHand, H1Touch, H1Strong, G1
from .envs.cube import Cube
from .envs.bookshelf import BookshelfSimple, BookshelfHard
from .envs.window import Window
from .envs.spoon import Spoon
from .envs.door import Door
from .envs.basketball import Basketball
from .envs.basic_locomotion_envs import (
    Stand,
    Walk,
    Run,
    Hurdle,
    Crawl,
    Sit,
    SitHard,
    Stair,
    Slide,
)
from .envs.reach import Reach
from .envs.pole import Pole
from .envs.push import Push
from .envs.maze import Maze
from .envs.highbar import HighBarSimple, HighBarHard
from .envs.kitchen import Kitchen
from .envs.truck import Truck
from .envs.package import Package
from .envs.cabinet import Cabinet
from .envs.balance import BalanceHard, BalanceSimple
from .envs.room import Room
from .envs.powerlift import Powerlift
from .envs.insert import Insert

# CustomTask 추가
from .envs.custom_task.baseline.custom_task_combine import CustomTaskCombine
from .envs.custom_task.baseline.base_v2 import Base
from .envs.custom_task.base_v2_exp_early_terminate import BaseEarlyTerminate
from .envs.custom_task.base_v2_long_episode import BaseLong

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 5.0,
    "lookat": np.array((0.0, 0.0, 1.0)),
    "elevation": -20.0,
}
DEFAULT_RANDOMNESS = 0.01

ROBOTS = {
    "h1": H1,
    "h1hand": H1Hand,
    "h1simplehand": H1SimpleHand,
    "h1strong": H1Strong,
    "h1touch": H1Touch,
    "g1": G1,
}
TASKS = {
    "stand": Stand,
    "walk": Walk,
    "run": Run,
    "kitchen": Kitchen,
    "maze": Maze,
    "hurdle": Hurdle,
    "cube": Cube,
    "bookshelf_simple": BookshelfSimple,
    "bookshelf_hard": BookshelfHard,
    "highbar_simple": HighBarSimple,
    "highbar_hard": HighBarHard,
    "crawl": Crawl,
    "window": Window,
    "spoon": Spoon,
    "door": Door,
    "push": Push,
    "reach": Reach,
    "basketball": Basketball,
    "truck": Truck,
    "package": Package,
    "cabinet": Cabinet,
    "sit_simple": Sit,
    "sit_hard": SitHard,
    "balance_simple": BalanceSimple,
    "balance_hard": BalanceHard,
    "stair": Stair,
    "slide": Slide,
    "pole": Pole,
    "room": Room,
    "insert_normal": Insert,
    "insert_small": Insert,  # This is not an error
    "powerlift": Powerlift,
    # CustomTaskCombine 추가
    "custom_task_combine": CustomTaskCombine,
    "proto": Base,
    "proto_slide": BaseLong,
    "proto_slide_early_terminate": BaseEarlyTerminate,
    #     "walk_with_hurdle": Walk, # basic graph
    #     "walk_with_stair": Walk, # basic graph
    #     "walk_with_pole": Walk, # basic graph
    #     "hurdle_with_walk": Hurdle, # basic graph
    #     "hurdle_with_stair": Hurdle, # basic graph
    #     "hurdle_with_pole": Hurdle, # basic graph
    #     "stair_with_walk": Stair, # basic graph
    #     "stair_with_hurdle": Stair, # basic graph
    #     "stair_with_pole": Stair, # basic graph
    #     "pole_with_walk": Pole, # basic graph
    #     "pole_with_stair": Pole, # basic graph
    #     "pole_with_hurdle": Pole, # basic graph
}


class HumanoidEnv(MujocoEnv, gym.utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        robot=None,
        control=None,
        task=None,
        render_mode="rgb_array",
        width=256,
        height=256,
        randomness=DEFAULT_RANDOMNESS,
        **kwargs,
    ):
        assert robot and control and task, f"{robot} {control} {task}"
        gym.utils.EzPickle.__init__(self, metadata=self.metadata)
        self.random_start = kwargs.get("random_start", False)

        asset_path = os.path.join(os.path.dirname(__file__), "assets")
        model_path = f"envs/{robot}_{control}_{task}.xml"
        model_path = os.path.join(asset_path, model_path)

        self.robot = ROBOTS[robot](self)
        task_info = TASKS[task](self.robot, None, **kwargs)

        self.obs_wrapper = kwargs.get("obs_wrapper", None)
        if self.obs_wrapper is not None:
            self.obs_wrapper = kwargs.get("obs_wrapper", "False").lower() == "true"
        else:
            self.obs_wrapper = False

        self.blocked_hands = kwargs.get("blocked_hands", None)
        if self.blocked_hands is not None:
            self.blocked_hands = kwargs.get("blocked_hands", "False").lower() == "true"
        else:
            self.blocked_hands = False

        MujocoEnv.__init__(
            self,
            model_path,
            frame_skip=task_info.frame_skip,
            observation_space=task_info.observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            render_mode=render_mode,
            width=width,
            height=height,
            camera_name=task_info.camera_name,
        )

        self.action_high = self.action_space.high
        self.action_low = self.action_space.low
        self.action_space = Box(
            low=-1, high=1, shape=self.action_space.shape, dtype=np.float32
        )

        self.task = TASKS[task](self.robot, self, **kwargs)

        if self.blocked_hands:
            self.task = BlockedHandsLocoWrapper(self.task, **kwargs)

        # Wrap for hierarchical control
        if (
            "policy_type" in kwargs
            and kwargs["policy_type"]
            and kwargs["policy_type"] is not None
            and kwargs["policy_type"] != "flat"
        ):
            if kwargs["policy_type"] == "reach_single":
                assert "policy_path" in kwargs and kwargs["policy_path"] is not None
                self.task = SingleReachWrapper(self.task, **kwargs)
            elif kwargs["policy_type"] == "reach_double_absolute":
                assert "policy_path" in kwargs and kwargs["policy_path"] is not None
                self.task = DoubleReachAbsoluteWrapper(self.task, **kwargs)
            elif kwargs["policy_type"] == "reach_double_relative":
                assert "policy_path" in kwargs and kwargs["policy_path"] is not None
                self.task = DoubleReachRelativeWrapper(self.task, **kwargs)
            else:
                raise ValueError(f"Unknown policy_type: {kwargs['policy_type']}")

        if self.obs_wrapper:
            # Note that observation wrapper is not compatible with hierarchical policy
            self.task = ObservationWrapper(self.task, **kwargs)
            self.observation_space = self.task.observation_space

        # Keyframe
        self.keyframe = (
            self.model.key(kwargs["keyframe"]).id if "keyframe" in kwargs else 0
        )

        self.randomness = randomness
        if isinstance(self.task, (BookshelfHard, BookshelfSimple, Kitchen, Cube)):
            self.randomness = 0
        print(isinstance(self.task, (BookshelfHard, BookshelfSimple, Kitchen, Cube)))

        # Set up named indexing.
        data = MjDataWrapper(self.data)
        model = MjModelWrapper(self.model)
        axis_indexers = index.make_axis_indexers(model)
        self.named = NamedIndexStructs(
            model=index.struct_indexer(model, "mjmodel", axis_indexers),
            data=index.struct_indexer(data, "mjdata", axis_indexers),
        )

        assert self.robot.dof + self.task.dof == len(data.qpos), (
            self.robot.dof,
            self.task.dof,
            len(data.qpos),
        )

    def step(self, action):
        return self.task.step(action)

    def reset_model(self):

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.keyframe)

        if self.random_start:
            self.randomize_initial_position()

        mujoco.mj_forward(self.model, self.data)

        # Add randomness
        init_qpos = self.data.qpos.copy()
        init_qvel = self.data.qvel.copy()
        r = self.randomness
        self.set_state(
            init_qpos + self.np_random.uniform(-r, r, size=self.model.nq), init_qvel
        )

        # Task-specific reset and return observations
        return self.task.reset_model()

    def seed(self, seed=None):
        np.random.seed(seed)

    def render(self):
        return self.task.render()

    # 오브젝트 색 변경
    def change_geom_color(self, geom_id, color):
        """지정된 geom의 색상을 변경"""
        self.model.geom_rgba[geom_id] = color

    # 랜덤한 색으로 변경
    def randomize_visual_geom_colors(self):
        from mujoco import mj_id2name, mjtObj

        """visual class의 모든 geom의 색상을 무작위로 변경"""
        for geom_id in range(self.model.ngeom):
            # ID로 이름을 가져옴 (Python 바인딩에서 mj_id2name 함수 사용)
            geom_name = mj_id2name(self.model, mjtObj.mjOBJ_GEOM, geom_id)

            # 만약 geom_name이 None인 경우는 continue
            if geom_name is None:
                continue

            # geom의 이름에 "visual"이 포함되어 있는지 확인
            if "visual" in geom_name and "wall" not in geom_name:
                rng = np.random.default_rng()  # 새로운 난수 생성기 생성
                random_color = rng.random(4)
                random_color[3] = 1  # 불투명도는 1로 고정
                # geom의 색상 직접 변경
                self.model.geom_rgba[geom_id] = random_color

    def randomize_initial_position(self):
        """시작 위치 랜덤 조정"""
        rng = np.random.default_rng()
        # x 값을 0, 19, 27 중 하나로 무작위 설정
        x_position = rng.choice([0, 19, 52])
        self.data.qpos[0] = x_position


if __name__ == "__main__":
    register(
        id="temp-v0",
        entry_point="humanoid_bench.env:HumanoidEnv",
        max_episode_steps=1000,
        kwargs={
            "robot": "h1hand",
            "control": "pos",
            "task": "maze_hard",
        },
    )

    env = gym.make("temp-v0", render_mode="human")
    ob, _ = env.reset()
    print(f"ob_space = {env.observation_space}, ob = {ob.shape}")
    print(f"ac_space = {env.action_space.shape}")
    env.render()
    while True:
        action = env.action_space.sample()
        ob, rew, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            env.reset()
    env.close()
