from gymnasium.envs import register

from .env import ROBOTS, TASKS

for robot in ROBOTS:
    if robot == "g1" or robot == "digit":
        control = "torque"
    else:
        control = "pos"
    for task, task_info in TASKS.items():
        task_info = task_info()
        kwargs = task_info.kwargs.copy()
        kwargs["robot"] = robot
        kwargs["control"] = control
        kwargs["task"] = task
        register(
            id=f"{robot}-{task}-v0",
            entry_point="humanoid_bench.env:HumanoidEnv",
            max_episode_steps=task_info.max_episode_steps,
            kwargs=kwargs,
        )

import gym

# 등록된 모든 환경 목록 출력
env_specs = gym.envs.registry
env_list = list(env_specs.keys())

for env_name in env_list:
    print(env_name)
