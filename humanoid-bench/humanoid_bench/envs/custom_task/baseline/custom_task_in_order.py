import numpy as np
from humanoid_bench.envs.pole import Pole
from humanoid_bench.envs.basic_locomotion_envs import Hurdle, ClimbingUpwards

class CustomTaskInOrder:
    """
    3개 task를 순서대로
    """
    def __init__(self, ):
        self.pole_task = Pole()
        self.hurdle_task = Hurdle()
        self.climbing_task = ClimbingUpwards()
        self.current_task = 'pole'
        self.completed_tasks = []

    def reset(self):
        self.completed_tasks = []
        self.pole_task.reset()
        self.hurdle_task.reset()
        self.climbing_task.reset()
        self.current_task = 'pole'
        return self._get_observation()

    def _get_observation(self):
        if self.current_task == 'pole':
            return self.pole_task.get_observation()
        elif self.current_task == 'hurdle':
            return self.hurdle_task.get_observation()
        elif self.current_task == 'climbing':
            return self.climbing_task.get_observation()

    def step(self, action):
        if self.current_task == 'pole':
            obs, reward, done, info = self.pole_task.step(action)
            if done:
                self.completed_tasks.append('pole')
                self.current_task = 'hurdle'
                obs = self.hurdle_task.reset()
        elif self.current_task == 'hurdle':
            obs, reward, done, info = self.hurdle_task.step(action)
            if done:
                self.completed_tasks.append('hurdle')
                self.current_task = 'climbing'
                obs = self.climbing_task.reset()
        elif self.current_task == 'climbing':
            obs, reward, done, info = self.climbing_task.step(action)
            if done:
                self.completed_tasks.append('climbing')
                done = True  # All tasks completed
        return obs, reward, done, info

    def render(self):
        if self.current_task == 'pole':
            self.pole_task.render()
        elif self.current_task == 'hurdle':
            self.hurdle_task.render()
        elif self.current_task == 'climbing':
            self.climbing_task.render()
