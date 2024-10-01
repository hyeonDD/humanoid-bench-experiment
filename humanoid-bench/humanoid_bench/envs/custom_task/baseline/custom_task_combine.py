import numpy as np
from humanoid_bench.envs.pole import Pole
from humanoid_bench.envs.basic_locomotion_envs import Hurdle, ClimbingUpwards
from humanoid_bench.tasks import Task

class CustomTaskCombine(Task):
    """
    3개 task를 병합해서 실행
    """
    def __init__(self, robot=None, env=None, **kwargs):
        super().__init__(robot, env, **kwargs)
        self.pole_task = Pole()
        self.hurdle_task = Hurdle()
        self.climbing_task = ClimbingUpwards()

    def reset(self):
        # 모든 task를 초기화
        pole_obs = self.pole_task.reset()
        hurdle_obs = self.hurdle_task.reset()
        climbing_obs = self.climbing_task.reset()
        
        # 각 task의 관찰값을 통합
        return self._combine_observations(pole_obs, hurdle_obs, climbing_obs)

    def _combine_observations(self, pole_obs, hurdle_obs, climbing_obs):
        # 각 task의 관찰값을 결합하여 하나의 상태 벡터로 만듦
        combined_obs = np.concatenate([pole_obs, hurdle_obs, climbing_obs])
        return combined_obs

    def step(self, action):
        # 각 task에 동일한 action을 적용
        pole_obs, pole_reward, pole_done, pole_info = self.pole_task.step(action)
        hurdle_obs, hurdle_reward, hurdle_done, hurdle_info = self.hurdle_task.step(action)
        climbing_obs, climbing_reward, climbing_done, climbing_info = self.climbing_task.step(action)

        # 관찰값, 보상, 종료 조건 통합
        combined_obs = self._combine_observations(pole_obs, hurdle_obs, climbing_obs)
        combined_reward = self._combine_rewards(pole_reward, hurdle_reward, climbing_reward)
        done = pole_done and hurdle_done and climbing_done
        info = {'pole': pole_info, 'hurdle': hurdle_info, 'climbing': climbing_info}

        return combined_obs, combined_reward, done, info

    def _combine_rewards(self, pole_reward, hurdle_reward, climbing_reward):
        # 각 task의 보상을 합치거나 가중치를 두어 결합
        combined_reward = pole_reward + hurdle_reward + climbing_reward
        # TODO 각 task에 가중치를 반영해서 해보기
        # combined_reward = 0.5 * pole_reward + 0.3 * hurdle_reward + 0.2 * climbing_reward
        return combined_reward

    def render(self):
        # 모든 task의 렌더링을 호출
        self.pole_task.render()
        self.hurdle_task.render()
        self.climbing_task.render()
