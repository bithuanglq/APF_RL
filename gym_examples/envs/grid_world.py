import gym
from gym import spaces
import pygame
import numpy as np
import random


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=8):
        '''
            地图左上角为原点,向右x,向下y
        '''
        self.size = size  # The size of the square grid
        self.window_size = 1024  # The size of the PyGame window

        self.reward = 0
        self.step_length = 0

        # Observation space
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "loc_obs": spaces.Box(-2, 2, shape=(25, 6), dtype=int),     # 5x5局部视野,从上到下从左到右，每个位置维度为6
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, -1]),
            2: np.array([-1, 0]),
            3: np.array([0, 1]),
        }


        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location, "loc_obs": self._local_observation.flatten()}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            ),
            "episode": {
                "r": self.reward, "l": self.step_length, \
                "achieve_target": np.array_equal(self._agent_location, self._target_location)
            }
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.reward = 0
        self.step_length = 0

        # Position of obstacles
        self._dynamic_obstacle_location = np.random.randint(0, self.size-1, 2)
        self._static_obstacle_location = np.array([4,4])

        # 采用随机终点
        self._target_location = np.random.randint(0, self.size-1, 2)
        while np.array_equal(self._target_location, self._dynamic_obstacle_location) \
            or np.array_equal(self._target_location, self._static_obstacle_location):
            self._target_location = np.random.randint(0, self.size-1, 2)

        # 采用随机起点
        self._agent_location = np.random.randint(0, self.size-1, 2)
        while np.array_equal(self._agent_location, self._target_location) or \
            np.array_equal(self._agent_location, self._dynamic_obstacle_location) or \
            np.array_equal(self._agent_location, self._static_obstacle_location):
            self._agent_location = np.random.randint(0, self.size-1, 2)


        self._local_observation = np.zeros(shape=(25, 6))   # 6个维度：[x,y, 是否可行、目标、障碍物、越界]
        self._get_surronding_observations()     # 计算周围观察量
        observation = self._get_obs()       # 计算完成后获取观察量
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _get_surronding_observations(self):
       # Local observation
        self._local_observation = np.zeros(shape=(25, 6))   # [x,y, 是否可行、目标、障碍物、越界]
        for idx in range(25):       # 从左到右、从上到下逐次编码
            x,y = idx//5 -2, idx%5 -2          # 此处x,y为相对agent位置
            self._local_observation[idx][0], self._local_observation[idx][1] = x, y
            x,y = x + self._agent_location[0], y + self._agent_location[1]  # 此处x,y为绝对位置
            if x==self._target_location[0] and y==self._target_location[1]: 
                self._local_observation[idx][3] = 1
            elif (x==self._dynamic_obstacle_location[0] and y==self._dynamic_obstacle_location[1]) or \
                (x==self._static_obstacle_location[0] and y==self._static_obstacle_location[1]):
                self._local_observation[idx][4] = 1
            elif (x<0 or x>=self.size) or (y<0 or y>=self.size):  self._local_observation[idx][5] = 1
            else: self._local_observation[idx][2] = 1

    def _obstacle_step(self):
        '''
            dynamic obstacle move one step or stand still.
        '''
        prob = random.random() 
        x,y = self._dynamic_obstacle_location
        if prob <= 0.2: ret = np.array([x,y])
        elif 0.2 < prob <= 0.4: ret = np.clip([x+1,y], 0,self.size-1)
        elif 0.4 < prob <= 0.6: ret = np.clip([x,y-1], 0,self.size-1)
        elif 0.6 < prob <= 0.8: ret = np.clip([x-1,y], 0,self.size-1)
        elif 0.8 < prob <=1:    ret = np.clip([x,y+1], 0,self.size-1)
        if np.array_equal(ret, self._target_location):
            ret = np.array([x,y])

        self._dynamic_obstacle_location = ret

    def step(self, action):
        self.step_length += 1

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]

        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        self._obstacle_step()   # dynamic obstacle moves one step
        self._get_surronding_observations()

        # An episode is done iff the agent has reached the target or obstacle
        terminated = np.array_equal(self._agent_location, self._target_location) \
                        or np.array_equal(self._agent_location, self._dynamic_obstacle_location) \
                        or np.array_equal(self._agent_location, self._static_obstacle_location)
                




        # reward function
        reward = -np.abs(self._agent_location - self._target_location).mean() / self.size # r = -d , [-1,1]
        if np.array_equal(self._agent_location, self._target_location): 
            reward += 1
        if np.array_equal(self._agent_location, self._dynamic_obstacle_location) or \
            np.array_equal(self._agent_location, self._static_obstacle_location):
            reward += -1
        self.reward += reward
        




        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        if self.step_length >= 200:         # max step length
            return observation, reward, True, False, info
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the obstacle -- black
        pygame.draw.rect(
            canvas,
            (0, 0, 0),
            pygame.Rect(
                pix_square_size * self._dynamic_obstacle_location,
                (pix_square_size, pix_square_size),
            ),
        )
        pygame.draw.rect(
            canvas,
            (0, 0, 0),
            pygame.Rect(
                pix_square_size * self._static_obstacle_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # we draw the target -- green
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent -- blue
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
