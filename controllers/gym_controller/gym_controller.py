import gym
import numpy as np
from stable_baselines3 import HER, DDPG
from controller import Supervisor



class WebotsStickEnv(Supervisor, gym.Env):
    def __init__(self):
        super().__init__()

        # Open AI Gym generic
        self.theta_threshold_radians = 1
        self.x_threshold = 2000
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max
            ],
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.state = None
        self.steps_beyond_done = None

        # Environment specific
        self.__timestep = int(self.getBasicTimeStep())
        self.__previous_x = 0
        self.__previous_theta = 0
        self.__left_motor = None
        self.__right_motor = None
        self.__pendulum_sensor = None

    def __get_state(self):
        x = (self.__left_motor.getPositionSensor().getValue() + self.__right_motor.getPositionSensor().getValue()) / 2
        theta = self.__pendulum_sensor.getValue()
        x_dot = (x - self.__previous_x) / (self.__timestep * 1e3)
        theta_dot = (theta - self.__previous_theta) / (self.__timestep * 1e3)
        return np.array([x, x_dot, theta, theta_dot])

    def reset(self):
        # Reset the simulation
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.__timestep)

        # Motors
        self.__left_motor = self.getDevice('left wheel motor')
        self.__right_motor = self.getDevice('right wheel motor')
        self.__left_motor.setPosition(float('inf'))
        self.__right_motor.setPosition(float('inf'))

        # Sensors
        self.__left_motor.getPositionSensor().enable(self.__timestep)
        self.__right_motor.getPositionSensor().enable(self.__timestep)
        self.__pendulum_sensor = self.getDevice('pendulum sensor')
        self.__pendulum_sensor.enable(self.__timestep)

        # Internals
        self.__previous_x = 0
        self.__previous_theta = 0

        # Open AI Gym generic
        self.steps_beyond_done = None
        self.state = self.__get_state()
        return self.state

    def step(self, action):
        obs = None
        reward = None
        done = None
        info = {}

        super().step(self.__timestep)

        self.state = self.__get_state()
        obs = self.state

        # Done
        done = bool(
            self.state[0] < -self.x_threshold
            or self.state[0] > self.x_threshold
            or self.state[2] < -self.theta_threshold_radians
            or self.state[2] > self.theta_threshold_radians
        )

        # Reward
        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                gym.logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return obs, reward, done, info


import sys
import pathlib
import importlib

sys.path.insert(0, pathlib.Path(__file__).parent.absolute())


gym.envs.register(
    id='WebotsStick-v0',
    entry_point='gym_controller:WebotsStickEnv',
    max_episode_steps=1000
)
env = gym.make('CartPole-v0')

model = HER('MlpPolicy', env, DDPG)

for i in range(1000):
    observation = env.reset()
    for t in range(1000):
        # action, _states = model.predict(obs)
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
