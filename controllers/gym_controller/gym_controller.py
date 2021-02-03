import sys
import random
from controller import Supervisor

try:
    import gym
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
except ImportError:
    sys.exit(
        'Please make sure you have all dependencies installed. '
        'Run: "pip3 install numpy gym stable_baselines3"'
    )


class WebotsStickEnv(Supervisor, gym.Env):
    def __init__(self, max_episode_steps=1000):
        super().__init__()

        # Open AI Gym generic
        self.theta_threshold_radians = 0.4
        self.x_threshold = 0.3
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max
            ],
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=-0.01, high=0.01, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.state = None
        self.spec = gym.envs.registration.EnvSpec(id='WebotsEnv-v0', max_episode_steps=max_episode_steps)

        # Environment specific
        self.__timestep = int(self.getBasicTimeStep())
        self.__previous_x = 0
        self.__previous_theta = 0
        self.__motor = None
        self.__pendulum_sensor = None

        # Tools
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.__timestep)

    def wait_keyboard(self):
        while env.keyboard.getKey() != ord('Y'):
            super().step(self.__timestep)

    def reset(self):
        # Reset the simulation
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.__timestep)

        # Set random initial position
        initial_x = random.uniform(-self.x_threshold / 8, self.x_threshold / 8)
        initial_theta = random.uniform(-self.theta_threshold_radians / 8, self.theta_threshold_radians / 8)
        self.getFromDef('SLIDER_PARAMETERS').getField('position').setSFFloat(initial_x)
        self.getFromDef('PENDULUM_PARAMETERS').getField('position').setSFFloat(initial_theta)

        # Motors
        self.__motor = self.getDevice('linear motor')
        self.__motor.setPosition(float('inf'))
        self.__motor.setVelocity(0)

        # Sensors
        self.__motor.getPositionSensor().enable(self.__timestep)
        self.__pendulum_sensor = self.getDevice('pendulum sensor')
        self.__pendulum_sensor.enable(self.__timestep)

        # Internals
        self.__previous_x = 0
        self.__previous_theta = 0
        super().step(self.__timestep)

        # Open AI Gym generic
        return np.array([initial_x, 0, initial_theta, 0])

    def step(self, action):
        # Execute the action
        print(float(action))
        self.__motor.setVelocity(float(action))
        super().step(self.__timestep)

        # Observation
        x = self.__motor.getPositionSensor().getValue()
        theta = self.__pendulum_sensor.getValue()
        x_dot = (x - self.__previous_x) / (self.__timestep * 1e3)
        theta_dot = (theta - self.__previous_theta) / (self.__timestep * 1e3)
        self.__previous_x = x
        self.__previous_theta = theta
        self.state = np.array([x, x_dot, theta, theta_dot])

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
        else:
            reward = 0.0

        return self.state, reward, done, {}


# Initialize the environment
env = WebotsStickEnv()
check_env(env)

# Train
model = PPO('MlpPolicy', env, n_steps=2048, verbose=1)
model.learn(total_timesteps=1e3)

# Replay
print('Training is finished, press `Y` for replay...')
env.wait_keyboard()

obs = env.reset()
for t in range(100000):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    print(obs)
    if done:
        obs = env.reset()
