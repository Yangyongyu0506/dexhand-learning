import gymnasium as gym
import gymnasium_robotics
import numpy as np

gym.register_envs(gymnasium_robotics)

env = gym.make(
    "HandReachDense-v3",
    render_mode="human",
)

obs, info = env.reset()

# action_l = np.zeros(20)
# action_h = np.zeros(20)
# for i in range(1000):
#     action = env.action_space.sample()
#     action_l = np.minimum(action_l, action)
#     action_h = np.maximum(action_h, action)

# print(f"Action low: {action_l}\n")
# print(f"Action high: {action_h}\n")
# The code above assures that the action space is indeed [-1, 1] for each of the 20 dimensions.

# action is a 20-dimensional vector
try:
    while True:
        for id in range(20):
            action = np.zeros(20)
            for i in range(51):
                action[id] = -1 + (i / 50) * 2
                obs, reward, terminated, truncated, info = env.step(action)

                print(f"Action: {action}\n")
                print(f"Reward: {reward}\n")

                if terminated or truncated:
                    obs, info = env.reset()

except KeyboardInterrupt:
    env.close()