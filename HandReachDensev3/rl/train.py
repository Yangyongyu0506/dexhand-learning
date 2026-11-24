import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

gym.register_envs(gymnasium_robotics)

# 训练时关闭实时渲染
env = Monitor(gym.make("HandReachDense-v3"))

try:
    model = SAC(
        "MultiInputPolicy",
        env,
        policy_kwargs={
            "net_arch": {
                "pi": [256, 256],  # actor 隐藏层
                "qf": [256, 256],  # critic 隐藏层
            },
        },
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=256,
        seed=42,
        verbose=1,
        device=0
    )

    print(model.actor)
    print(model.critic)

    model.learn(total_timesteps=100_000)
    model.save("hand_reach_sac")
finally:
    env.close()