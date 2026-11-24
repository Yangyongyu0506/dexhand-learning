"""加载 SAC 模型，以独立目标种子评估 HandReachDense-v3。"""

import argparse
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from stable_baselines3 import SAC


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path("hand_reach_sac.zip"),
        help="模型路径，相对于当前工作目录（与 train.py 的保存方式一致）",
    )
    parser.add_argument("--episodes", type=int, default=20, help="评估回合数")
    parser.add_argument("--seed", type=int, default=10_000, help="评估起始种子")
    parser.add_argument("--render", action="store_true", help="打开可视化窗口")
    parser.add_argument("--device", default="cpu", help="推理设备：cpu、cuda 或 auto")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes 必须大于 0")
    if args.seed < 0:
        parser.error("--seed 必须非负")
    checkpoint = args.model.expanduser()
    if not checkpoint.is_file() and checkpoint.suffix != ".zip":
        checkpoint = Path(f"{checkpoint}.zip")
    if not checkpoint.is_file():
        parser.error(
            f"找不到模型：{checkpoint.resolve()}。请先运行 train.py，"
            "或用 --model 指定已有模型。"
        )

    gym.register_envs(gymnasium_robotics)
    env = gym.make(
        "HandReachDense-v3", render_mode="human" if args.render else None
    )
    returns, final_distances = [], []
    final_successes, ever_successes = [], []
    try:
        # 传入环境以核对空间；评估只调用 predict，不调用 learn。
        model = SAC.load(checkpoint, env=env, device=args.device)
        print(f"模型：{checkpoint.resolve()}；设备：{model.device}")
        for episode in range(args.episodes):
            episode_seed = args.seed + episode
            obs, _ = env.reset(seed=episode_seed)
            episode_return = 0.0
            steps = 0
            ever_success = False
            while True:
                # 传入整个目标字典；deterministic=True 关闭动作随机采样。
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                steps += 1
                final_success = bool(info["is_success"])
                ever_success = ever_success or final_success
                # 成功不意味着回合结束，继续执行到环境终止或时间上限。
                if terminated or truncated:
                    break

            distance = float(np.linalg.norm(
                obs["achieved_goal"] - obs["desired_goal"]
            ))
            returns.append(episode_return)
            final_distances.append(distance)
            final_successes.append(final_success)
            ever_successes.append(ever_success)
            print(
                f"回合 {episode + 1}/{args.episodes} seed={episode_seed} "
                f"步数={steps} 回报={episode_return:.4f} "
                f"最终距离={distance:.6f} m "
                f"最终成功={final_success} 曾成功={ever_success}"
            )

        print(f"\n平均回报：{np.mean(returns):.4f} ± {np.std(returns):.4f}（标准差）")
        print(f"平均最终目标距离（15 维误差范数）：{np.mean(final_distances):.6f} m")
        print(f"回合末成功率：{np.mean(final_successes):.1%}")
        print(f"回合中曾成功率（至少一个执行后状态成功）：{np.mean(ever_successes):.1%}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
