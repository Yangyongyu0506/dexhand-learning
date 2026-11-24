"""绘制 Monitor 日志中的回合奖励与累计环境步数。"""

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np


def load_rewards(path):
    """Monitor 每行对应一个完整回合，r 为回报，l 为回合步数。"""
    with path.open(encoding="utf-8") as file:
        rows = csv.DictReader(line for line in file if not line.startswith("#"))
        if not {"r", "l"}.issubset(rows.fieldnames or []):
            raise ValueError("日志缺少 r、l 列，请使用 Monitor 生成的 CSV")
        returns, lengths = [], []
        for row in rows:
            # 训练期间读取时，忽略尚未写完的最后一行。
            if not row.get("r") or not row.get("l") or any(v is None for v in row.values()):
                continue
            reward, length = float(row["r"]), int(row["l"])
            if not np.isfinite(reward) or length <= 0:
                raise ValueError("日志中的回报必须有限，回合步数必须大于 0")
            returns.append(reward)
            lengths.append(length)
    if not returns:
        raise ValueError("还没有完整回合记录，请等待训练至少完成一个回合")
    return np.asarray(returns), np.asarray(lengths, dtype=np.int64)


def moving_average(values, window):
    """尾随窗口均值；不足一个窗口时使用当前已有的回合。"""
    sums = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    ends = np.arange(1, len(values) + 1)
    starts = np.maximum(0, ends - window)
    return (sums[ends] - sums[starts]) / (ends - starts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, help="训练日志目录，默认选择最新一次训练")
    parser.add_argument("--window", type=int, default=100, help="平滑窗口的回合数，默认 100")
    parser.add_argument("--output", type=Path, help="输出 PNG 路径，默认位于日志目录")
    parser.add_argument("--show", action="store_true", help="保存图片后打开绘图窗口")
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window 必须大于 0")

    if args.log_dir is None:
        root = Path(__file__).resolve().parents[1] / "outputs" / "train"
        logs = sorted(root.glob("*/monitor.csv"))
        if not logs:
            parser.error("找不到训练日志，请先运行 train.py，或通过 --log-dir 指定目录")
        log_file = logs[-1]
    else:
        log_file = args.log_dir.expanduser() / "monitor.csv"
    try:
        returns, lengths = load_rewards(log_file)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    # 横坐标是每个回合结束时累计的 env.step 次数，不是回合编号。
    steps = np.cumsum(lengths)
    mean_rewards = returns / lengths
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    for ax, values, label in zip(
        axes, (returns, mean_rewards), ("Episode return", "Mean reward per step (within episode)")
    ):
        ax.plot(steps, values, color="tab:blue", alpha=0.25, linewidth=0.7, label="Each episode")
        ax.plot(steps, moving_average(values, args.window), color="tab:blue", linewidth=1.8,
                label=f"Trailing mean ({args.window} episodes)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_title(f"HandReachDense-v3 / SAC — {log_file.parent.name}")
    axes[-1].set_xlabel("Training environment steps")
    output = args.output.expanduser() if args.output else log_file.parent / "reward_curve.png"
    if output.suffix.lower() != ".png":
        parser.error("--output 请使用 .png 扩展名")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output, dpi=160)
        print(f"日志：{log_file.resolve()}")
        print(f"完整回合数：{len(returns)}；累计环境步数：{steps[-1]}")
        print(f"曲线已保存：{output.resolve()}")
        if args.show:
            plt.show()
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
