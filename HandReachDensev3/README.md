# HandReachDense-v3：动作、观测与 SAC 训练指南

适用于本目录 `main.py` 使用的默认环境 `gym.make("HandReachDense-v3")`，即 `relative_control=False`。已按本地 **Gymnasium 1.4.0、Gymnasium Robotics 1.4.2、MuJoCo 3.3.7** 的源码和运行时模型核对（2026-11-24）。所有索引从 **0** 开始，切片右端不包含在内。

## 1. 数据结构与范围约定

`reset()` 和 `step()` 返回的 `obs` 是字典，不是单一向量：

| 数据 | 形状 | 空间 dtype | 各分量的空间范围 | 内容 |
| --- | --- | --- | --- | --- |
| `action` | `(20,)` | `float32` | `[-1, 1]` | 20 个执行器的归一化位置目标 |
| `obs["observation"]` | `(63,)` | `float64` | `(-∞, +∞)` | 24 个关节角度 + 24 个角速度 + 15 个指尖坐标 |
| `obs["achieved_goal"]` | `(15,)` | `float64` | `(-∞, +∞)` | 五个指尖当前的世界坐标 |
| `obs["desired_goal"]` | `(15,)` | `float64` | `(-∞, +∞)` | 五个指尖目标的世界坐标 |

无界表示 `Box.low=-inf`、`Box.high=inf`，并不意味着手指可以运动到任意位置。下表另列模型关节限位；它是物理约束参数，不能当作对观测值的硬裁剪。角速度和指尖坐标没有声明有限上下界，也不能用有限次随机采样的最小／最大值替代真实边界。

名称缩写：`WR` 手腕、`FF` 食指、`MF` 中指、`RF` 无名指、`LF` 小指、`TH` 拇指；`CMC` 腕掌关节、`MCP` 掌指关节、`PIP` 近端指间关节、`DIP` 远端指间关节、`IP` 拇指指间关节。关节简称的完整名称是 `robot0:<简称>`。“水平／垂直”描述模型关节运动方向，不是摄像机画面坐标。

## 2. action：20 个分量

输入为无量纲的归一化动作。默认绝对位置控制下，环境先将动作裁剪到 `[-1, 1]`，再按 `model.actuator_ctrlrange` 映射为目标角度：

```text
ctrl[i] = (low[i] + high[i]) / 2
        + action[i] * (high[i] - low[i]) / 2
```

`action[i] = -1 / 0 / 1` 分别对应目标范围的下限／中点／上限。这是位置目标，不是力矩，也不保证执行一步后关节立即到达目标。

| action 索引 | 执行器 | 控制含义 | action 范围 | 目标角度范围（rad） |
| --- | --- | --- | --- | --- |
| 0 | `robot0:A_WRJ1` | 手腕桡偏／尺偏 | `[-1, 1]` | `[-0.489, 0.140]` |
| 1 | `robot0:A_WRJ0` | 手腕屈伸 | `[-1, 1]` | `[-0.698, 0.489]` |
| 2 | `robot0:A_FFJ3` | 食指 MCP 内收／外展 | `[-1, 1]` | `[-0.349, 0.349]` |
| 3 | `robot0:A_FFJ2` | 食指 MCP 屈伸 | `[-1, 1]` | `[0.000, 1.571]` |
| 4 | `robot0:A_FFJ1` | 食指 PIP 屈伸 | `[-1, 1]` | `[0.000, 1.571]` |
| 5 | `robot0:A_MFJ3` | 中指 MCP 内收／外展 | `[-1, 1]` | `[-0.349, 0.349]` |
| 6 | `robot0:A_MFJ2` | 中指 MCP 屈伸 | `[-1, 1]` | `[0.000, 1.571]` |
| 7 | `robot0:A_MFJ1` | 中指 PIP 屈伸 | `[-1, 1]` | `[0.000, 1.571]` |
| 8 | `robot0:A_RFJ3` | 无名指 MCP 内收／外展 | `[-1, 1]` | `[-0.349, 0.349]` |
| 9 | `robot0:A_RFJ2` | 无名指 MCP 屈伸 | `[-1, 1]` | `[0.000, 1.571]` |
| 10 | `robot0:A_RFJ1` | 无名指 PIP 屈伸 | `[-1, 1]` | `[0.000, 1.571]` |
| 11 | `robot0:A_LFJ4` | 小指 CMC 对掌运动 | `[-1, 1]` | `[0.000, 0.785]` |
| 12 | `robot0:A_LFJ3` | 小指 MCP 内收／外展 | `[-1, 1]` | `[-0.349, 0.349]` |
| 13 | `robot0:A_LFJ2` | 小指 MCP 屈伸 | `[-1, 1]` | `[0.000, 1.571]` |
| 14 | `robot0:A_LFJ1` | 小指 PIP 屈伸 | `[-1, 1]` | `[0.000, 1.571]` |
| 15 | `robot0:A_THJ4` | 拇指 CMC 水平方向转动 | `[-1, 1]` | `[-1.047, 1.047]` |
| 16 | `robot0:A_THJ3` | 拇指 CMC 垂直方向转动 | `[-1, 1]` | `[0.000, 1.222]` |
| 17 | `robot0:A_THJ2` | 拇指 MCP 内收／外展 | `[-1, 1]` | `[-0.209, 0.209]` |
| 18 | `robot0:A_THJ1` | 拇指 MCP 屈伸 | `[-1, 1]` | `[-0.524, 0.524]` |
| 19 | `robot0:A_THJ0` | 拇指 IP 屈伸 | `[-1, 1]` | `[-1.571, 0.000]` |

例如 `action[3]=0` 对应食指 `FFJ2` 的目标角度 `0.7855 rad`，不是零角度。当前 `main.py` 扫描某一维时先设置 `np.zeros(20)`，因此其余执行器会被命令回到各自范围中点，并非保持当前姿态。

食指、中指、无名指、小指的 `J0` 没有独立执行器，通过模型腱约束与对应 `J1` 耦合。因此有 24 个关节角度，却只有 20 个动作分量。动作与观测索引不是一一对应的，例如 `action[5]` 控制 `observation[6]` 中的 `MFJ3`。

## 3. observation：63 个分量

源码按以下顺序拼接，没有对这些观测做归一化：

```python
state = obs["observation"]
qpos = state[0:24]        # 24 个关节角度，rad
qvel = state[24:48]       # 相同顺序的 24 个关节角速度，rad/s
fingertips = state[48:63] # 五个指尖世界坐标，m
```

### 3.1 关节角度：索引 0–23

“观测范围”是 API 声明，“模型限位”来自 `model.jnt_range`，两者含义不同。

| observation 索引 | 关节 | 分量含义 | 单位 | 观测范围 | 模型限位（rad） |
| --- | --- | --- | --- | --- | --- |
| 0 | `WRJ1` | 手腕桡偏／尺偏角度 | rad | `(-∞, +∞)` | `[-0.489, 0.140]` |
| 1 | `WRJ0` | 手腕屈伸角度 | rad | `(-∞, +∞)` | `[-0.698, 0.489]` |
| 2 | `FFJ3` | 食指 MCP 内收／外展角度 | rad | `(-∞, +∞)` | `[-0.349, 0.349]` |
| 3 | `FFJ2` | 食指 MCP 屈伸角度 | rad | `(-∞, +∞)` | `[0.000, 1.571]` |
| 4 | `FFJ1` | 食指 PIP 屈伸角度 | rad | `(-∞, +∞)` | `[0.000, 1.571]` |
| 5 | `FFJ0` | 食指 DIP 屈伸（耦合关节）角度 | rad | `(-∞, +∞)` | `[0.000, 1.571]` |
| 6 | `MFJ3` | 中指 MCP 内收／外展角度 | rad | `(-∞, +∞)` | `[-0.349, 0.349]` |
| 7 | `MFJ2` | 中指 MCP 屈伸角度 | rad | `(-∞, +∞)` | `[0.000, 1.571]` |
| 8 | `MFJ1` | 中指 PIP 屈伸角度 | rad | `(-∞, +∞)` | `[0.000, 1.571]` |
| 9 | `MFJ0` | 中指 DIP 屈伸（耦合关节）角度 | rad | `(-∞, +∞)` | `[0.000, 1.571]` |
| 10 | `RFJ3` | 无名指 MCP 内收／外展角度 | rad | `(-∞, +∞)` | `[-0.349, 0.349]` |
| 11 | `RFJ2` | 无名指 MCP 屈伸角度 | rad | `(-∞, +∞)` | `[0.000, 1.571]` |
| 12 | `RFJ1` | 无名指 PIP 屈伸角度 | rad | `(-∞, +∞)` | `[0.000, 1.571]` |
| 13 | `RFJ0` | 无名指 DIP 屈伸（耦合关节）角度 | rad | `(-∞, +∞)` | `[0.000, 1.571]` |
| 14 | `LFJ4` | 小指 CMC 对掌运动角度 | rad | `(-∞, +∞)` | `[0.000, 0.785]` |
| 15 | `LFJ3` | 小指 MCP 内收／外展角度 | rad | `(-∞, +∞)` | `[-0.349, 0.349]` |
| 16 | `LFJ2` | 小指 MCP 屈伸角度 | rad | `(-∞, +∞)` | `[0.000, 1.571]` |
| 17 | `LFJ1` | 小指 PIP 屈伸角度 | rad | `(-∞, +∞)` | `[0.000, 1.571]` |
| 18 | `LFJ0` | 小指 DIP 屈伸（耦合关节）角度 | rad | `(-∞, +∞)` | `[0.000, 1.571]` |
| 19 | `THJ4` | 拇指 CMC 水平方向转动角度 | rad | `(-∞, +∞)` | `[-1.047, 1.047]` |
| 20 | `THJ3` | 拇指 CMC 垂直方向转动角度 | rad | `(-∞, +∞)` | `[0.000, 1.222]` |
| 21 | `THJ2` | 拇指 MCP 内收／外展角度 | rad | `(-∞, +∞)` | `[-0.209, 0.209]` |
| 22 | `THJ1` | 拇指 MCP 屈伸角度 | rad | `(-∞, +∞)` | `[-0.524, 0.524]` |
| 23 | `THJ0` | 拇指 IP 屈伸角度 | rad | `(-∞, +∞)` | `[-1.571, 0.000]` |

### 3.2 关节角速度：索引 24–47

每项是同名关节角度对时间的变化率，正负号与关节角度的正负方向一致。这里直接使用 MuJoCo `qvel`，没有乘以控制周期；角度限位不能用作角速度限位。

| observation 索引 | 关节 | 分量含义 | 单位 | 观测范围 |
| --- | --- | --- | --- | --- |
| 24 | `WRJ1` | 手腕桡偏／尺偏角速度 | rad/s | `(-∞, +∞)` |
| 25 | `WRJ0` | 手腕屈伸角速度 | rad/s | `(-∞, +∞)` |
| 26 | `FFJ3` | 食指 MCP 内收／外展角速度 | rad/s | `(-∞, +∞)` |
| 27 | `FFJ2` | 食指 MCP 屈伸角速度 | rad/s | `(-∞, +∞)` |
| 28 | `FFJ1` | 食指 PIP 屈伸角速度 | rad/s | `(-∞, +∞)` |
| 29 | `FFJ0` | 食指 DIP 屈伸（耦合关节）角速度 | rad/s | `(-∞, +∞)` |
| 30 | `MFJ3` | 中指 MCP 内收／外展角速度 | rad/s | `(-∞, +∞)` |
| 31 | `MFJ2` | 中指 MCP 屈伸角速度 | rad/s | `(-∞, +∞)` |
| 32 | `MFJ1` | 中指 PIP 屈伸角速度 | rad/s | `(-∞, +∞)` |
| 33 | `MFJ0` | 中指 DIP 屈伸（耦合关节）角速度 | rad/s | `(-∞, +∞)` |
| 34 | `RFJ3` | 无名指 MCP 内收／外展角速度 | rad/s | `(-∞, +∞)` |
| 35 | `RFJ2` | 无名指 MCP 屈伸角速度 | rad/s | `(-∞, +∞)` |
| 36 | `RFJ1` | 无名指 PIP 屈伸角速度 | rad/s | `(-∞, +∞)` |
| 37 | `RFJ0` | 无名指 DIP 屈伸（耦合关节）角速度 | rad/s | `(-∞, +∞)` |
| 38 | `LFJ4` | 小指 CMC 对掌运动角速度 | rad/s | `(-∞, +∞)` |
| 39 | `LFJ3` | 小指 MCP 内收／外展角速度 | rad/s | `(-∞, +∞)` |
| 40 | `LFJ2` | 小指 MCP 屈伸角速度 | rad/s | `(-∞, +∞)` |
| 41 | `LFJ1` | 小指 PIP 屈伸角速度 | rad/s | `(-∞, +∞)` |
| 42 | `LFJ0` | 小指 DIP 屈伸（耦合关节）角速度 | rad/s | `(-∞, +∞)` |
| 43 | `THJ4` | 拇指 CMC 水平方向转动角速度 | rad/s | `(-∞, +∞)` |
| 44 | `THJ3` | 拇指 CMC 垂直方向转动角速度 | rad/s | `(-∞, +∞)` |
| 45 | `THJ2` | 拇指 MCP 内收／外展角速度 | rad/s | `(-∞, +∞)` |
| 46 | `THJ1` | 拇指 MCP 屈伸角速度 | rad/s | `(-∞, +∞)` |
| 47 | `THJ0` | 拇指 IP 屈伸角速度 | rad/s | `(-∞, +∞)` |

### 3.3 指尖坐标与两个 goal 向量：逐分量对照

下表同时列出 `observation[48:63]`、`achieved_goal[0:15]` 和 `desired_goal[0:15]` 的全部分量。前两个索引表示该指尖的**当前位置**，第三个表示同一指尖的**目标位置**。单位均为米（m），坐标系均为 MuJoCo 世界坐标系，不是相对于手掌的位置，也不是目标误差。

三个数组对应分量的空间范围均为 `(-∞, +∞)`。实际指尖受几何结构和运动约束限制；目标由环境采样，没有提供固定的有限坐标边界。

| observation 索引 | achieved_goal 索引 | desired_goal 索引 | 指尖／坐标分量 | 当前指尖对应的 site | 单位 | 三者的空间范围 |
| --- | --- | --- | --- | --- | --- | --- |
| 48 | 0 | 0 | 食指指尖 x | `robot0:S_fftip` | m | `(-∞, +∞)` |
| 49 | 1 | 1 | 食指指尖 y | `robot0:S_fftip` | m | `(-∞, +∞)` |
| 50 | 2 | 2 | 食指指尖 z | `robot0:S_fftip` | m | `(-∞, +∞)` |
| 51 | 3 | 3 | 中指指尖 x | `robot0:S_mftip` | m | `(-∞, +∞)` |
| 52 | 4 | 4 | 中指指尖 y | `robot0:S_mftip` | m | `(-∞, +∞)` |
| 53 | 5 | 5 | 中指指尖 z | `robot0:S_mftip` | m | `(-∞, +∞)` |
| 54 | 6 | 6 | 无名指指尖 x | `robot0:S_rftip` | m | `(-∞, +∞)` |
| 55 | 7 | 7 | 无名指指尖 y | `robot0:S_rftip` | m | `(-∞, +∞)` |
| 56 | 8 | 8 | 无名指指尖 z | `robot0:S_rftip` | m | `(-∞, +∞)` |
| 57 | 9 | 9 | 小指指尖 x | `robot0:S_lftip` | m | `(-∞, +∞)` |
| 58 | 10 | 10 | 小指指尖 y | `robot0:S_lftip` | m | `(-∞, +∞)` |
| 59 | 11 | 11 | 小指指尖 z | `robot0:S_lftip` | m | `(-∞, +∞)` |
| 60 | 12 | 12 | 拇指指尖 x | `robot0:S_thtip` | m | `(-∞, +∞)` |
| 61 | 13 | 13 | 拇指指尖 y | `robot0:S_thtip` | m | `(-∞, +∞)` |
| 62 | 14 | 14 | 拇指指尖 z | `robot0:S_thtip` | m | `(-∞, +∞)` |

始终有 `obs["observation"][48:63] == obs["achieved_goal"]`（逐元素相等）。目标坐标没有包含在 63 维的 `observation` 中，需要从 `desired_goal` 单独读取。

默认通常选择拇指与另一根手指在掌上方相遇，其余指尖保持初始目标位置；有 10% 的概率将全部目标设为初始指尖位置。相遇点叠加标准差为 `0.005 m` 的高斯噪声，因此不能把目标坐标写成严格的有限区间。

## 4. Dense 奖励与目标误差

```python
error = obs["achieved_goal"] - obs["desired_goal"]
reward = -np.linalg.norm(error)
```

奖励是整个 15 维误差向量的欧氏范数取负，不是五个指尖距离的平均值。越接近 `0` 越好，数学范围为 `(-∞, 0]`；默认 `info["is_success"]` 判断该范数是否严格小于 `0.01 m`。

## 5. 本地核对方法与来源

在仓库根目录运行以下命令，不会打开渲染窗口：

```bash
uv run python - <<'PY'
import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)
env = gym.make("HandReachDense-v3")
try:
    obs, _ = env.reset(seed=42)
    print("action:", env.action_space)
    print("obs:", env.observation_space)
    model = env.unwrapped.model
    for i in range(model.nu):
        print("action", i, model.actuator(i).name, model.actuator_ctrlrange[i])
    for i in range(model.njnt):
        print("qpos", i, "qvel", i + 24, model.joint(i).name, model.jnt_range[i])
finally:
    env.close()
PY
```

本文以本地安装包的以下源码和模型为依据：

- `gymnasium_robotics/envs/shadow_dexterous_hand/reach.py`：指尖顺序、`_get_obs()`、目标采样及奖励。
- `gymnasium_robotics/envs/shadow_dexterous_hand/hand_env.py`：`MujocoHandEnv._set_action()` 的动作映射。
- `gymnasium_robotics/envs/robot_env.py`：动作与观测空间边界。
- `gymnasium_robotics/envs/assets/hand/robot.xml`、`shared.xml`：关节限位、腱耦合及执行器定义。

[官方 HandReach 文档](https://robotics.farama.org/envs/shadow_dexterous_hand/reach/)可用于对照任务和关节定义，但部分示例仍使用 v2；本目录使用 v3。修改模型、环境版本或控制方式后，应重新读取模型参数。

## 6. 评估训练好的 SAC 策略

`rl/train.py` 在训练结束后，将模型保存为当前工作目录中的 `hand_reach_sac.zip`。
在仓库根目录运行：

```bash
# 训练结束且模型已保存后，无窗口评估 20 个回合
uv run HandReachDensev3/rl/eval.py --model hand_reach_sac.zip --episodes 20

# 打开窗口，观察 5 个回合
uv run HandReachDensev3/rl/eval.py --model hand_reach_sac.zip --episodes 5 --render
```

评估使用 `SAC.load()` 加载模型，将完整 `obs` 字典传入
`model.predict(obs, deterministic=True)`，不再采样随机动作，也不更新网络。
这与 [SB3 官方 SAC 回放示例](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html#example)一致。
默认使用 CPU 推理，可通过 `--device cuda` 修改。

默认每回合种子为 `10000 + 回合索引`，与当前训练种子 42 分开；可用 `--seed` 指定一组固定评估目标以比较不同模型。
输出包括累计回报的均值与标准差、最终 15 维目标误差范数，以及两种成功率：

- **回合末成功率**：最后一步的 `info["is_success"]` 为真。
- **回合中曾成功率**：至少一个执行动作后的状态成功，不要求保持到回合结束。

即使中途成功也继续评估到 `terminated` 或 `truncated`，避免将短暂到达目标误认为持续成功。

## 7. train.py 逐段解释

本节对应 [rl/train.py](rl/train.py) 当前代码，网络结构和默认值按本地 **Stable-Baselines3 2.9.2、PyTorch 2.15.0** 实际构建核对。检查时仅构建模型，没有启动 10 万步训练；使用 CPU 检查不改变网络结构或参数量。

### 7.1 导入的四个对象分别负责什么

```python
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
```

| 对象 | 当前脚本中的职责 |
| --- | --- |
| `gym` | 通过环境 ID 查找并创建任务，提供 `reset()`、`step()`、空间与包装器接口 |
| `gymnasium_robotics` | 提供 HandReach 的 MuJoCo 模型、观测、控制、奖励和目标采样代码 |
| `SAC` | 创建 actor、critic、目标网络、优化器和经验回放池，管理训练循环 |
| `Monitor` | 统计每个回合的累计奖励、长度、耗时，供训练日志使用 |

脚本还使用标准库 `datetime` 和 `Path` 创建带时间戳的独立日志目录，避免不同训练运行的奖励记录互相覆盖。

```python
gym.register_envs(gymnasium_robotics)
```

在当前 Gymnasium 1.4.0 中，这个辅助函数本身是空操作，主要用于明确插件被使用；真正的环境注册在导入 `gymnasium_robotics` 时完成。注册只是建立“环境 ID → 实现类和默认配置”的映射，还没有创建手的物理仿真实例。

### 7.2 环境是怎么构造的

```python
env = Monitor(
    gym.make("HandReachDense-v3"),
    filename=str(log_dir / "monitor.csv"),
    info_keywords=("is_success",),
)
```

`gym.make()` 查找注册项，然后实例化 `gymnasium_robotics.envs.shadow_dexterous_hand.reach.MujocoHandReachEnv`，并传入 `reward_type="dense"`。底层读取 `envs/assets/hand/reach.xml` 及其引用的模型文件，创建 MuJoCo 的模型与运行状态，设置初始关节姿态，并定义第 1–4 节中的动作、观测与奖励。

当前实际包装顺序（外层到内层）是：

```text
Monitor                  将完整回合统计写入 monitor.csv
  TimeLimit              每回合最多 50 个环境步
    OrderEnforcing       检查先 reset 再 step 等调用顺序
      PassiveEnvChecker  检查环境接口、空间和返回值
        MujocoHandReachEnv
          MuJoCo model / data
```

没有传 `render_mode="human"`，所以默认不打开渲染窗口。这里仍然有完整的物理仿真，只是不显示画面。

| 环境配置 | 当前值 | 含义 |
| --- | --- | --- |
| 独立环境数 | 1 | 只有一只手在一个环境中采集数据 |
| MuJoCo 积分步长 | `0.002 s` | 一次底层物理积分推进的模拟时间 |
| `n_substeps` | 20 | 一次 `env.step(action)` 内执行 20 个物理步 |
| 控制周期 | `0.04 s`，即 25 Hz | 同一个动作在该控制周期内生效 |
| 回合时间上限 | 50 个环境步 | 对应 2 秒模拟时间，不是现实运行耗时 |
| 控制方式 | `relative_control=False` | 20 维动作映射为绝对关节位置目标 |
| 奖励 | `reward_type="dense"` | 15 维指尖目标误差范数的负值 |
| 成功阈值 | `distance_threshold=0.01` | 误差范数严格小于 0.01 m 时成功 |

`SAC` 接收该环境后，会再自动套上 **`DummyVecEnv`**，让输入带有环境批次维度。当前仍是 `n_envs=1`；名字中的 “Vec” 不代表已经使用多个进程或多个并行仿真实例。一个环境通常在时间上限处 `truncated=True`，随后训练框架自动重置并采样新目标；达到成功阈值本身不会结束回合。

### 7.3 SAC(...) 在构造什么

```python
model = SAC(
    "MultiInputPolicy",
    env,
    policy_kwargs={
        "net_arch": {
            "pi": [256, 256],
            "qf": [256, 256],
        },
    },
    # 后面是经验回放、训练和设备配置
)
```

`SAC(...)` 会检查空间、包装环境，建立策略网络、目标网络、优化器和经验回放池。构造对象不会开始学习；训练从 `model.learn(...)` 开始。

`MultiInputPolicy` 中的 “MultiInput” 指**处理字典里的多个输入字段**，不表示多个智能体或多个 actor。SB3 的 `policy` 对象包括训练需要的 actor、critic 和目标 critic，范围比“只输出动作的策略网络”更大。

### 7.4 93 维网络输入从哪里来，有卷积吗

当前 `MultiInputPolicy` 默认使用 `CombinedExtractor`。三个输入字段都是向量，因此每个分支只执行 `Flatten`，再拼接，没有带参数的特征提取层。实际拼接顺序由观测空间的键顺序决定：

```text
achieved_goal   (15) ── Flatten ──┐
desired_goal    (15) ── Flatten ──┼── 拼接为 93 维特征 x
observation    (63) ── Flatten ──┘
```

因此当前网络输入是 `15 + 15 + 63 = 93`，不是 63。`achieved_goal` 与 `observation` 的后 15 维重复，默认仍保留；`desired_goal` 告诉策略本回合想让指尖到哪里。

**当前没有卷积层、池化层、循环层、Attention、Dropout、BatchNorm 或 LayerNorm。** 所有可学习的网络层都是全连接 `Linear`。向量观测在进入网络前转为 `float32`；没有自动按均值／方差标准化。当前也没有使用 `VecNormalize`。

默认 `CombinedExtractor` 只有在字典中遇到符合图像空间定义的输入时，才会为相应分支创建 CNN。打开 `render_mode="human"` 只会显示画面，不会把画面放进 `obs`，也不会自动增加卷积层。

### 7.5 Actor 的层数、输出和激活函数

Actor 对应 `model.actor`，`net_arch["pi"]=[256, 256]` 指定两个隐藏层，每层 256 个单元：

```text
x (93)
  → Linear(93, 256) → ReLU
  → Linear(256, 256) → ReLU
      ├→ Linear(256, 20) → mu（高斯分布均值）
      └→ Linear(256, 20) → log_std（高斯分布标准差的对数）
```

层数有两种计数口径：**两个隐藏层**；包含输出头时，每条输入到输出路径有三个 Linear 层，而整个 actor 共有四个 Linear 模块，因为最后有两个并列输出头。`net_arch` 列表不包含输出头，它们由 SAC 根据 20 维动作空间自动创建。

当前隐藏层激活函数是 **`torch.nn.ReLU`**，来自 `MultiInputPolicy` 的默认参数 `activation_fn`，并未在 `train.py` 中显式写出。两个输出头都是线性输出，不追加 ReLU。

训练时，actor 从该高斯分布采样，再通过 `tanh` 将动作压到 `[-1, 1]`；评估的 `deterministic=True` 使用 `tanh(mu)`。注意后者是变换均值，不必等同于变换后分布的数学期望。`tanh` 在动作分布代码中执行，所以 `print(model.actor)` 不一定显示一个独立的 `Tanh` 层。

### 7.6 Critic 与目标 Critic

`net_arch["qf"]=[256, 256]` 指定 **每一个 Q 网络**的两个隐藏层。默认 `n_critics=2`，因此 `model.critic` 包含两个参数独立的 Q 网络 `qf0` 和 `qf1`，不是同一个网络仅有两个输出头。

每个 Q 网络的结构是：

```text
93 维特征 x + 20 维动作 a → 拼接为 113 维
  → Linear(113, 256) → ReLU
  → Linear(256, 256) → ReLU
  → Linear(256, 1) → Q(x, a)
```

每个 critic 输出一个标量，估计在当前目标下执行这个动作的长期软回报。输出层没有 ReLU，允许 Q 值为负。SAC 在相关更新中取两个 Q 值的较小者，减轻过高估计。

此外，`model.critic_target` 保存这两个 Q 网络的目标副本，结构完全相同。它们用于计算训练目标，通过软更新跟随在线 critic：

```text
target ← (1 - tau) × target + tau × critic
tau = 0.005
```

目标 critic 不由优化器直接做梯度更新；当前实现不另建目标 actor。`share_features_extractor=False` 是默认值，actor 与 critic 使用各自的特征提取器，不过本任务的提取器仅含无参数的 Flatten。

### 7.7 模型到底有多大

下表是实际构建后按 `sum(p.numel() for p in module.parameters())` 得到的数量。全连接层 `Linear(in, out)` 包括权重和偏置，共 `(in + 1) × out` 个参数。

| 模块 | 参数量 | 是否由优化器直接更新 |
| --- | ---: | --- |
| Actor | 100,136 | 是 |
| 在线 Q1 | 95,233 | 是 |
| 在线 Q2 | 95,233 | 是 |
| 两个目标 Q 网络 | 190,466 | 否，使用软更新 |
| `model.policy` 合计 | **481,068** | 包含以上全部网络 |
| 自动熵系数的 `log_ent_coef` | 1 | 是，位于 `model.policy` 之外 |

计算示例：

```text
Actor = (93+1)×256 + (256+1)×256 + 2×(256+1)×20
      = 100136
单个 Q = (113+1)×256 + (256+1)×256 + (256+1)×1
       = 95233
```

直接优化的 actor 和在线 critic 共 **290,602** 个网络参数，再加 1 个熵系数参数。仅使用策略执行动作时需要 actor，不需要运行 critic。481,068 个 float32 参数本身约占 **1.84 MiB**，这不包括优化器状态、梯度、经验回放和运行时内存，也不等于保存后的 ZIP 文件大小。

不能仅用 `requires_grad=True` 统计“由优化器更新的参数”：当前 SB3 中目标网络的参数也可能保留这个标志，但它们不在 critic 优化器中，更新方式仍是软更新。

### 7.8 参数在哪里设置

**在 `train.py` 显式设置的参数：**

| 位置／参数 | 当前值 | 作用 |
| --- | --- | --- |
| `SAC` 第一个参数 | `"MultiInputPolicy"` | 接收字典观测 |
| `policy_kwargs.net_arch.pi` | `[256, 256]` | actor 两个隐藏层的宽度 |
| `policy_kwargs.net_arch.qf` | `[256, 256]` | 每个 critic 两个隐藏层的宽度 |
| `buffer_size` | `100_000` | 最多保存 10 万条交互数据，满后循环覆盖旧数据 |
| `learning_starts` | `1_000` | 前 1000 步随机采样动作并积累经验，随后开始学习 |
| `batch_size` | `256` | 每次更新从回放池抽取 256 条数据，不是 256 个并行环境 |
| `seed` | `42` | 设置算法相关随机数及初始环境种子，后续重置仍会产生新目标 |
| `verbose` | `1` | 输出设备、环境包装和训练统计信息 |
| `device` | `0` | 当前脚本的设备写法，详见下文 |
| `learn(total_timesteps=...)` | `100_000` | 采集 10 万个环境步，不是 10 万回合或 epoch |

当前 `device=0` 会经 PyTorch 解释为设备索引，实际行为依赖可用设备与运行环境。SB3 的公开参数类型是字符串或 `torch.device`，建议在需要修改时明确写 `device="cuda:0"`、`"cpu"` 或 `"auto"`，并查看 `print(model.device)` 确认实际设备。SB3 在 CUDA 不可用时可回退到 CPU；这里的设备选择控制网络计算，不会把当前原生 MuJoCo 环境变成 GPU 并行仿真。

**脚本没有填写、由当前 SB3 默认提供的主要参数：**

| 参数 | 当前默认值 | 设置入口／含义 |
| --- | --- | --- |
| `activation_fn` | `torch.nn.ReLU` | `policy_kwargs`；actor 和 critic 隐藏层激活 |
| `features_extractor_class` | `CombinedExtractor` | `policy_kwargs`；处理字典的各输入分支 |
| `n_critics` | `2` | `policy_kwargs`；在线 Q 网络数量 |
| `share_features_extractor` | `False` | `policy_kwargs`；actor 与 critic 不共享提取器 |
| `optimizer_class` | `torch.optim.Adam` | `policy_kwargs`；actor 和 critic 的优化器 |
| `learning_rate` | `0.0003` | `SAC(...)`；当前为常数学习率，也用于自动熵系数优化 |
| `gamma` | `0.99` | `SAC(...)`；未来回报折扣因子 |
| `tau` | `0.005` | `SAC(...)`；目标 critic 软更新比例 |
| `train_freq` | `1` | `SAC(...)`；每采集一个环境步触发一次训练调用 |
| `gradient_steps` | `1` | `SAC(...)`；每次训练调用进行一轮梯度更新 |
| `target_update_interval` | `1` | `SAC(...)`；当前配置下每轮更新都软更新目标 critic |
| `ent_coef` | `"auto"` | `SAC(...)`；自动学习熵系数，初始 alpha 为 1 |
| `target_entropy` | `"auto"` → `-20` | `SAC(...)`；默认取动作维数的负值 |
| `use_sde` | `False` | `SAC(...)`；使用普通 SAC 随机策略，不启用 gSDE |
| `replay_buffer_class` | 自动选择 `DictReplayBuffer` | 当前字典观测的普通回放池，没有启用 HER |

网络配置与训练配置是两类参数。扩大 `buffer_size` 或 `batch_size` 不会增加网络层数；改变 `net_arch` 才会改变隐藏层和参数量。`"qf"` 是 SAC 的 Q 网络键名，不要照搬 PPO 示例里的 `"vf"`。

### 7.9 learn、打印与保存分别做什么

```python
print(model.actor)
print(model.critic)
model.learn(total_timesteps=100_000)
model.save("hand_reach_sac")
```

两条 `print` 仅显示网络模块结构，不代表已经训练，也不会显示所有算法超参数。改为 `print(model.policy)` 可同时看到目标 critic。

`learn()` 内部承担了原来手动编写的交互循环，并增加以下学习步骤：

1. 重置环境获得包含状态与目标的观测。前 1000 步使用随机动作，之后从 actor 的分布中采样动作进行探索。
2. 执行 `env.step(action)`，保存观测、动作、奖励、下一观测及结束信息到经验回放池。到达回合上限后自动重置环境。
3. 开始学习后，每次从回放池抽取 256 条数据。数据可来自不同回合，单条经验也可以被重复使用，这就是离策略训练的一部分。
4. 使用目标 critic 构造包含折扣和熵项的训练目标，更新两个在线 critic；再利用 critic 对动作的评价更新 actor，并优化自动熵系数。
5. 软更新目标 critic，继续采样。时间上限导致的截断与真实终止在计算训练目标时区别处理，不能一概把截断后的价值设为零。

在当前单环境、每回合 50 步的默认配置下，10 万步对应 2000 个完整回合；前 1000 步是预热，通常有 **99,000 轮梯度更新**。这里一轮更新包含多个优化器的工作，不等于仅更新一个网络一次。实际学习效果仍需第 6 节中的独立评估判断。

`model.save("hand_reach_sac")` 在当前工作目录生成 `hand_reach_sac.zip`，保存网络、配置及优化器等训练状态，但默认不保存经验回放池，也不保存当前仿真状态。需要保存回放池时另用 `model.save_replay_buffer(...)`。当前脚本只在 `learn()` 正常返回后保存；`finally: env.close()` 用于释放环境资源，中断时不会自动保存模型。

### 7.10 如何修改与检查网络

例如要改为三层隐藏层并显式指定激活函数，可以替换 `SAC(...)` 中的 `policy_kwargs`；以下是修改示例，不是当前训练配置：

```python
import torch.nn as nn

policy_kwargs = {
    "net_arch": {
        "pi": [256, 256, 128],
        "qf": [256, 256, 128],
    },
    "activation_fn": nn.ReLU,  # 传类，不写 nn.ReLU()
    "n_critics": 2,
}
# 在 SAC(..., policy_kwargs=policy_kwargs, ...) 中使用
```

列表有几个数字，就有几个隐藏层；每个数字是对应层的宽度。改用 `nn.Tanh` 会改变隐藏层激活，但 SAC 的动作范围仍由输出分布的 tanh 变换保证。需要自定义 CNN 或特征处理时，应配置 `features_extractor_class`，而不是只修改 `net_arch`；当前纯向量任务并不需要 CNN。

可在 `model = SAC(...)` 之后、`model.learn(...)` 之前插入以下检查代码：

```python
print("实际设备：", model.device)
print("完整网络：", model.policy)
print("并行环境数：", model.n_envs)
for name, module in [
    ("actor", model.actor),
    ("critic", model.critic),
    ("critic_target", model.critic_target),
]:
    print(name, sum(p.numel() for p in module.parameters()))
```

当前 `train.py` 在顶层直接运行训练，执行 `import train` 也会触发训练，因此不应通过导入它来单纯查看配置。

### 7.11 默认参数的源码位置

以下文件均位于本地 `.venv/lib/python3.13/site-packages/` 下：

| 文件 | 可以核对的内容 |
| --- | --- |
| `gymnasium_robotics/__init__.py` | `HandReachDense-v3` 的注册、dense 奖励和 50 步上限 |
| `gymnasium_robotics/envs/shadow_dexterous_hand/reach.py` | MuJoCo 环境实现、默认参数与观测拼接 |
| `stable_baselines3/sac/sac.py` | `SAC.__init__` 默认训练参数、`SAC.train` 更新逻辑 |
| `stable_baselines3/sac/policies.py` | `MultiInputPolicy`、`SACPolicy`、actor 和双 critic 的构建 |
| `stable_baselines3/common/torch_layers.py` | `CombinedExtractor`、全连接网络构建函数 |
| `stable_baselines3/common/distributions.py` | 高斯动作分布、采样与 tanh 变换 |
| `stable_baselines3/common/off_policy_algorithm.py` | 随机预热、数据采集、经验回放与训练调度 |
| `stable_baselines3/common/monitor.py` | 回合统计包装器 |

网络与默认参数以本地实例化结果和上述源码为准；可结合 [SB3 SAC 参数文档](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)及[自定义网络指南](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html)进一步阅读。

## 8. 绘制训练奖励随 step 的变化

奖励函数的定义不会在训练中变化；要画的是策略与环境交互时**得到的奖励**如何变化。当前采用 Monitor CSV 和 Matplotlib，支持训练期间生成快照，也支持训练结束后绘图。

### 8.1 记录奖励

从仓库根目录启动训练：

```bash
uv run HandReachDensev3/rl/train.py
```

每次运行自动生成独立目录，并在终端打印路径：

```text
HandReachDensev3/outputs/train/<时间戳>/monitor.csv
```

日志目录相对于脚本位置确定，不受启动时的工作目录影响。每完成一个回合，Monitor 写入并刷新一行，包含：

| 列 | 含义 |
| --- | --- |
| `r` | 回合累计奖励，即这一回合所有即时奖励之和 |
| `l` | 回合执行的环境步数，默认 50 |
| `t` | 从 Monitor 创建起经过的现实时间，单位秒 |
| `is_success` | 回合最后一步是否成功 |

CSV 第一行以 `#` 开头，是元数据；第二行才是列名。Monitor 的记录方式见[官方文档](https://stable-baselines3.readthedocs.io/en/master/common/monitor.html)。

### 8.2 画图命令

```bash
# 自动选择最新一次训练，生成 PNG，无需图形桌面
uv run HandReachDensev3/rl/plot_rewards.py

# 指定运行目录，将平滑窗口改为 20 个回合
uv run HandReachDensev3/rl/plot_rewards.py \
  --log-dir HandReachDensev3/outputs/train/<时间戳> --window 20

# 保存图片并打开绘图窗口（需要图形桌面）
uv run HandReachDensev3/rl/plot_rewards.py --show
```

默认保存到对应日志目录的 `reward_curve.png`；也可以用 `--output /path/to/curve.png` 指定位置。训练时可以另开终端执行绘图命令，每次读取已经完成的回合并更新图片；这不是自动刷新的实时窗口。最新一次训练还没有完整回合时，脚本会提示等待。

### 8.3 曲线怎么看

图片包含两个面板，共用横坐标 **累计环境步数**：

- 上图 `Episode return`：每个回合的累计奖励。
- 下图 `Mean reward per step (within episode)`：该回合累计奖励除以回合长度，即回合内平均单步奖励。
- 浅色线：每个完整回合的原始数据；深色线：最近 100 个回合的尾随移动平均，可用 `--window` 修改。开头不足一个窗口时使用已有回合。

横坐标计算为 `np.cumsum(episode_lengths)`，因此默认依次为 50、100、150……，不是回合编号，也不是 MuJoCo 的内部积分步数。移动平均的窗口单位是**回合**；100 个回合在本任务中对应 5000 个环境步。

HandReachDense 的即时奖励是目标距离的负值，因此曲线通常为负，**越接近 0 越好**。两张图都不是每一步即时 reward 的逐点记录；回合长度相同时，它们只相差一个固定比例。不同回合的目标难度不同，训练中的动作也包含探索，曲线不必单调上升；判断最终性能仍需使用第 6 节的独立确定性评估。

旧版本 `train.py` 没有为 Monitor 指定文件，因此之前已经完成的训练没有这份 CSV。单独保存的模型 ZIP 无法还原历史奖励曲线，需要从启用日志后的训练开始记录。
