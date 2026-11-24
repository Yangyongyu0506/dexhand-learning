# HandReachDense-v3：动作与观测分量速查

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
