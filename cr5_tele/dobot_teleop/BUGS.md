# BUG 记录与排查文档

## 约定

- 每条 BUG 记录必须包含：**现象、触发条件、根因（带代码引用）、修复方案、验证方法**
- 按时间倒序排列（最新的在最上面）
- 排查新 BUG 时先查本文档是否有类似问题

---

## BUG-001: 首次 RG 按下时机械臂剧烈抖动（Euler 角表示不唯一）

**日期**：2026-06-27

### 现象

- 遥操作启动（`auto_enable`）后，**第一次**按下 RG 手柄侧键时，机械臂会突然剧烈抖一下
- 运动中松开 RG 再重新按下（离合器）**不会**抖动
- 与奇异点无关（J5 不在 0° 或 90° 附近也会触发）
- 切换为 ServoP 模式不会出现

### 触发条件

1. 启动遥操作脚本 → `auto_enable` 对齐
2. 按下 RG 侧键（第一次）
3. 机械臂抖动

### 代码链路

#### 步骤 1：对齐时 `_last_target` 的初始化

`toolframe_governor_teleop.py` 中 `enable_teleop()` 调用 `mapper.reset()`：

```python
# toolframe_governor_teleop.py → enable_teleop()
mapper.reset(latest_pose, robot_pose)   # robot_pose 来自 GetPose()
```

`toolframe_mapping.py` 中 `ToolFrameQuestTeleopMapper.reset()`：

```python
# toolframe_mapping.py → reset()
self.robot_origin = list(robot_pose)          # 原始 GetPose 返回值
self._last_target = list(robot_pose)          # ← 关键：存储了原始 Euler 角
self._accum_delta_pos = np.zeros(3)
self._accum_delta_rot = np.zeros(3)
```

此时 `_last_target` 存储的是 Dobot 控制器 `GetPose()` 直接返回的值，例如：
```
[-436.9, -538.8, 508.8,  -172.2,  -3.7,  146.9]
                          ↑ Rx    ↑ Ry   ↑ Rz   (Euler XYZ, 度)
```

#### 步骤 2：首次 RG 按下 → 计算目标位姿

`toolframe_mapping.py` 中 `target_from_quest()` 构建 `raw_target` 的旋转部分：

```python
# toolframe_mapping.py → target_from_quest()
origin_R = self._euler_deg_to_R(self.robot_origin[3:])   # GetPose 原始 Euler → 矩阵
delta_R_mat = R.from_rotvec(self._accum_delta_rot).as_matrix()  # 累积旋转 (首次=0 → I)
target_R = origin_R @ delta_R_mat                        # = origin_R (首次)
target_euler = R.from_matrix(target_R).as_euler("XYZ", degrees=True)  # ← 矩阵 → Euler
```

**关键**：`R.from_matrix(target_R).as_euler()` 返回的 Euler 角是 scipy 的 **canonical 表示**，可能与 `GetPose()` 返回的原始值**不同**，但它们代表**同一个旋转矩阵**。

例如：
```
GetPose 返回:    Rx=-172.2°  Ry=-3.7°   Rz=146.9°
as_euler 返回:   Rx=  7.8°   Ry= 3.7°   Rz=-33.1°    ← 等价的另一个 Euler 表示！
```

两个三元组对应同一个旋转矩阵 `R`，但数值完全不同。

#### 步骤 3：计算 sent_step → 触发虚假的大角度差

`toolframe_mapping.py` 中 `_info_for_target()` 计算本帧的旋转步长：

```python
# toolframe_mapping.py → _info_for_target()
rot_diffs = [
    angle_diff_deg(target[i], self._last_target[i]) for i in range(3, 6)
]
# target = [..., 7.8, 3.7, -33.1]       ← canonical 表示
# _last_target = [..., -172.2, -3.7, 146.9]  ← GetPose 原始值
# rot_diffs[0] = angle_diff_deg(7.8, -172.2) ≈ 180.0°   ← 虚假大差值！
```

`angle_diff_deg(7.8, -172.2)` = `(7.8 - (-172.2) + 180) % 360 - 180` = `360 % 360 - 180` = **-180°**

#### 步骤 4：触发发送命令

`toolframe_governor_teleop.py` 中 `should_send_target()`：

```python
# toolframe_governor_teleop.py
sent_rot = info.get("sent_step_deg", 0.0)   # ≈ 180°
# 远超 deadband (1.0°) 和 max_step (0.5°)
raw_changed = rg_pressed and should_send_target(info)  # → True
send_target = True
```

代码认为机械臂需要"修正"一个 180° 的旋转误差，但实际上没有任何旋转误差——只是同一个朝向用了两种等价的 Euler 角表示。

#### 步骤 5：ServoJ 发送大角度命令 → 抖动

```python
# toolframe_governor_teleop.py
desired_joints = client.inverse_kin(cmd_target)
# IK 求解包含了虚假的大角度旋转 → 关节角大幅偏离当前值
planned_joints, joint_step, _ = plan_joint_step(desired_joints, last_sent_joints, max_joint_step_deg)
# plan_joint_step 限制了每步最大 0.7°，但第一个非零步仍然会突然开始旋转
client.servoj(planned_joints, ...)
```

即使 `plan_joint_step` 把每步限制在 0.7°，但从静止突然开始转 0.7°，仍然会被感知为"抖一下"。而且如果 `max_joint_step_deg` 被之前的命令消耗掉（比如之前有微小位置移动），第一帧可能会有更大的旋转步。

#### 为什么后续离合器不会触发？

第一次 RG 按下后，`_last_target` 被更新为 `target_from_quest` 返回的目标位姿（使用 canonical Euler）：

```python
# toolframe_mapping.py → target_from_quest()
self._last_target = target   # 已经是 canonical 表示
```

后续 RG 松开再按下时，`sync_accum_to_pose()` 把 `_accum_delta` 同步到当前位置后，`target_from_quest` 构建的 `target_euler` 与 `_last_target` 使用同一套 Euler 表示，`angle_diff_deg` 返回 ≈ 0，不会触发虚假修正。

### 根因

**Euler 角表示不唯一**：`scipy.spatial.transform.Rotation` 的 `as_euler()` 和 Dobot 控制器的 `GetPose()` 可能返回同一个旋转矩阵的不同 Euler 角三元组。代码在不同路径中混用了两种表示，导致 `angle_diff_deg` 算出虚假的大角度差。

涉及的关键变量：

| 变量 | 来源 | Euler 表示 | 设置位置 |
|---|---|---|---|
| `_last_target` | `mapper.reset()` → `GetPose()` | 控制器原始值 | `toolframe_mapping.py:144` |
| `target_euler` | `target_from_quest()` → `R.from_matrix().as_euler()` | scipy canonical | `toolframe_mapping.py:204-207` |
| `raw_target` | 同上 | scipy canonical | 同上 |

### 修复

`toolframe_mapping.py` 的 `sync_accum_to_pose()` 中，同步完 `_accum_delta_pos` 和 `_accum_delta_rot` 后，额外将 `_last_target` 也更新为 canonical Euler 表示：

```python
# toolframe_mapping.py → sync_accum_to_pose() 新增代码
canonical_euler = R.from_matrix(
    origin_R @ R.from_rotvec(self._accum_delta_rot).as_matrix()
).as_euler("XYZ", degrees=True)
self._last_target = [
    target_pose[0], target_pose[1], target_pose[2],
    float(canonical_euler[0]),
    float(canonical_euler[1]),
    float(canonical_euler[2]),
]
```

`sync_accum_to_pose()` 在每次 RG 按下（包括首次）时被调用（`toolframe_governor_teleop.py` 中离合器逻辑），因此 `_last_target` 在第一次 RG 按下前就被统一到 canonical 表示，消除了 `angle_diff_deg` 的虚假差值。

### 验证方法

- 启动遥操作，按下 `e` 对齐
- 按下 RG 侧键 → 机械臂不应抖动，应平滑跟随手柄运动
- 运动中松开 RG，移动手柄到新位置，再按下 RG → 也不应抖动

### 教训

- **任何比较 Euler 角的场景，必须确保两个值来自同一种表示**（全部用 `R.from_matrix().as_euler()` 或全部用原始值）
- `GetPose()` 返回的原始角度和 `as_euler()` 返回的角度**不能直接比较**
- 如果必须比较两个方向，用旋转矩阵或四元数计算角度差（`R1 * R2.inv()` 的 `magnitude()`），不要比较 Euler 角分量

---

## BUG-002: ServoJ 模式下 IK unsolvable（`joint_near` 参数格式错误）

**日期**：2026-06-27

### 现象

遥操作启动后，所有 `inverse_kin()` 调用返回失败，日志持续输出 `IK unsolvable`。

### 触发条件

`dobot_dashboard.py` 中 `inverse_kin()` 传入 `joint_near` 参数后触发。

### 根因

SDK `InverseKin` 方法发送命令时，参数名使用大写 `JointNear=`，但 Dobot 控制器协议要求小写 `jointNear=`。详见 SDK 源码：

```python
# TCP-IP-Python-V4/dobot_api.py:882
params.append('JointNear={:s}'.format(JointNear))  # 大写 J → 控制器不识别
```

不传 `joint_near` 时（默认行为），控制器根据**当前关节角**就近选解，已经满足需求。

### 修复

`toolframe_governor_teleop.py` 和 `servoj_toolframe_teleop.py` 中恢复为无参数调用：

```python
desired_joints = client.inverse_kin(cmd_target)
# 不传 joint_near: 控制器默认根据当前关节角就近选解
```

---

## BUG-003: 机械臂"不受控制一直伸直"（离合器 position 滞后）

**日期**：2026-06-27

### 现象

- 用户手柄往回拉，机械臂却继续往外伸
- 日志中 `cmd_lag_pos` 达到 300mm+

### 触发条件

1. 按 RG 移动机械臂到远处
2. 松开 RG
3. 手柄移回近处
4. 再次按 RG

### 根因

RG 松开再按下时，mapper 的 `_accum_delta_pos`（累积位移）不会被清零。上一段运动中累积的大位移（如 300mm）继续作为 `raw_target`。Governor 拼命从当前位置追赶这个远方目标，而不是从当前位置重新开始。

### 修复

在 `toolframe_governor_teleop.py` 中，检测到 RG 从 False→True 时，调用 `mapper.sync_accum_to_pose(governor.current_target())` 把 mapper 累积位移同步到 governor 当前位置，消除追赶滞后。

---

## BUG-004: 离合器旋转滞后（`sync_accum_to_pose` 未同步旋转）

**日期**：2026-06-27

### 现象

- 第一次 RG 按下后机械臂旋转方向不对（或抖动）
- 位置跟随正常，旋转不正常

### 触发条件

BUG-003 修复后，`sync_accum_to_pose` 只同步了位置，未同步旋转。

### 根因

`sync_accum_to_pose` 最初只更新 `_accum_delta_pos`，未更新 `_accum_delta_rot`。RG 重新按下时，旋转累积还是上一个周期的旧值。

### 修复

在 `sync_accum_to_pose` 中增加旋转同步：

```python
origin_R = self._euler_deg_to_R(self.robot_origin[3:])
target_R = self._euler_deg_to_R(target_pose[3:])
delta_R = origin_R.T @ target_R
self._accum_delta_rot = R.from_matrix(delta_R).as_rotvec()
self._origin_delta_base_rot = self._accum_delta_rot.copy()
```

---

## BUG-005: ServoJ 触发碰撞检测 → robot_mode=11

**日期**：2026-06-27

### 现象

- 机械臂运动一段时间后突然停下
- 日志显示 `ServoJ rejected, robot_mode=11`
- Dobot V4 文档：mode 11 = `ROBOT_MODE_COLLISION`（碰撞状态）

### 触发条件

ServoJ 模式下，`servo_gain` 过高 + `servo_t` 过短，导致控制器施加大力矩 → 被碰撞检测误判为碰撞。

### 修复

1. 降低 `servo_gain` 800→700，增大 `servo_t` 0.02→0.03，匹配 `command_rate` 33Hz
2. 添加 `SetCollisionLevel(1)` 降低碰撞检测灵敏度
3. 添加自动清除：当 `robot_mode==11` 时自动调用 `ClearError()`

---

## 碰撞检测 / IK / 伺服相关错误码速查

| robot_mode | 含义 | 处理 |
|---|---|---|
| 5 | ENABLE（空闲） | 正常 |
| 7 | RUNNING（运动中） | 正常 |
| 11 | COLLISION（碰撞） | 自动 ClearError，降低 gain/servo_t |

| error_id | 含义 | 处理 |
|---|---|---|
| 0 | 成功 | — |
| -1 | 命令失败 | 检查 robot_mode，若 mode=11 则 ClearError |
| 其他负数 | 见 Dobot 文档 | 查 `alarm_controller.json` |

---

# 代码修改记录（非 BUG，架构扩展）

## MOD-001: openpi 新增 CR5A 训练支持

**日期**：2026-07-01

### 目标

让 openpi/pi0 能够微调 CR5A 录制的 LeRobot 格式数据集。

### 涉及文件

| 文件 | 操作 | 说明 |
|---|---|---|
| `openpi/src/openpi/policies/cr5a_policy.py` | **新增** | CR5A 数据 ↔ pi0 模型格式的转换策略 |
| `openpi/src/openpi/training/config.py` | 修改 | 新增 `LeRobotCR5ADataConfig` + 注册 4 个训练 config |
| `openpi/src/openpi/training/data_loader.py` | 修改 | 新增 `root` 参数支持本地数据集路径 |

### 修改详情

#### 1. `cr5a_policy.py`（新增）

CR5A LeRobot 数据集的 key 名和 pi0 模型期望的 key 名不同，需要映射：

```text
LeRobot 数据集 key                →  pi0 模型 key
──────────────────────────────────────────────────
observation.state                 →  "observation/state" (7D: j1..j6+gripper)
observation.images.d435           →  "observation/image"  (场景相机 → base_0_rgb)
observation.images.d415           →  "observation/wrist_image" (腕部相机 → left_wrist_0_rgb)
action                            →  "actions" (7D Cartesian delta, 无需额外delta变换)
prompt (optional)                 →  "prompt" (任务指令)
```

两个核心类：

- **`CR5AInputs`**：数据集 → 模型输入。负责：
  - 图像 float→uint8 转换、CHW→HWC 重排（`_parse_image`）
  - D435→`base_0_rgb`、D415→`left_wrist_0_rgb`、无右腕→填零
  - `image_mask` 标记真实/填充图像（pi0 需要，pi0-FAST 不需要）
  - 透传 `state`、`actions`、`prompt`

- **`CR5AOutputs`**：模型输出 → 动作。负责：
  - 从模型输出的 action chunk 中提取前 7 维（CR5A action dim）

#### 2. `config.py`（修改）

**A. 新增 `LeRobotCR5ADataConfig` 类**

放在 `LeRobotLiberoDataConfig` 和 `RLDSDroidDataConfig` 之间。继承 `DataConfigFactory`，实现 `create()` 方法：

- `repack_transform`：数据集 key → 模型 key 的重命名（`observation.state` → `observation/state`、`observation.images.d435` → `observation/image` 等）
- `data_transforms`：使用 `CR5AInputs` / `CR5AOutputs`
- **不需要** delta action 变换（CR5A 数据集 action 已经是 delta 格式）

**B. 注册 4 个训练 config**

在 `_CONFIGS` 列表末尾追加：

| config-name | 模型 | 预训练权重 | 特点 |
|---|---|---|---|
| `pi0_cr5a` | `Pi0Config()` | `pi0_base/params` | 完整微调 |
| `pi05_cr5a` | `Pi0Config(pi05=True)` | `pi0_base/params` | pi0.5 完整微调 |
| `pi0_cr5a_lora` | `Pi0Config(...lora)` | `pi0_base/params` | **LoRA 低显存微调** |
| `pi0_fast_cr5a` | `Pi0FASTConfig(action_dim=7, ...)` | `pi0_fast_base/params` | pi0-FAST 完整微调 |

**C. 新增 `root` 字段**

`DataConfig` 和 `DataConfigFactory` 各新增 `root: str | None = None` 字段，支持从 CLI 传入本地数据集根目录：

```bash
--data.repo-id=lerobot_dataset --data.root=/root/autodl-tmp/openpi/lerobot_dataset
```

> `--data.root` 指向**包含 `meta/` 和 `data/` 的数据集根目录**，不是父目录。

#### 3. `data_loader.py`（修改）

`create_torch_dataset()` 函数中，将 `root` 参数传递到 LeRobot 库：

```python
# 原来：只传 repo_id，LeRobot 去 HF Hub 下载
dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
dataset = lerobot_dataset.LeRobotDataset(data_config.repo_id, ...)

# 改后：传 repo_id + root，从本地 root/repo_id 加载
root = data_config.root
dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root)
dataset = lerobot_dataset.LeRobotDataset(data_config.repo_id, root=root, ...)
```

LeRobot 库的 `root` 参数应**直接指向数据集根目录**（即包含 `meta/` 和 `data/` 的文件夹）。当 `root=None` 时，`repo_id` 被当做 HF Hub 路径。当 `root` 不为 None 时，数据集从 `root` 本地加载，`repo_id` 仅用于命名。

### 数据流全景

```
本地磁盘: /root/autodl-tmp/openpi/lerobot_dataset/
  ├── meta/{info.json, stats.json, ...}
  └── data/chunk-000/episode_*.parquet
           ↓
LeRobotDataset(repo_id="lerobot_dataset", root="/root/autodl-tmp/openpi")
           ↓  (读取 parquet，返回 observation.state / observation.images.d435 / ...)
repack_transform: key 重命名
           ↓  (observation.images.d435 → observation/image, 等等)
CR5AInputs: 格式转换
           ↓  (float→uint8, CHW→HWC, 填零 right_wrist, 设置 image_mask)
pi0 模型 (base_0_rgb + left_wrist_0_rgb + state + prompt)
           ↓  (输出 action chunk)
CR5AOutputs: 提取前 7 维 action
           ↓
7D Cartesian delta action [dx,dy,dz,dRx,dRy,dRz,gripper]
```

### 训练命令

```bash
cd /root/autodl-tmp/openpi
source /root/miniconda3/etc/profile.d/conda.sh && conda activate openpi && source $HOME/.local/bin/env

# LoRA 微调（推荐，显存 < 24GB）
uv run python scripts/train.py \
  pi0_cr5a_lora \
  --data.repo-id=lerobot_dataset \
  --data.root=/root/autodl-tmp/openpi/lerobot_dataset \
  --batch-size=4 \
  --num-train-steps=10000 \
  --checkpoint-base-dir=./checkpoints/cr5a \
  --exp-name=cr5a_lora_v1 \
  --seed=42 \
  --no-wandb-enabled \
  --overwrite

# 完整微调（显存 ≥ 32GB）
uv run python scripts/train.py \
  pi0_cr5a \
  --data.repo-id=lerobot_dataset \
  --data.root=/root/autodl-tmp/openpi/lerobot_dataset \
  --batch-size=4 \
  --num-train-steps=20000 \
  --checkpoint-base-dir=./checkpoints/cr5a \
  --exp-name=cr5a_full_v1 \
  --seed=42
```

---

## BUG-010: AutoDL 服务器上 pi0_cr5a_lora 训练链路排障

**日期**：2026-07-01

### 现象

在 AutoDL 服务器 `/root/autodl-tmp/openpi` 中运行：

```bash
uv run python scripts/train.py \
  pi0_cr5a_lora \
  --data.repo-id=lerobot_dataset \
  --data.root=/root/autodl-tmp/openpi/lerobot_dataset \
  --batch-size=4 \
  --num-train-steps=10000 \
  --checkpoint-base-dir=./checkpoints/cr5a \
  --exp-name=cr5a_lora_v1 \
  --seed=42 \
  --no-wandb-enabled
```

先后遇到：

1. `RepositoryNotFoundError: https://huggingface.co/api/datasets/lerobot_dataset/refs`
2. `KeyError: "Column actions not in the dataset"`
3. `KeyError: 'prompt'`
4. `ValueError: Normalization stats not found`

### 根因

1. 本地 LeRobot 数据集 `meta/info.json` 记录了 51 个 episode，但实际缺少：

```text
data/chunk-000/episode_000008.parquet
data/chunk-000/episode_000009.parquet
```

LeRobot 检测到本地 episode 不完整后，会尝试从 Hugging Face Hub 拉取 `lerobot_dataset`，因此出现 401/Repository Not Found。

2. CR5A 数据集的原始 action 列名是单数 `action`，但 `DataConfig.action_sequence_keys` 默认是 `("actions",)`。`delta_timestamps` 在 repack 之前生效，所以必须读取原始列 `action`。

3. 当前 LeRobot 数据集没有独立 `prompt` 列，只有 `task_index` + `meta/tasks.jsonl`。需要开启 `prompt_from_task=True`，由 `PromptFromLeRobotTask` 将 task 转成 `prompt`。

4. OpenPI 训练必须有 `assets/pi0_cr5a_lora/lerobot_dataset/norm_stats.json`，否则 Normalize transform 会拒绝启动。

### 修改内容

#### 1. 新增数据集重编号脚本

新增：

```text
openpi/scripts/reindex_lerobot_dataset.py
```

作用：

- 跳过缺失 episode。
- 将旧 episode 索引压缩成连续编号。例如：

```text
old 000000 -> new 000000
...
old 000007 -> new 000007
old 000010 -> new 000008
old 000011 -> new 000009
...
old 000050 -> new 000048
```

- 同步改写 parquet 内部的 `episode_index`、`frame_index`、`index`。
- 修复旧 parquet 里 HuggingFace metadata 的 `_type: List`，改为当前 `datasets` 版本可识别的 `_type: Sequence`。
- 重写 `meta/info.json`、`episodes.jsonl`、`episodes_stats.jsonl`、`tasks.jsonl`、`stats.json`。

服务器运行：

```bash
cd /root/autodl-tmp/openpi

uv run python scripts/reindex_lerobot_dataset.py \
  --input-dir /root/autodl-tmp/openpi/lerobot_dataset \
  --output-dir /root/autodl-tmp/openpi/lerobot_dataset_reindexed
```

验证：

```bash
uv run python - <<'PY'
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("lerobot_dataset", root="/root/autodl-tmp/openpi/lerobot_dataset_reindexed")
print("episodes:", ds.num_episodes)
print("frames:", ds.num_frames)
print("ok")
PY
```

期望输出：

```text
episodes: 49
frames: 18367
ok
```

替换旧数据集：

```bash
mv /root/autodl-tmp/openpi/lerobot_dataset /root/autodl-tmp/openpi/lerobot_dataset_bad_missing_8_9
mv /root/autodl-tmp/openpi/lerobot_dataset_reindexed /root/autodl-tmp/openpi/lerobot_dataset
```

#### 2. 修复 CR5A action sequence key

在 `openpi/src/openpi/training/config.py` 的 `LeRobotCR5ADataConfig.create()` 中加入：

```python
action_sequence_keys=("action",),
```

最终片段：

```python
return dataclasses.replace(
    base,
    repack_transforms=repack_transform,
    data_transforms=data_transforms,
    model_transforms=model_transforms,
    action_sequence_keys=("action",),
    prompt_from_task=self.prompt_from_task,
)
```

#### 3. 修复 prompt 来源

将 `LeRobotCR5ADataConfig` 默认值改成：

```python
prompt_from_task: bool = True
```

原因：CR5A LeRobot parquet 中没有 `prompt` 列，任务文本在 `task_index` + `meta/tasks.jsonl` 中。

验证：

```bash
uv run python - <<'PY'
import openpi.training.config as c

cfg = c.get_config("pi0_cr5a_lora")
data = cfg.data.create(cfg.assets_dirs, cfg.model)
print("prompt_from_task:", data.prompt_from_task)
print("action_sequence_keys:", data.action_sequence_keys)
PY
```

期望输出：

```text
prompt_from_task: True
action_sequence_keys: ('action',)
```

### 服务器完整跑通步骤

#### 1. 检查数据集

```bash
cd /root/autodl-tmp/openpi

uv run python - <<'PY'
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("lerobot_dataset", root="/root/autodl-tmp/openpi/lerobot_dataset")
print("episodes:", ds.num_episodes)
print("frames:", ds.num_frames)
print("ok")
PY
```

已验证输出：

```text
episodes: 49
frames: 18367
ok
```

#### 2. 下载 pi0_base 预训练权重

```bash
cd /root/autodl-tmp/openpi
export OPENPI_DATA_HOME=/root/autodl-tmp/openpi/openpi_cache

uv run python - <<'PY'
from openpi.shared import download

path = download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params")
print("Downloaded to:", path)
PY
```

已验证下载到：

```text
/root/autodl-tmp/openpi/openpi_cache/openpi-assets/checkpoints/pi0_base/params
```

#### 3. 生成 norm stats

`scripts/compute_norm_stats.py` 默认不能传 `--data.root`，所以用下面的内联脚本覆盖 config：

```bash
cd /root/autodl-tmp/openpi

uv run python - <<'PY'
import dataclasses
import numpy as np
import tqdm

import openpi.training.config as _config
import openpi.shared.normalize as normalize
from scripts import compute_norm_stats as cns

config = _config.get_config("pi0_cr5a_lora")
config = dataclasses.replace(
    config,
    data=dataclasses.replace(
        config.data,
        repo_id="lerobot_dataset",
        root="/root/autodl-tmp/openpi/lerobot_dataset",
    ),
    batch_size=4,
    num_workers=0,
)

data_config = config.data.create(config.assets_dirs, config.model)
data_loader, num_batches = cns.create_torch_dataloader(
    data_config,
    config.model.action_horizon,
    config.batch_size,
    config.model,
    config.num_workers,
)

stats = {key: normalize.RunningStats() for key in ["state", "actions"]}
for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
    for key in stats:
        stats[key].update(np.asarray(batch[key]))

norm_stats = {key: stats[key].get_statistics() for key in stats}
output_path = config.assets_dirs / data_config.repo_id
print("Writing stats to:", output_path)
normalize.save(output_path, norm_stats)
PY
```

确认生成：

```bash
ls -lh /root/autodl-tmp/openpi/assets/pi0_cr5a_lora/lerobot_dataset/norm_stats.json
```

#### 4. 启动 LoRA 训练

```bash
cd /root/autodl-tmp/openpi
export OPENPI_DATA_HOME=/root/autodl-tmp/openpi/openpi_cache

uv run python scripts/train.py \
  pi0_cr5a_lora \
  --data.repo-id=lerobot_dataset \
  --data.root=/root/autodl-tmp/openpi/lerobot_dataset \
  --batch-size=4 \
  --num-train-steps=10000 \
  --checkpoint-base-dir=./checkpoints/cr5a \
  --exp-name=cr5a_lora_v1 \
  --seed=42 \
  --no-wandb-enabled \
  --overwrite
```

### 注意事项

- `--data.root` 对当前服务器上的 LeRobot 版本，应指向**包含 `meta/` 和 `data/` 的数据集根目录**：

```text
/root/autodl-tmp/openpi/lerobot_dataset
  ├── meta/
  └── data/
```

- `uv run python - <<'PY' ... PY` 的结束标记必须是单独一行 `PY`，否则 shell 会继续等待输入，后续命令可能被喂进 Python 导致 `SyntaxError`。
- 训练目录已存在时加 `--overwrite`；继续上次训练才用 `--resume`，两者不能同时使用。

### openpi `scripts/train.py` CLI 参数踩坑记录

这些问题与业务逻辑无关，纯粹是 tyro CLI 框架的参数格式要求：

**1. config 名称是位置参数，不是 `--config-name`**

```bash
# 错误：
--config-name=pi0_cr5a_lora

# 正确：放在第一个位置参数
uv run python scripts/train.py pi0_cr5a_lora ...
```

**2. 参数名用连字符 `-`，不是下划线 `_`**

tyro 自动把 dataclass 字段名 `batch_size` 转成 `--batch-size`：

```bash
# 错误：
--batch_size=4  --num_train_steps=10000

# 正确：
--batch-size=4  --num-train-steps=10000
```

特殊情况：嵌套 dataclass 的字段用 `.` 连接，如 `--data.repo-id`。

**3. boolean 开关用 `--no-xxx` 而不是 `--xxx=false`**

```bash
# 错误：
--wandb-enabled=false

# 正确：
--no-wandb-enabled
```

**4. `--exp-name` 是必填的**

不填 tyro 会报 `Required options: --exp-name`。

**5. `--checkpoint-base-dir` 不是 `--checkpoint-dir`**

dataclass 字段是 `checkpoint_base_dir`，tyro 转成 `--checkpoint-base-dir`。

**6. `--overwrite` 每次失败后都需要**

第一次运行成功后创建了 checkpoint 目录。之后参数调整（如 `--data.root` 路径修正）再次运行时必须加 `--overwrite`，否则报：

```text
FileExistsError: Checkpoint directory .../cr5a_lora_v1 already exists.
```

清除命令：`rm -rf ./checkpoints/cr5a` 或加 `--overwrite`。

**7. `export OMP_NUM_THREADS=1`**

服务器上需要设置，否则会打印 `libgomp: Invalid value for environment variable OMP_NUM_THREADS`。虽然只是 warning 不影响运行，但每次都会看到。

**8. `source /etc/network_turbo` 网络加速**

AutoDL 服务器访问 huggingface.co / github.com 可能超时。训练前先执行：

```bash
source /etc/network_turbo
```

注意：这会**降低**访问 pip 源等国内资源的速度。训练完成或不需 HF 访问时可以关掉。

**9. 必须使用 `uv run python` 而非 `python`**

- `conda activate openpi` 后直接 `python` 用的是 conda 的 Python，**没有** `.venv` 里的 lerobot / openpi 等依赖
- `uv run python` 才会启用 `.venv` 中的完整依赖环境

---

## 下一次训练：完整操作清单

以下步骤按顺序执行，每步验证通过后再进行下一步：

### Step 1: 激活环境

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate openpi
source $HOME/.local/bin/env
source /etc/network_turbo        # AutoDL 网络加速
export OPENPI_DATA_HOME=/root/autodl-tmp/openpi/openpi_cache
export OMP_NUM_THREADS=1
cd /root/autodl-tmp/openpi
```

### Step 2: 验证数据集完整性

```bash
uv run python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('lerobot_dataset', root='/root/autodl-tmp/openpi/lerobot_dataset')
print(f'episodes={ds.num_episodes} frames={ds.num_frames}')
# 期望: episodes=49 frames=18367
"
```

### Step 3: 验证 config 参数（只需首次或修改 config 后）

```bash
uv run python -c "
import openpi.training.config as c
cfg = c.get_config('pi0_cr5a_lora')
data = cfg.data.create(cfg.assets_dirs, cfg.model)
assert data.prompt_from_task == True, 'prompt_from_task 应为 True'
assert data.action_sequence_keys == ('action',), 'action key 应为 (action,)'
print('config OK')
"
```

### Step 4: 下载预训练权重（只需首次）

```bash
uv run python -c "
from openpi.shared import download
path = download.maybe_download('gs://openpi-assets/checkpoints/pi0_base/params')
print('weights at:', path)
"
# 已缓存在 openpi_cache/，后续运行自动跳过下载
```

### Step 5: 生成 norm stats（只需首次，数据集不变则跳过）

```bash
uv run python - "
import dataclasses, numpy as np, tqdm
import openpi.training.config as _config
import openpi.shared.normalize as normalize
from scripts import compute_norm_stats as cns

config = _config.get_config('pi0_cr5a_lora')
config = dataclasses.replace(config,
    data=dataclasses.replace(config.data,
        repo_id='lerobot_dataset',
        root='/root/autodl-tmp/openpi/lerobot_dataset'),
    batch_size=4, num_workers=0)

data_config = config.data.create(config.assets_dirs, config.model)
dl, n = cns.create_torch_dataloader(data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers)
stats = {k: normalize.RunningStats() for k in ['state', 'actions']}
for batch in tqdm.tqdm(dl, total=n, desc='norm stats'):
    for k in stats: stats[k].update(np.asarray(batch[k]))
norm_stats = {k: stats[k].get_statistics() for k in stats}
normalize.save(config.assets_dirs / data_config.repo_id, norm_stats)
print('done →', config.assets_dirs / data_config.repo_id)
"
```

### Step 6: 启动训练

```bash
rm -rf ./checkpoints/cr5a   # 只在新训练时需要，resume 不用删

uv run python scripts/train.py \
  pi0_cr5a_lora \
  --data.repo-id=lerobot_dataset \
  --data.root=/root/autodl-tmp/openpi/lerobot_dataset \
  --batch-size=4 \
  --num-train-steps=10000 \
  --checkpoint-base-dir=./checkpoints/cr5a \
  --exp-name=cr5a_lora_v1 \
  --seed=42 \
  --no-wandb-enabled \
  --overwrite
```

### Step 7: 恢复上次训练（从 checkpoint 继续）

```bash
uv run python scripts/train.py \
  pi0_cr5a_lora \
  --data.repo-id=lerobot_dataset \
  --data.root=/root/autodl-tmp/openpi/lerobot_dataset \
  --batch-size=4 \
  --num-train-steps=20000 \
  --checkpoint-base-dir=./checkpoints/cr5a \
  --exp-name=cr5a_lora_v1 \
  --seed=42 \
  --no-wandb-enabled \
  --resume
```

### 常见错误速查

| 错误 | 原因 | 解决 |
|---|---|---|
| `FileExistsError: checkpoint ... already exists` | 上次创建了目录 | `rm -rf ./checkpoints/cr5a` 或加 `--overwrite` |
| `RepositoryNotFoundError: lerobot_dataset` | 数据集路径不对或缺少 episode | Step 2 验证 |
| `KeyError: "Column actions not in the dataset"` | `action_sequence_keys` 配置错误 | `config.py` 中应为 `("action",)` |
| `KeyError: 'prompt'` | `prompt_from_task` 未开启 | `config.py` 中 `prompt_from_task: bool = True` |
| `ValueError: Normalization stats not found` | 未生成 norm_stats.json | 执行 Step 5 |
| `ConnectTimeout / HTTPSConnectionPool` | 网络不通 | `source /etc/network_turbo` |
| `Unrecognized options: --batch_size` | tyro 参数格式错误 | 用 `--batch-size`（连字符，非下划线） |
| `Required options: --exp-name` | 缺少必填参数 | 加上 `--exp-name=xxx` |

---

## 为什么需要这些步骤？（原理说明）

这部分解释训练链路中每个环节存在的**原因**，而不只是操作步骤。理解原理后，换数据集、换机器人、换模型时就能自己推导。

### 为什么需要预训练权重？

pi0 是一个在**多机器人、多任务**数据上预训练过的大模型（~12GB 参数量）。它已经学会了：

- 从图像中理解物体位置、形状、姿态
- "抓取""放置""推动"等基础操作概念
- 语言指令与动作之间的对应关系

如果从零训练（随机初始化），需要百万级别的数据量和数周的训练时间。**微调（fine-tuning）**只需在预训练权重基础上，用你的 CR5A 数据（~18000 帧）调整模型，让它适配你的机械臂的运动空间和相机视角。

不下载预训练权重直接训练 = 随机初始化 = 模型什么都不懂 = 训练不会收敛。

### 为什么需要 norm stats？

深度学习模型对输入数值范围敏感。pi0 模型在预训练时，所有输入数据都被**标准化**到均值 0、标准差 1 的范围。

你的 CR5A 数据：

```
state:  J1≈-93°（均值），范围 -116°~-73°
action: dx≈0.2mm（均值），范围 -1.0~1.0mm
```

如果不做标准化，模型收到的关节角是 -93，而预训练时见过的数据范围完全不一样 → 模型"看不懂" → Loss 不降。

`norm_stats.json` 记录了你的数据每个维度的 min/max/mean/std/q01/q99。训练时 openpi 自动用这些统计量做归一化（z-score 或 quantile normalization），推理时也同样归一化 → 保持和预训练一致的输入分布。

### 为什么需要 `action_sequence_keys=("action",)` 而不是 `("actions",)`？

openpi 训练时会从数据集中读取**连续多帧 action** 组成一个 action chunk（默认 50 帧）。

`LeRobotDataset` 用 `delta_timestamps` 参数来指定"从当前帧往后取多少帧的 action"。它按 `action_sequence_keys` 中指定的**列名**去 parquet 里读取。

你的 parquet 中 action 列名叫 `action`（单数），不是 `actions`（复数）。如果不改，openpi 在 repack transform 之前就去读 `actions` 列 → `KeyError: Column actions not in the dataset`。

### 为什么需要 `prompt_from_task=True`？

pi0 是**语言条件**模型——它需要知道"我当前的任务是什么"才能生成正确的动作。

你的 CR5A 数据集中没有独立的 `prompt` 列，任务文本存在 `task_index`（数字）+ `meta/tasks.jsonl`（task_index → "pick the object" 的映射）中。

`prompt_from_task=True` 告诉 openpi：从 LeRobot 的 task 字段自动生成 prompt，而不是从 `prompt` 列读取。

如果不设置：repack transform 试图读取 `prompt` 列 → `KeyError: 'prompt'`。

### 为什么需要 repack transform（key 名映射）？

pi0 模型的 `CR5AInputs` 策略期望的输入 key 是固定的：

```
observation/image          ← 场景相机
observation/wrist_image    ← 腕部相机
observation/state          ← 关节状态
actions                    ← 动作
prompt                     ← 语言指令
```

但你的 LeRobot 数据集的实际 key 是：

```
observation.images.d435    ← 场景相机（命名规则不同）
observation.images.d415    ← 腕部相机（命名规则不同）
observation.state          ← 关节状态
action                     ← 动作（单复数不同）
（无独立 prompt 列）
```

`repack_transform` 就是一个**重命名表**，在数据进入模型之前把 key 统一掉。不改模型代码，只改配置。

### 为什么用 LoRA 而不是完整微调？

| | LoRA | 完整微调 |
|---|---|---|
| 可训练参数量 | ~1% | 100% |
| 显存占用 (24GB 卡) | ~15GB | >24GB (OOM) |
| 训练速度 | 快 | 慢 |
| 效果 | 接近完整微调 | 理论上最好 |

LoRA 的原理：冻结原始权重 `W`，只训练两个小矩阵 `A` 和 `B`，使得 `W' = W + A·B`。因为 `A·B` 参数量极小，显存和计算量大幅降低。

从你的训练日志可以看到模型加载了很多 `lora_a` / `lora_b` 参数——这些就是微调时唯一在更新的部分。

### 为什么新建 `cr5a_policy.py` 而不是复用 Libero 或 Aloha 的？

每个机器人的数据格式不同，不能直接共用：

| | Libero | Aloha | CR5A（我们） |
|---|---|---|---|
| state 维度 | 8D | 14D（双臂×7） | **7D**（单臂） |
| action 维度 | 7D | 14D | **7D** |
| 相机数量 | 2 个 | 1 个 | **2 个** |
| action 类型 | delta | delta（需 adapter 转换） | **delta** |
| 图像 key 名 | `observation/image` | `observation.images.top` | `observation.images.d435/d415` |

如果硬套 Libero 的 policy，它会去读 `observation/image` 这个 key → 你的数据里没有 → `KeyError`。

如果硬套 Aloha 的 policy，它会期望 14D action、7D×2 state、关节空间转换 → 你的 7D Cartesian delta 全对不上。

所以必须新建一个 CR5A 专用的 policy，只做最小的事情：key 重命名 + 格式转换。

### 为什么 CR5AInputs 要填充一个 `right_wrist_0_rgb` 的全零图像？

pi0 模型在预训练时**固定了输入结构**：必须有 3 个图像输入槽位——`base_0_rgb`（场景）、`left_wrist_0_rgb`（左腕）、`right_wrist_0_rgb`（右腕）。

CR5A 只有 2 个相机（D435 场景 + D415 腕部），没有右腕相机。如果直接不填 `right_wrist_0_rgb`，模型输入维度不匹配 → 报错。

做法：用 `np.zeros_like(base_image)` 生成一张全黑图填充 `right_wrist_0_rgb`，然后通过 `image_mask` 标记它为 `False`（pi0 不关注这个槽位）。pi0-FAST 的处理略有不同，mask 设为 `True`。

### 为什么 `image_mask` 需要区分 pi0 和 pi0-FAST？

- **pi0**：使用 `image_mask=False` 告诉注意力机制"忽略这个图像槽位" → 不消耗计算量
- **pi0-FAST**：不使用 image_mask，所以即使是无意义的零图也标记为 `True`

这是模型架构层面的差异，不需要我们手动处理——`CR5AInputs` 根据 `model_type` 自动选择正确的方式。

### 为什么要改 `config.py`？不能直接在命令行传入所有参数吗？

不能。openpi 的训练入口 `scripts/train.py` 依赖 `_CONFIGS` 注册表来选择数据集配置：

```python
# config.py
_CONFIGS = [
    TrainConfig(name="pi0_aloha", data=LeRobotAlohaDataConfig(...), ...),
    TrainConfig(name="pi0_libero", data=LeRobotLiberoDataConfig(...), ...),
    TrainConfig(name="pi0_cr5a_lora", data=LeRobotCR5ADataConfig(...), ...),  # 我们加的
]
```

如果不在 `_CONFIGS` 中注册，`scripts/train.py` 根本不知道 CR5A 数据集长什么样。命令行 `--data.repo-id` 只是告诉它"去哪里找数据"，但"数据进来后怎么处理"必须由 config 定义。

换句话说：`--data.repo-id` 控制**数据的来源**，`DataConfig` 控制**数据的处理流水线**。

### 为什么 `DataConfig` 要加 `root` 字段？

openpi 原始设计只支持从 HuggingFace Hub 加载数据：

```python
# 原始流程：repo_id 必须是 "namespace/dataset_name"
dataset = LeRobotDataset(repo_id="physical-intelligence/libero")
# → 自动从 huggingface.co 下载
```

但我们的数据在**本地磁盘**。LeRobot 库支持 `root` 参数指定本地路径，但 openpi 没有暴露这个参数。所以在 `DataConfig` 和 `DataConfigFactory` 各加一个 `root: str | None = None` 字段，让它能一路传递到 `LeRobotDataset`。

如果不加，openpi 会把 `repo_id="lerobot_dataset"` 当成 HuggingFace 仓库名 → 去 `huggingface.co/api/datasets/lerobot_dataset` 查 → 404 → 崩溃。

### 为什么用 LeRobot 格式而不是直接读 parquet？

LeRobot v2.1 格式不只是"parquet 文件"，它还提供了：

1. **`meta/info.json`**：数据集元信息（帧率、feature 定义、episode 数量）→ openpi 自动读取，不需要手动指定 action_dim、fps
2. **`meta/stats.json`**：预计算好的归一化统计 → openpi 自动加载做 Normalize
3. **`meta/tasks.jsonl`**：task_index → 文本任务的映射 → `prompt_from_task` 自动生成 prompt
4. **`delta_timestamps`**：LeRobot 库自带"从当前帧往后取 N 帧 action"的功能 → 自动构建 action chunk

如果直接读裸 parquet，上面所有功能都要自己实现。LeRobot 格式就是 openpi 的"标准接口"。

---

## 训练参数详解

以下按 `scripts/train.py` 中出现的顺序解释每个参数的含义、默认值和调优建议。

### CLI 参数

#### `--batch-size`（默认 32）

**每步训练用多少帧数据**。GPU 同时处理 batch_size 帧，计算一次梯度更新。

- 设大了 → 显存不够 OOM
- 设小了 → 梯度噪声大、收敛慢
- 24GB 显存 LoRA 微调建议 **4**
- 完整微调建议 **2**（如果 OOM 降到 1）

> `batch_size` 是**全局** batch size。单卡训练时 `local_batch_size = batch_size`。多卡时会被均分。

#### `--num-train-steps`（默认 30000）

**训练总步数**。每一步 = 处理一个 batch。总帧数 = batch_size × num_train_steps。

- 你的数据集 18367 帧 / batch_size=4 ≈ 4591 步 = **1 个 epoch**
- 10000 步 ≈ 2.2 个 epoch，对微调来说足够
- 如果想多练几轮，设 20000~30000

#### `--checkpoint-base-dir`（默认 `./checkpoints`）

**checkpoint 保存的根目录**。实际保存路径为：

```text
{checkpoint-base-dir}/{config-name}/{exp-name}/
例: ./checkpoints/cr5a/pi0_cr5a_lora/cr5a_lora_v1/
```

#### `--exp-name`（必填）

**实验名称**，用于区分不同训练配置的结果。建议用有意义的名字：

```bash
--exp-name=cr5a_v2_bs8_20k   # 版本2, batch_size=8, 20000步
```

#### `--seed`（默认 42）

**随机种子**。固定后训练结果可复现。改数据、改参数后最好换个 seed。

#### `--no-wandb-enabled`

**关闭 Weights & Biases 日志**。服务器不能联网时关掉。本地有 W&B 账号的话开着可以看实时 loss 曲线。

#### `--overwrite` / `--resume`

- `--overwrite`：删除已有 checkpoint 重新开始
- `--resume`：从上次 checkpoint 继续训练（步数从上次结束位置开始）
- 两者互斥，不能同时用

### 学习率参数（`--lr-schedule.*`）

#### `--lr-schedule.peak-lr`（默认 2.5e-5）

**学习率峰值**。控制每次梯度更新的步长：

- 太大 → Loss 震荡甚至发散
- 太小 → 收敛极慢
- pi0 微调建议 **2e-5 ~ 5e-5**，默认 2.5e-5 适用大多数情况

#### `--lr-schedule.warmup-steps`（默认 1000）

**预热步数**。训练前 N 步学习率从 0 线性增长到 peak-lr。

为什么需要预热？模型刚开始训练时梯度方向很不稳定，直接上大学习率容易把预训练权重"炸飞"。预热期让模型慢慢适应新数据。

#### `--lr-schedule.decay-steps`（默认 30000）

**衰减步数**。学习率从 peak-lr 余弦衰减到 decay-lr 的总步数。

训练后期需要小学习率做精细调整，所以随着训练进行逐步降低学习率。

#### `--lr-schedule.decay-lr`（默认 2.5e-6）

**最终学习率**。衰减结束时的学习率，是 peak-lr 的 **1/10**。

### 学习率变化曲线

```
lr
^
|     /\
|    /  \
|   /    \_________
|  /              \
| /                \
|/                  \
+--|----|-----------|----> step
   0   warmup     decay
```

- 0 ~ warmup_steps：线性增长 0 → peak_lr
- warmup_steps ~ decay_steps：余弦衰减 peak_lr → decay_lr
- decay_steps 之后：保持 decay_lr

### 优化器参数（`--optimizer.*`）

#### `--optimizer.weight-decay`（默认 1e-10）

**权重衰减（L2 正则化）**。防止模型过拟合。pi0 默认值极小（几乎为零），因为微调数据量少，过强的正则化反而不好。

#### `--optimizer.clip-gradient-norm`（默认 1.0）

**梯度裁剪**。每个训练步，如果梯度的 L2 范数超过这个值，就按比例缩回去。

防止某一步梯度突然爆炸（常见于数据中有极端样本）→ 参数突变 → 之前学到的全忘光。

### 数据和模型参数

#### `--data.repo-id`（必填）

**数据集标识**。可以是 HuggingFace 仓库名（如 `physical-intelligence/libero`）或本地目录名（如 `lerobot_dataset`）。配合 `--data.root` 使用。

#### `--data.root`

**数据集本地根目录**。不为 None 时优先从本地加载，跳过 HuggingFace Hub。

#### `--data.prompt-from-task`

**是否从 LeRobot task 字段生成 prompt**。你的数据集没有独立 `prompt` 列，必须设为 `true`。

#### `--model.action-dim`（默认 32）

**模型 action 维度**。注意这是模型的**内部**维度，不是你数据集的 7D。

pi0 内部用 32 维来表示任意机器人的 action。训练时 `PadStatesAndActions` 自动把你的 7D action 填充到 32D，推理时 `CR5AOutputs` 再取回 7D。**不需要改这个参数**。

#### `--model.action-horizon`（默认 50）

**动作预测长度（action chunk）**。模型一次输出未来 50 步的动作指令。

- 50 步 × 15fps ≈ 3.3 秒的动作序列
- 值越大，模型"看得越远"，但计算量也越大
- CR5A 用默认 50 即可

### 保存和日志参数

#### `--log-interval`（默认 100）

**每隔多少步打印一次 loss**。训练日志中每隔 100 步看到的 `Step 100: loss=0.2632` 就是这样来的。

#### `--save-interval`（默认 1000）

**每隔多少步保存一次 checkpoint**。checkpoint 保存到 `{checkpoint-base-dir}/{config-name}/{exp-name}/`。

#### `--keep-period`（默认 5000）

**保留策略**：只保留 step % 5000 == 0 的 checkpoint。旧的会被自动清理，节省磁盘空间。

### 完整训练命令参数对照

```bash
uv run python scripts/train.py \
  pi0_cr5a_lora \                          # config 名称（位置参数，不是 --config-name）
  --data.repo-id=lerobot_dataset \         # 数据集 ID
  --data.root=/root/autodl-tmp/openpi/lerobot_dataset \  # 本地路径
  --data.prompt-from-task=true \           # 从 task 字段生成 prompt（你的数据必须开）
  --batch-size=4 \                         # 单步帧数（24GB 卡 LoRA 建议 4）
  --num-train-steps=10000 \                # 总训练步数（2.2 epoch）
  --checkpoint-base-dir=./checkpoints/cr5a \  # checkpoint 根目录
  --exp-name=cr5a_lora_v1 \                # 实验名称（必填）
  --seed=42 \                              # 随机种子
  --no-wandb-enabled \                     # 关 W&B
  --overwrite                              # 覆盖旧 checkpoint（或 --resume）
```
