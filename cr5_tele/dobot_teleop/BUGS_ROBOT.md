# CR5A 机械臂遥操作 BUG 记录

> 本文档记录**机械臂遥操作控制**相关 BUG（ServoJ/ServoP、Quest3 映射、碰撞检测等）。
> pi0 训练文档见 [`PI0_TRAINING.md`](./PI0_TRAINING.md)。

## 约定

- 每条 BUG 记录必须包含：**现象、触发条件、根因（带代码引用）、修复方案、验证方法**
- 按时间倒序排列（最新的在最上面）
- 排查新 BUG 时先查本文档是否有类似问题

---

## BUG-008: LeRobot gripper 标签仍使用 Quest trigger 而不是真实夹爪状态

**日期**：2026-07-05

### 现象

- BUG-007 已经把 action 前 6 维改成真实机械臂轨迹差分，但第 7 维 gripper 仍可能来自 Quest 右扳机。
- Quest/teleop 有 300-400ms 延迟时，画面里的夹爪实际还没闭合，训练标签却可能已经是“闭合”。
- policy 上机后容易表现为提前夹取、空夹，尤其在接近物体阶段更明显。

### 触发条件

1. 遥操作端通过 Quest trigger 控制 PGE 夹爪。
2. 录制端使用旧逻辑：

```python
grippers.append(0.0)
gripper_actions.append(sample.gripper_command)
actions[:, 6] = gripper_commands
```

其中 `sample.gripper_command` 是遥操作命令，不是 PGE 夹爪当前位置。

### 根因

CR5A feedback 端口 `30004` 能实时读取 `ToolVectorActual/QActual`，但不包含 PGE 夹爪位置。录制端不能再单独占用 Dashboard `29999` 去开第二个 Modbus 连接，因为遥操作端已经用它控制机械臂和夹爪。

因此真实夹爪状态必须从已经持有 Modbus 连接的遥操作端读取，然后随 UDP action stream 一起发给录制端。

### 修复方案

在 `toolframe_governor_teleop.py` 的 `PgeModbusGripper` 中增加 PGE 反馈寄存器：

```python
REG_GRIP_STATUS = 0x0201
REG_CURRENT_POSITION = 0x0202
```

遥操作端每隔 `--gripper-state-read-interval-s` 从 `0x0202` 读取当前位置，并换算为：

```text
0.0 = 张开
1.0 = 闭合
```

然后通过 `TeleopAction.gripper_state` 发布到 UDP。

录制端改为：

```python
observation.state[6] = sample.gripper_state
action[i, 6] = gripper_state[i + 1]
```

也就是说：

- 当前观测里的 gripper 是当前真实夹爪状态。
- 训练 action 的 gripper 是下一帧真实夹爪状态，和前 6 维“当前帧到下一帧”的真实运动标签保持一致。
- 如果旧遥操作端没有发布 `gripper_state`，录制端会兼容性退回 `gripper_command`，但新采集必须使用更新后的遥操作端。

### 验证方法

- 启动遥操作端后日志应出现 `gripper_state=...`。
- 张开时 `gripper_state` 应接近 0，闭合时应接近 1。
- 录制端日志每 30 帧会打印 `gripper_state=...`。
- 新保存的数据中 `observation.state[:, 6]` 和 `action[:, 6]` 应随真实夹爪运动变化，不应再固定为 0 或直接等于 Quest trigger 时间线。

### 回退方法

如需严格回退到旧 trigger 行为：

1. 删除 `TeleopAction.gripper_state` 字段及发布逻辑。
2. 将 `record_cr5a_pi0_dataset.py` 中的：

```python
grippers.append(sample.gripper_state)
gripper_states.append(sample.gripper_state)
actions_arr = _feedback_delta_actions(gripper_center_poses, gripper_states)
```

改回：

```python
grippers.append(0.0)
gripper_actions.append(sample.gripper_command)
actions_arr = _feedback_delta_actions(gripper_center_poses, gripper_actions)
```

3. 将 `_feedback_delta_actions()` 的第 7 维改回 `actions[:, 6] = gripper_commands`。

---

## BUG-007: LeRobot action 标签仍可能使用遥操作命令增量导致时间错位

**日期**：2026-07-05

### 现象

- BUG-006 修复后，图像和 `observation.state` 已经改为真实反馈对齐。
- 但如果 `action` 仍直接保存遥操作 UDP 中的命令 delta，Quest/teleop 300-400ms 延迟仍可能污染训练标签。
- 结果可能仍表现为 policy 提前停止、提前夹取或对真实画面反应不稳定。

### 触发条件

1. 录制端使用 `--action-source teleop`。
2. 旧逻辑中直接执行：

```python
actions.append(validate_cr5a_action(sample.action))
```

这里的 `sample.action` 来自遥操作端：

```python
action = pose_action_delta_mm_deg(prev_cmd_target, cmd_target) + [gripper_cmd]
```

### 根因

遥操作命令 delta 属于控制端时间线，不一定与录制端当前相机帧和当前真实机器人反馈严格同步。

BUG-006 已经把状态改为：

```text
image_t -> ToolVectorActual_t / QActual_t
```

但旧 action 仍可能是：

```text
teleop_command_t
```

如果 Quest/teleop 链路延迟明显，训练样本仍可能变成：

```text
真实 image/state_t -> 时间错位的 command action
```

### 修复方案

保存 episode 时不再使用 `sample.action` 作为训练 action。改为根据录制到的真实 gripper_center 位姿序列生成 action：

```python
action[i, :6] = pose_delta(gripper_center_pose[i], gripper_center_pose[i + 1])
action[i, 6] = gripper_state[i + 1]
```

其中：

- 前 6 维：真实机械臂反馈轨迹的相对位姿差分。
- 第 7 维：BUG-008 修复后使用 PGE 实际夹爪状态，不再使用 Quest trigger 命令。
- 最后一帧没有下一帧，位姿 delta 记为 0。

### 验证方法

- 录制时日志仍会打印 `teleop_action=...`，但这只是调试参考，不再是保存到 LeRobot 的训练 action。
- 保存后检查 `Nonzero action ratio` 应来自真实运动轨迹差分。
- 新数据集中的 `action` 前 6 维应对应相邻两帧 `gripper_center` 真实位姿变化。

### 回退方法

如需回到旧行为，将 `record_cr5a_pi0_dataset.py` 保存前的：

```python
actions_arr = _feedback_delta_actions(gripper_center_poses, gripper_actions)
```

改回实时循环中保存：

```python
actions.append(validate_cr5a_action(sample.action))
```

## BUG-006: 采集数据的图像与机械臂状态可能不同步（Quest 控制参考污染）

**日期**：2026-07-05

### 现象

- 使用 Quest3 遥操作采集 LeRobot 数据后，pi0/LoRA 推理时容易出现提前停止、提前夹取。
- 两个相机画面和训练视频本身看起来正常，但模型在真实机械臂上接近物体时会空夹。
- Quest3 遥操作链路估计有 300-400ms 延迟，约等于 15Hz 采集下的 4.5-6 帧。
- 初版改为 Dashboard `GetPose/GetAngle` 后，录制端报错：`Connection refused, IP:Port has been occupied`，因为遥操作端已经占用 29999。

### 触发条件

1. 遥操作端发布 `--publish-action-stream`。
2. 录制端使用 `record_cr5a_pi0_dataset.py --action-source teleop`。
3. 旧默认 `--teleop-state-source stream` 时，录制端使用遥操作 UDP 包里的 `current_pose/current_joints` 作为训练状态。

### 根因

`stream` 模式下，数据集中的状态不是录制时刻机械臂的物理反馈，而是遥操作进程里的控制参考：

```python
# toolframe_governor_teleop.py
current_pose=tuple(prev_cmd_target)
target_pose=tuple(cmd_target)
current_joints=tuple(last_sent_joints)
```

录制脚本随后把这些 stream state 与相机帧配对：

```python
# record_cr5a_pi0_dataset.py 旧逻辑
raw_tcp = sample.controller_pose
joints = sample.controller_joints
```

当 Quest/teleop 链路有 300-400ms 延迟时，图像可能是真实夹爪还没到物体的画面，但 state/action/gripper label 已经来自更靠前的控制参考。模仿学习会把这种错误配对学成“看到还没到位就停止或闭合夹爪”。

### 修复方案

保留原有采集操作逻辑：action/deadman/drop/camera 采集流程不变。

只修改训练状态来源：

- `--teleop-state-source` 默认从 `stream` 改为 `feedback`。
- 每帧成功取到 RealSense 图像后，立即通过 Dobot feedback 端口 30004 读取真实机械臂反馈：

```python
d435, d415 = camera.get_rgb_images()
image_timestamp_s = time.time()
raw_tcp = np.asarray(feedback["ToolVectorActual"], dtype=np.float32)
joints = np.asarray(feedback["QActual"], dtype=np.float32)
```

- 保存的 `timestamps` 使用相机帧取到后的时间 `image_timestamp_s`，而不是完成状态读取后的更晚时间。
- `stream` 模式保留为 legacy fallback，但不再作为默认训练采集方式。
- `dashboard` 模式保留为显式调试选项，但真实遥操作采集不要使用它，因为 29999 通常已经被控制端占用。

### 验证方法

- 启动录制时确认终端打印：

```text
Teleop state source: feedback
State/image alignment: camera frame -> immediate live ToolVectorActual/QActual sample
```

- 录制时不再显式传入 `--teleop-state-source stream`。
- 录制时也不要显式传入 `--teleop-state-source dashboard`，除非终端 1 没有占用 29999。
- 重新采集少量 episode 后，重点检查夹爪闭合帧：画面中夹爪应已经接近/接触物体，而不是明显提前闭合。

### 回退方法

如需回到旧行为，在录制命令中显式加入：

```bash
--teleop-state-source stream
```

或将 `record_cr5a_pi0_dataset.py` 中 `--teleop-state-source` 默认值改回 `stream`。

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
