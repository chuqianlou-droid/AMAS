# CR5A 机械臂遥操作 BUG 记录

> 本文档记录**机械臂遥操作控制**相关 BUG（ServoJ/ServoP、Quest3 映射、碰撞检测等）。
> pi0 训练文档见 [`PI0_TRAINING.md`](./PI0_TRAINING.md)。

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
