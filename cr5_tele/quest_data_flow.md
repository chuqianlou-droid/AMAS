# Quest 数据到机器人的完整流程

## 整体流程图

```
Quest 3/3s
  ⬇ WiFi/UDP
OculusReader → 4x4 矩阵 + 按键
  ⬇
坐标系变换 (Oculus→机器人)
  ⬇
缩放 (×2.5, ×2.0) + 通道符号
  ⬇
[可选] Placo IK 求解 → 7关节角
  ⬇
Teleop.get_action()
  ⬇
record_loop()
  ⬇
Franka.send_action()
  ⬇ ZeroRPC (TCP)
NUC 上的 FrankaInterfaceServer
  ⬇ Polymetis
Franka Research 3 机器人
```

---

## 1️⃣ 获取 Quest 原始数据

**文件:** `lerobot_teleoperator_franka/.../oculus/oculus_robot.py`，`get_action()` 方法 (第 259 行)

```python
transforms, buttons = self._oculus_reader.get_transformations_and_buttons()
```

通过 WiFi/UDP 从 Quest 拿到：

- 右手控制器的 **6-DOF 位姿**（4x4 变换矩阵）
- 按键状态：`RG`（握把）、`A` 键、`rightTrig`（扳机）

---

## 2️⃣ 坐标系变换（Oculus → 机器人）

**`_compute_delta_pose()`** (第 155 行) — 最关键的处理步骤之一

Oculus 和机器人的坐标系不同，需要映射：

| 方向 | Oculus | 机器人 |
|------|--------|--------|
| X | 向右 | 向前 |
| Y | 向上 | 向左 |
| Z | 向后（朝向用户） | 向上 |

```python
T_OCULUS_TO_ROBOT = np.array([
    [ 0.,  0., -1.],
    [-1.,  0.,  0.],
    [ 0.,  1.,  0.],
])
```

映射关系：

```
机器人 x = -Oculus z
机器人 y = -Oculus x
机器人 z =  Oculus y
```

- **位置增量**：`当前位置 - 上一帧位置`，再乘旋转矩阵
- **旋转增量**：`当前旋转 @ 上一帧旋转的逆`，然后 re-map 轴：

```
robot_rx = oculus_rz
robot_ry = oculus_rx
robot_rz = oculus_ry
```

---

## 3️⃣ 缩放 & 符号修正

在 `get_action()` 中（第 294-305 行），对 delta 应用：

- **位置缩放**：`× 2.5`（放大手部小运动为机器人的大运动）
- **旋转缩放**：`× 2.0`
- **通道符号**：`[1, 1, 1, -1, -1, 1]`（取反 rx, ry）

```python
delta_ee_pose[0] = delta_robot[0] * position_scale * channel_signs[0]   # x
delta_ee_pose[1] = delta_robot[1] * position_scale * channel_signs[1]   # y
delta_ee_pose[2] = delta_robot[2] * position_scale * channel_signs[2]   # z
delta_ee_pose[3] = delta_robot[3] * orientation_scale * channel_signs[3]  # rx
delta_ee_pose[4] = delta_robot[4] * orientation_scale * channel_signs[4]  # ry
delta_ee_pose[5] = delta_robot[5] * orientation_scale * channel_signs[5]  # rz
```

---

## 4️⃣ 逆运动学（IK）求解

**`_solve_ik()`** (第 212 行) — 使用 **Placo IK** 库

把期望的末端位姿（delta 累加得到的绝对位姿）转为 7 个关节角度：

```
读取真实关节位置 → 作为冗余解锚点 →
设置末端目标位姿 → Placo 求解器迭代 30 次 →
输出 7 个关节角度 joint_1..7
```

如果不用 IK（`execute_mode: "ee_pose"`），则跳过此步，直接把 delta 位姿发给机器人。

**扳机处理**（第 335 行）：

```python
gripper_position = 1.0 - trigger_value
```

扳机完全按下 → 夹爪闭合 (0.0)，松开 → 张开 (1.0)

---

## 5️⃣ 通过中间层发送

**`scripts/core/run_record.py`** → `record_loop()` 循环：

```python
action = teleop.get_action()        # 拿到上面处理好的 action 字典
robot.send_action(action)           # 发给机器人
```

**`Franka.send_action()`** (`franka.py:232`) 根据模式分发：

| 模式 | 路径 | 发送内容 |
|------|------|----------|
| **joint**（默认） | `_send_action_oculus_joint()` (376行) | 7 个关节角度，直接控制每个关节 |
| **ee_pose** | `_send_action_cartesian()` (277行) | delta 位姿（带 EMA 平滑，α=0.4） |

安全保护：

- 关节变化 **>1.5 rad** → 跳过（防止危险）
- **>0.1 rad** → 逐步插值（每步 0.02 rad，间隔 10ms）

---

## 6️⃣ 网络传输到实际机器人

```
Franka.send_action()
    → FrankaInterfaceClient (ZeroRPC 客户端)
        → 网络 (TCP)
            → FrankaInterfaceServer (ZeroRPC 服务器，运行在 NUC 上)
                → Polymetis RobotInterface/GripperInterface
                    → 实际的 Franka Research 3 机器人
```

服务器运行在连接机器人的 NUC 上（`192.168.110.15:4242`），通过 ZeroRPC 暴露底层 Polymetis API。

---

## 控制频率

| 层级 | 频率 | 决定因素 |
|------|------|----------|
| 主循环（teleop→robot） | **15 Hz** | `fps: 15` 配置 |
| 大幅动作插值 | **100 Hz** | `time.sleep(0.01)` |
| Polymetis 底层控制 | **500 Hz** | `self._dt = 0.002` |
| Quest 原始数据率 | ~72-90 Hz | 头显硬件限制（被 15Hz 降采样） |

---

## 配置（record_cfg.yaml）

| 参数 | 值 | 说明 |
|------|-----|------|
| Quest IP | `192.168.110.62` | Oculus 头显 IP |
| NUC (robot) IP | `192.168.110.15` | 机器人控制 NUC IP |
| ZeroRPC port | `4242` | 通信端口 |
| fps | `15` | 主循环 / 相机帧率 |
| pose_scaler | `[2.5, 2.0]` | 位置/旋转缩放 |
| channel_signs | `[1,1,1,-1,-1,1]` | 通道符号 |
| execute_mode | `"joint"` | 执行模式（joint / ee_pose） |
| ik_iterations | `30` | IK 求解迭代次数 |
| ik_pos_weight | `2.0` | IK 位置跟踪权重 |
| ik_ori_weight | `3.0` | IK 姿态跟踪权重 |
| ik_joints_weight | `0.2` | 关节锚定权重 |
| ik_regularization | `1.0e-4` | 正则化权重 |
