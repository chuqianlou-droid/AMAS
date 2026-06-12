# Quest3 → Dobot CR5 直连遥操作

不走 ROS2，通过 UDP 直连 Quest 3，TCP 直连 Dobot 控制器。

## 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    电脑 (PC)                             │
│                                                         │
│  quest_udp.py              teleop_mapping.py            │
│  ┌──────────────┐         ┌─────────────────────┐       │
│  │ UDP 5005 接收 │ ──────→ │ 坐标系变换           │       │
│  │ 最新帧读取    │         │ 帧间差分 / 原点差分    │       │
│  │ 按钮解析     │         │ EMA 滤波             │       │
│  └──────────────┘         │ 缩放 / 限幅          │       │
│                           │ 死人开关(RG)         │       │
│                           │ 夹爪(trigger)        │       │
│                           └──────────┬──────────┘       │
│                                      ↓                  │
│                           ┌─────────────────────┐       │
│                           │  servop_teleop.py    │       │
│                           │  ServoP → TCP:29999  │       │
│                           └──────────┬──────────┘       │
└──────────────────────────────────────┼──────────────────┘
                                       │
          UDP:5005 ←───────────────────┤
          (Quest 手柄数据)              │
                                       │ TCP:29999 (ServoP 命令)
                                       ↓
┌─────────────────────┐    ┌──────────────────────────┐
│   Quest 3 头显      │    │   Dobot CR5 控制器       │
│   Unity App         │    │   dashboard 端口 29999   │
│   com.sjtu.quest... │    │   执行 ServoP(xyz,rxryrz)│
└─────────────────────┘    └──────────────────────────┘
```

## 核心处理流程（参考 lerobot_franka_teleop）

```
Quest 手柄原始位姿 (x,y,z, qx,qy,qz,qw)
  │
  ├─[可选] EMA 滤波器 (位置 + 旋转 rotvec)
  │
  ├─ 计算增量（两种模式可选）:
  │   ├─ 帧间差分 (默认, 推荐) — 每帧 ∆ = 当前 - 上一帧
  │   │   RG 松手 → 清零 ∆，防抖动跳变
  │   │   RG 握住 → 累加 ∆，平滑跟随
  │   │
  │   └─ 原点差分 — ∆ = 当前 - 复位原点
  │
  ├─ Oculus → Robot 坐标系变换 (3×3 矩阵)
  │   默认: robot_x = -oculus_x
  │         robot_y = -oculus_z
  │         robot_z = +oculus_y
  │
  ├─ 缩放 (位置 + 旋转独立缩放)
  │
  ├─ 每轴符号翻转
  │
  ├─ 总位移限幅 / 工作空间限幅
  │
  └─ 死区 + 每步限幅 → 最终 ServoP 目标
       (x, y, z 单位 mm; rx, ry, rz 单位 °)
```

### 与 ROS2 方案的关键区别

| 特性 | ROS2 (quest3_cr5_servop_teleop) | 本方案 |
|------|--------------------------------|--------|
| 自由度 | **3-DOF** (位置 only，旋转锁定) | **6-DOF** (位置 + 四元数旋转) |
| 增量方式 | 原点差分 | 帧间差分 (默认) / 原点差分 |
| 死人开关 | ❌ 无 | ✅ RG 按键控制 |
| 夹爪 | ❌ 无 | ✅ 右扳机 |
| 平滑方式 | EMA(0.80) + 速度/加速度规划器 | EMA (可配置) + 步长限幅 |
| 延迟 | ~500-800ms (双层平滑叠加) | ~50-150ms (无规划器) |
| 依赖 | ROS2 Humble + dobot_bringup | Python3 + numpy + scipy |

---

## 环境要求

- **Python 3.10+** (系统自带即可)
- **numpy**, **scipy**

```bash
pip install numpy scipy
```

- **TCP-IP-Python-V4** — Dobot 官方 SDK，放在 `cr5_tele/` 下
  - 路径: `/home/jiaotan/Workspace/cr5_tele/TCP-IP-Python-V4/dobot_api.py`

---

## VR 端准备

1. **Quest 3 头显**
   - 已安装 Unity App `com.sjtu.questcr5teleop`
   - 与电脑在**同一 WiFi 网络**下

2. **在 App 中配置电脑 IP**
   - 电脑上执行 `ip a` 或 `ifconfig` 查看 IP
   - 在 Quest App 界面中输入该 IP 地址
   - App 会向该 IP 的 UDP 5005 端口发送手柄数据

3. **启动 App**
   - 方法 A: 头显中手动点开
   - 方法 B: ADB 自动启动
     ```bash
     adb shell monkey -p com.sjtu.questcr5teleop 1
     ```
   - 如果 ADB 权限不允许，手动打开即可

---

## 使用步骤

### 1. 快速测试：单步 ServoP 验证

先确认电脑能连到机械臂:

```bash
cd /home/jiaotan/Workspace/cr5_tele/dobot_teleop

python3 servop_test_direct.py \
  --robot-ip 192.168.5.1 \
  --dx 10 \
  --speed-mm-s 50 \
  --enable-robot
```

如果 `--clear-error` 或 `--enable-robot` 返回错误码，说明 TCP 通信正常，但可能需要先手动 ClearError。

### 2. 启动遥操作

```bash
cd /home/jiaotan/Workspace/cr5_tele/dobot_teleop

python3 servop_teleop.py \
  --robot-ip 192.168.5.1 \
  --enable-robot \
  --auto-enable \
  --log-targets
```

### 3. 完整参数示例

```bash
python3 servop_teleop.py \
  --robot-ip 192.168.5.1 \
  --enable-robot \
  --clear-error \
  --auto-enable \
  --log-targets \
  --command-rate 10 \
  --servo-t 0.10 \
  --servo-gain 200 \
  --position-scale 0.20 \
  --rotation-scale 0.50 \
  --rotation-mode origin-delta \
  --target-deadband-mm 2.0 \
  --target-deadband-deg 1.0 \
  --max-step-mm 6.0 \
  --max-step-deg 3.0 \
  --max-total-translation-mm 120.0 \
  --max-total-rotation-deg 90.0
```

### 4. 不连机械臂的模拟模式

```bash
python3 servop_teleop.py \
  --robot-ip 192.168.5.1 \
  --dry-run \
  --auto-enable \
  --log-targets
```

---

## 键盘控制

| 按键 | 功能 |
|------|------|
| `e` | 对齐当前 Quest 手柄与机械臂位姿 → 启用遥操作 |
| `p` | 暂停（发送 Stop 命令） |
| `g` | 打印 GetPose 当前值和 GetAngle 关节角 |
| `c` | ClearError |
| `s` | Stop |
| `q` | 退出 |

---

## Quest 手柄操作

| 控制器操作 | 功能 |
|-----------|------|
| **RG** (右手柄握持键) | **死人开关**：握住才发送运动命令，松手冻结当前位置 |
| **右扳机** (rightTrig) | **夹爪控制**：0=张开，1=闭合（需 `--enable-gripper`） |
| **A 键** | 预留复位（当前未实现） |
| **手柄移动/旋转** | 控制机械臂末端 6-DOF 位姿 |

---

## 参数调优

### 基本调参

| 参数 | 默认值 | 作用 | 调优建议 |
|------|--------|------|---------|
| `position_scale` | 0.20 | 手柄位移 → 机器人位移比例 | 0.10~0.50，越大越灵敏 |
| `rotation_scale` | 0.50 | 手柄旋转 → 机器人旋转比例 | 0.30~1.0，越大越灵敏 |
| `rotation_mode` | `frame-delta` | 姿态映射方式 | 验证手柄姿态跟随时用 `origin-delta` |

### 安全限幅

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `target_deadband_mm` | 2.0 | 目标变化 < 此值不发送命令 (防抖动) |
| `target_deadband_deg` | 1.0 | 旋转死区 |
| `max_step_mm` | 6.0 | 单周期最大位移 (10Hz 下约 60mm/s) |
| `max_step_deg` | 3.0 | 单周期最大旋转 |
| `max_total_translation_mm` | 120.0 | 从复位点最大总位移 |
| `max_total_rotation_deg` | 90.0 | 从复位点最大总旋转 |
| `workspace_min/max_*` | ±700等 | 工作空间硬限位 |

### 坐标系变换

默认映射:
```
robot_x = -oculus_x
robot_y = -oculus_z
robot_z = +oculus_y
```

可通过 `--pos-transform 9个浮点数` 传入自定义 3×3 矩阵（行主序）:

```bash
--pos-transform -1 0 0  0 0 -1  0 1 0
--rot-transform -1 0 0  0 0 -1  0 1 0
```

---

## 运行模式

### 旋转帧间差分模式 (默认)

```bash
# 默认就是 frame-delta，不需要额外参数
python3 servop_teleop.py --robot-ip 192.168.5.1 ...
```

特点:
- 手柄旋转每帧只产生微小增量，运动平滑
- RG 松手 → 清零上一帧 → 再握住不跳变
- 来源: lerobot_franka_teleop 的 OculusRobot

### 旋转原点差分模式

```bash
python3 servop_teleop.py --robot-ip 192.168.5.1 --rotation-mode origin-delta ...
```

特点:
- 姿态由“当前手柄姿态 - 对齐时手柄姿态”直接映射
- 适合验证“固定机械臂姿态是否导致逆解限位”
- 建议先用较小 `--rotation-scale 0.05~0.20` 测试

---

## 文件结构

```
dobot_teleop/
├── README.md
├── servop_teleop.py              ← 主入口 (main)
├── servop_test_direct.py         ← 单步 ServoP 测试
└── dobot_teleop/                 ← Python 包
    ├── __init__.py
    ├── dobot_dashboard.py        ← Dobot TCP 封装 (ServoP, GetPose)
    ├── quest_udp.py              ← Quest UDP 接收 + 按钮解析
    └── teleop_mapping.py         ← 核心后处理: 坐标系变换/滤波/限幅
```

---

## 链路自检

如果遇到问题，按这个顺序排查:

```
1. Quest App 是否在发送数据?
   → 终端打印 "First Quest pose received"? 
   → 否则检查 WiFi / App IP 配置

2. UDP 端口是否被占用?
   → netstat -tulpn | grep 5005

3. 机械臂 TCP 是否可达?
   → nc -z 192.168.5.1 29999
   → 或用 servop_test_direct.py 测试

4. ServoP 返回错误?
   → 查看终端输出 error_id
   → 常见: 需要 ClearError → 按 c
   → 需要 EnableRobot → 加 --enable-robot
```

---

## 参考

- lerobot_franka_teleop: OculusRobot._compute_delta_pose() — 帧间差分 + 坐标系变换
- Dobot CR5 官方 SDK: `TCP-IP-Python-V4/dobot_api.py`
- ROS2 对标节点: `quest3_cr5_servop_teleop.py` — 3-DOF 缺失旋转，建议用本方案替代
