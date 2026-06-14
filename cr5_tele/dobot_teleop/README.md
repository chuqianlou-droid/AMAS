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
│                           │  servoj_teleop.py    │       │
│                           │  ServoJ → TCP:29999  │       │
│                           └──────────┬──────────┘       │
└──────────────────────────────────────┼──────────────────┘
                                       │
          UDP:5005 ←───────────────────┤
          (Quest 手柄数据)              │
                                       │ TCP:29999 (ServoJ 命令)
                                       ↓
┌─────────────────────┐    ┌──────────────────────────┐
│   Quest 3 头显      │    │   Dobot CR5 控制器       │
│   Unity App         │    │   dashboard 端口 29999   │
│   com.sjtu.quest... │    │   执行 ServoJ(J1..J6)    │
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
  └─ 死区 + 每步限幅 → 笛卡尔目标位姿
       (x, y, z 单位 mm; rx, ry, rz 单位 °)
       │
       ▼
  InverseKin → 关节限幅检查 (±0.5° 安全裕量)
       │
       ▼
  ServoJ(J1..J6, t, aheadtime, gain)
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
  - 路径: `/home/jiaotan/dobot_ws/src/AMAS/cr5_tele/TCP-IP-Python-V4/dobot_api.py`

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
cd /home/jiaotan/dobot_ws/src/AMAS/cr5_tele/dobot_teleop

python3 servop_test_direct.py \
  --robot-ip 192.168.5.1 \
  --dx 10 \
  --speed-mm-s 50 \
  --enable-robot
```

如果 `--clear-error` 或 `--enable-robot` 返回错误码，说明 TCP 通信正常，但可能需要先手动 ClearError。

### 2. 启动遥操作

默认使用 **ServoJ**（关节空间伺服），IK 预检查 + 关节限幅保护：

```bash
cd /home/jiaotan/dobot_ws/src/AMAS/cr5_tele/dobot_teleop

python3 servoj_teleop.py \
  --robot-ip 192.168.5.1 \
  --enable-robot \
  --auto-enable \
  --ignore-deadman \
  --log-targets
```

如需切回原来的 ServoP（笛卡尔伺服）：

```bash
python3 servoj_teleop.py --servo-mode cartesian ...
```

### 2.1 工具坐标系姿态遥操作

如果要让手柄旋转映射为末端工具坐标系旋转，使用新增入口：

```bash
python3 servoj_toolframe_teleop.py \
  --robot-ip 192.168.5.1 \
  --enable-robot \
  --clear-error \
  --auto-enable \
  --ignore-deadman \
  --log-targets \
  --log-joints \
  --servo-mode joint \
  --rotation-mode origin-delta \
  --position-scale 0.20 \
  --rotation-scale 0.05 \
  --filter-ratio-pos 0.20 \
  --filter-ratio-rot 0.0
```

这个脚本保留原位置映射不变，只把姿态改为 `target_R = origin_R @ delta_R` 的工具坐标系右乘组合。纯旋转时也会发送命令，日志会显示 `step_pos` 和 `step_rot`。

如果要跳过 PC 端 IK 和 ServoJ，直接使用 ServoP 工具坐标系版本：

```bash
python3 servop_toolframe_teleop.py \
  --robot-ip 192.168.5.1 \
  --enable-robot \
  --clear-error \
  --auto-enable \
  --ignore-deadman \
  --log-targets \
  --log-joints \
  --rotation-mode origin-delta \
  --position-scale 0.20 \
  --rotation-scale 0.05 \
  --filter-ratio-pos 0.20 \
  --filter-ratio-rot 0.0
```

`servop_toolframe_teleop.py` 会强制使用 ServoP，即使命令行里误传 `--servo-mode joint` 也会忽略。

### 2.2 在线速度规划 / target governor

如果手柄移动时 raw target 跳得比较快，但不想缓存历史手柄点，可以使用 online target governor 版本：

```bash
python3 toolframe_governor_teleop.py \
  --robot-ip 192.168.5.1 \
  --enable-robot \
  --clear-error \
  --auto-enable \
  --ignore-deadman \
  --log-targets \
  --log-joints \
  --log-timing \
  --servo-mode joint \
  --command-rate 50 \
  --servo-t 0.02 \
  --servo-gain 800 \
  --servo-aheadtime 70 \
  --rotation-mode origin-delta \
  --position-scale 0.80 \
  --rotation-scale 0.30 \
  --filter-ratio-pos 0.50 \
  --filter-ratio-rot 0.40 \
  --max-linear-speed-mm-s 35 \
  --max-angular-speed-deg-s 20 \
  --max-joint-speed-deg-s 35 \
  --enable-gripper \
  --log-quest
```

这个入口仍然每个周期只取最新 Quest pose，不排队旧手柄点。链路是：

```text
Quest UDP latest pose
→ mapper 生成 raw Cartesian target
→ Cartesian target governor 限速追踪
→ IK
→ joint target rate limit
→ ServoJ / ServoP at fixed command-rate
```

### 3. 完整参数示例

```bash
python3 servoj_teleop.py \
  --robot-ip 192.168.5.1 \
  --enable-robot \
  --clear-error \
  --auto-enable \
  --ignore-deadman \
  --log-targets \
  --log-joints \
  --command-rate 30 \
  --servo-t 0.03 \
  --servo-gain 500 \
  --servo-aheadtime 30 \
  --servo-mode joint \
  --position-scale 0.20 \
  --rotation-scale 0.30 \
  --filter-ratio-pos 0.20 \
  --filter-ratio-rot 0.20 \
  --target-deadband-mm 2.0 \
  --target-deadband-deg 1.0 \
  --max-step-mm 4.0 \
  --max-step-deg 2.0 \
  --max-total-translation-mm 150.0 \
  --max-total-rotation-deg 60.0
```

### 4. 不连机械臂的模拟模式

```bash
python3 servoj_teleop.py \
  --robot-ip 192.168.5.1 \
  --dry-run \
  --auto-enable \
  --ignore-deadman \
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

每次重新按下 RG 时，程序会把当前手柄位姿设为新的手柄原点，但保留最初对齐的机械臂原点和已累计的机器人目标位移，相当于“离合器”操作。

当前 Quest App 如果只发送 `x,y,z,qx,qy,qz,qw`，没有发送 RG 按键字段，启动时需要加 `--ignore-deadman`，否则机械臂会一直认为死人开关未按下。

---

## 参数调优

### 伺服参数

| 参数 | 默认值 | 作用 | 调优建议 |
|------|--------|------|---------|
| `servo_mode` | `joint` | 伺服模式 | `joint`=ServoJ (推荐)，`cartesian`=ServoP (旧版) |
| `command_rate` | 10.0 | PC 控制循环频率 (Hz) | 建议 30~50，太低卡顿，太高 TCP 来不及 |
| `servo_t` | 0.10 | 每步运行时间 (s) | 与 `1/command_rate` 匹配 |
| `servo_gain` | 500 | PID 的 P 项，关节跟踪力度 | 200~1000，越大越跟手，过大可能震动 |
| `servo_aheadtime` | 50 | PID 的 D 项，超前补偿 | 20~100，抑制过冲 |

### Online governor 参数

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `max_linear_speed_mm_s` | 30.0 | command target 追 raw target 的最大线速度 |
| `max_angular_speed_deg_s` | 15.0 | command target 追 raw target 的最大角速度 |
| `max_joint_speed_deg_s` | 30.0 | ServoJ 模式下每个关节目标的最大角速度 |
| `log_timing` | 关闭 | 打印 InverseKin 和 ServoJ/ServoP TCP 耗时 |
| `timing_log_interval` | 2.0 | timing 汇总日志间隔，单位秒 |

每周期实际步长由控制周期自动换算：

```text
max_pos_step_mm = max_linear_speed_mm_s * (1 / command_rate)
max_rot_step_deg = max_angular_speed_deg_s * (1 / command_rate)
max_joint_step_deg = max_joint_speed_deg_s * (1 / command_rate)
```

### 基本调参

| 参数 | 默认值 | 作用 | 调优建议 |
|------|--------|------|---------|
| `position_scale` | 0.20 | 手柄位移 → 机器人位移比例 | 0.10~0.50，越大越灵敏 |
| `rotation_scale` | 0.50 | 手柄旋转 → 机器人旋转比例 | 建议从 0.05~0.30 开始，旋转太大容易奇异 |
| `rotation_mode` | `frame-delta` | 姿态映射方式 | 帧间累积（平滑），`origin-delta` 直接映射 |
| `filter_ratio_pos` | 0.0 | 位置 EMA 滤波比例 | 0.20 可平滑抖动，越大越滞后 |
| `filter_ratio_rot` | 0.0 | 旋转 EMA 滤波比例 | 0.20 可平滑抖动 |

### 安全限幅

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `target_deadband_mm` | 2.0 | 增量 < 此值不发送命令 (防抖动) |
| `target_deadband_deg` | 1.0 | 旋转死区 |
| `max_step_mm` | 6.0 | 单周期最大位移 |
| `max_step_deg` | 3.0 | 单周期最大旋转 |
| `max_total_translation_mm` | 120.0 | 从复位点最大总位移 |
| `max_total_rotation_deg` | 90.0 | 从复位点最大总旋转 |
| `workspace_min/max_*` | ±700等 | 笛卡尔工作空间硬限位 |

### ServoJ 关节限幅

ServoJ 模式下，PC 端 IK 解算后自动检查关节角，**单边预留 0.5° 安全裕量**：

| 关节 | 机械限位 | 安全范围 |
|------|---------|---------|
| J1 | ±360° | ±359.5° |
| J2 | ±360° | ±359.5° |
| J3 | ±160° | ±159.5° |
| J4 | ±360° | ±359.5° |
| J5 | ±360° | ±359.5° |
| J6 | ±360° | ±359.5° |

超出安全范围 → 跳过该步，不崩溃。

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

### ServoJ 关节伺服 (默认)

```bash
# 默认就是 joint，不需要额外参数
python3 servoj_teleop.py --robot-ip 192.168.5.1 ...
```

特点:
- PC 端 IK 解算 + 关节限幅预检查，不会因 IK 失败导致机械臂急停
- 无奇异点问题（J5≈0° 不会红光）
- 单边 0.5° 安全裕量，超限自动跳过
- 2 次 TCP 往返/周期（IK + ServoJ），建议 `--command-rate 30`

### ServoP 笛卡尔伺服 (旧版)

```bash
python3 servoj_teleop.py --robot-ip 192.168.5.1 --servo-mode cartesian ...
```

特点:
- 控制器内部 IK，可能因奇异点/关节限位急停（红光）
- 1 次 TCP 往返/周期，但被拒会崩溃

### 旋转帧间差分模式 (默认)

特点:
- 手柄旋转每帧只产生微小增量，运动平滑
- RG 松手 → 清零上一帧 → 再握住不跳变
- 来源: lerobot_franka_teleop 的 OculusRobot

### 旋转原点差分模式

```bash
python3 servoj_teleop.py --robot-ip 192.168.5.1 --rotation-mode origin-delta ...
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
├── servoj_teleop.py              ← 主入口 (ServoJ 默认, 可切 ServoP)
├── servoj_toolframe_teleop.py    ← 工具坐标系姿态版本
├── servop_toolframe_teleop.py    ← 工具坐标系姿态 + ServoP 版本
├── toolframe_governor_teleop.py  ← 工具坐标系姿态 + 在线 target governor
├── servop_test_direct.py         ← 单步 ServoP 测试
└── dobot_teleop/                 ← Python 包
    ├── __init__.py
    ├── dobot_dashboard.py        ← Dobot TCP 封装 (ServoJ, ServoP, GetPose)
    ├── quest_udp.py              ← Quest UDP 接收 + 按钮解析
    ├── toolframe_mapping.py      ← 工具坐标系姿态 mapper
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
