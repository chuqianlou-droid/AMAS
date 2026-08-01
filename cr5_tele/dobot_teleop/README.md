# Quest3 → Dobot CR5 直连遥操作 + VLA 数据采集

不走 ROS2，通过 UDP 直连 Quest 3，TCP 直连 Dobot 控制器。

## 目录结构

```
dobot_teleop/
├── README.md                           ← 本文件
├── servoj_teleop.py                    ← 遥操作：ServoJ 关节伺服（默认入口）
├── servoj_toolframe_teleop.py          ← 遥操作：工具坐标系姿态映射版
├── toolframe_governor_teleop.py        ← 遥操作：工具坐标系 + 在线速度规划器
│
├── scripts/
│   ├── dataset/
│   │   ├── record_cr5a_pi0_dataset.py  ← VLA 数据录制（支持 raw / LeRobot 双格式）
│   │   ├── inspect_cr5a_pi0_dataset.py ← 检查录制好的 episode
│   │   └── convert_to_lerobot.py       ← 批量转换 raw → LeRobot 格式
│   └── bridge/
│       └── pi0_cr5a_bridge.py          ← OpenPI / π0 模型 → CR5A 桥接
│
├── dobot_teleop/                       ← Python 包（共享模块）
│   ├── __init__.py
│   ├── dobot_dashboard.py              ← Dobot TCP 封装 (ServoJ, ServoP, GetPose 等)
│   ├── quest_udp.py                    ← Quest UDP 接收 + 按钮解析
│   ├── teleop_mapping.py               ← 遥操作映射：坐标系变换/滤波/限幅（关节空间）
│   ├── toolframe_mapping.py            ← 遥操作映射：工具坐标系版本
│   ├── transforms.py                   ← 工具偏移变换：TCP ↔ gripper_center
│   ├── cr5a_pi0_schema.py             ← CR5A 观测 / 7D action 数据规范
│   ├── realsense_dual_rgb_provider.py  ← D415 + D435 双路 RGB provider
│   ├── teleop_action_stream.py         ← 遥操作 action 的 UDP 发布/订阅
│   └── lerobot_writer.py              ← LeRobot v2.1 格式写入器
│
├── scripts/
│   ├── dataset/
│   │   ├── record_cr5a_pi0_dataset.py  ← VLA 数据录制（支持 raw / LeRobot 双格式）
│   │   ├── inspect_cr5a_pi0_dataset.py ← 检查录制好的 episode
│   │   └── convert_to_lerobot.py       ← 批量转换 raw → LeRobot 格式
│   ├── bridge/
│   │   └── pi0_cr5a_bridge.py          ← OpenPI / π0 模型 → CR5A 桥接
│   └── test_tool_offset_transform.py   ← 工具偏移变换单元测试
│
├── tests/                              ← 单元测试
├── datasets/                           ← 数据目录（录制输出）
└── __pycache__/                        ← (已 gitignore)
```

---

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
│                           │ 夹爪(trigger 控制)    │       │
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

## 环境要求

- **Python 3.10+** (系统自带即可)
- **numpy**, **scipy**

```bash
pip install numpy scipy
```

CR5A PI0 数据采集还需要：

```bash
pip install Pillow pyrealsense2
```

LeRobot 格式需要（仅在 recording/convert 时需要）：

```bash
pip install datasets pyarrow
```

- **TCP-IP-Python-V4** — Dobot 官方 SDK，放在 `cr5_tele/` 下
  - 路径: `/home/jiaotan/AMAS/cr5_tele/TCP-IP-Python-V4/dobot_api.py`

---

## 快速开始

### 测试连接

```bash
cd /home/jiaotan/AMAS/cr5_tele/dobot_teleop
python3 tests/servop_test_direct.py --robot-ip 192.168.5.1 --dx 10 --enable-robot
```

### 启动遥操作（已调试参数 ✅）

**推荐入口 — 在线速度规划器版本**：

```bash
cd /home/jiaotan/AMAS/cr5_tele/dobot_teleop

python3 toolframe_governor_teleop.py \
  --robot-ip 192.168.5.1 \
  --enable-robot --clear-error --auto-enable \
  --log-targets --log-joints --log-timing --log-quest \
  --servo-mode joint \
  --command-rate 50 --servo-t 0.03 --servo-gain 700 --servo-aheadtime 65 \
  --rotation-mode frame-delta \
  --position-scale 0.80 --rotation-scale 0.30 \
  --filter-ratio-pos 0.50 --filter-ratio-rot 0.40 \
  --max-linear-speed-mm-s 35 --max-angular-speed-deg-s 20 --max-joint-speed-deg-s 35 \
  --collision-level 1 \
  --enable-gripper
```

**简化版（ServoJ 关节伺服）**：

```bash
python3 servoj_toolframe_teleop.py \
  --robot-ip 192.168.5.1 \
  --enable-robot --clear-error --auto-enable \
  --log-targets --log-joints \
  --servo-mode joint \
  --rotation-mode frame-delta \
  --position-scale 0.80 --rotation-scale 0.30 \
  --filter-ratio-pos 0.50 --filter-ratio-rot 0.40
```

**不连机械臂的模拟模式**：

```bash
python3 toolframe_governor_teleop.py --robot-ip 192.168.5.1 --dry-run
```

---

## VLA 数据采集（CR5A PI0 / LeRobot）

### 数据格式

| 字段 | Shape | 说明 |
|------|-------|------|
| `observation.state` | (7,) | `[j1, j2, j3, j4, j5, j6, gripper]`，gripper 为 PGE 实际位置读回，0=张开，1=闭合 |
| `action` | (7,) | `[dx_mm, dy_mm, dz_mm, dRx_deg, dRy_deg, dRz_deg, gripper]`，前 6 维由真实 gripper_center 轨迹差分得到，第 7 维为下一帧 PGE 实际夹爪状态 |
| `observation.images.d415` | (224, 224, 3) | D415 相机 RGB |
| `observation.images.d435` | (224, 224, 3) | D435 相机 RGB |

> **注意**：`observation.state` 不包含 TCP 位姿。TCP 位姿是关节角度的正向运动学结果，属于冗余信息。`action` 的前 6 维不是遥操作命令 delta，而是录制到的真实 gripper_center 位姿从当前帧到下一帧的差分；第 7 维 gripper 也不再使用 Quest trigger，而是来自 PGE `0x0202` 当前位置读回换算出的真实夹爪状态。

### 方式一：录制时直接输出 LeRobot 格式（推荐）

**采集前** — 确保 RealSense Viewer 没有占用相机：

```bash
pkill -f realsense-viewer || true
```

**终端 1** — 启动遥操作并发布 action stream：

```bash
cd /home/jiaotan/AMAS/cr5_tele/dobot_teleop

source ~/miniconda3/etc/profile.d/conda.sh
conda activate teleop_dobot

python3 toolframe_governor_teleop.py \
  --robot-ip 192.168.5.1 \
  --enable-robot --clear-error --auto-enable \
  --log-targets --log-joints --log-timing --log-quest \
  --servo-mode joint \
  --command-rate 33 --servo-t 0.03 --servo-gain 700 --servo-aheadtime 55 \
  --rotation-mode frame-delta \
  --position-scale 0.80 --rotation-scale 0.30 \
  --filter-ratio-pos 0.50 --filter-ratio-rot 0.40 \
  --max-linear-speed-mm-s 35 --max-angular-speed-deg-s 20 --max-joint-speed-deg-s 35 \
  --collision-level 1 \
  --enable-gripper \
  --gripper-state-read-interval-s 0.05 \
  --publish-action-stream --action-stream-host 127.0.0.1 --action-stream-port 5010
```

**终端 2** — 录制数据，直接生成 LeRobot parquet：

```bash
cd /home/jiaotan/AMAS/cr5_tele/dobot_teleop

source ~/miniconda3/etc/profile.d/conda.sh
conda activate teleop_dobot

python3 scripts/dataset/record_cr5a_pi0_dataset.py \
  --robot-ip 192.168.5.1 \
  --d415-serial 841612070371 \
  --d435-serial 801312070525 \
  --camera-width 224 --camera-height 224 --camera-fps 15 \
  --output-dir ./datasets/cr5a_lerobot \
  --format lerobot \
  --prompt "pick the object" \
  --duration-sec 40 \
  --action-source teleop \
  --teleop-action-host 127.0.0.1 --teleop-action-port 5010 \
  --teleop-state-source feedback \
  --max-action-age-ms 200 \
  --record-only-when-deadman --drop-no-action \
  --preview-frame-count 10
```

启动录制后确认终端打印：

```text
Teleop state source: feedback
State/image alignment: camera frame -> immediate live ToolVectorActual/QActual sample
```

同时确认终端 1 的 action stream 日志里 `gripper_state=...` 是数值；张开应接近 0，闭合应接近 1。如果显示 `gripper_state=unavailable`，说明 PGE 当前位置没有成功读回，此时不要继续采新训练数据。

默认录制目录是 `./datasets/cr5a_lerobot`。LeRobot parquet 会写到 `./datasets/cr5a_lerobot/data/chunk-000/episode_XXXXXX.parquet`，元数据写到 `./datasets/cr5a_lerobot/meta/`。`--teleop-state-source feedback` 会在每帧相机图像取到后立即通过 Dobot feedback 端口 `30004` 读取 `ToolVectorActual/QActual`，避免把 Quest 遥操作控制参考当成真实机械臂状态，也不会抢占遥操作正在使用的 Dashboard 端口 `29999`。保存时，action 前 6 维由连续两帧真实 gripper_center 位姿差分生成，第 7 维由遥操作端通过 PGE `0x0202` 当前位置寄存器读回的真实夹爪状态生成，避免继续使用有遥操作延迟的命令 action/trigger。不要用 `--teleop-state-source stream` 采新训练数据，除非只是为了回退/排错。

每个 episode 会自动在 `episode_XXXXXX_preview/d415/` 和 `d435/` 下各保存 10 张均匀抽样 PNG，方便肉眼检查相机视角。如果确实需要导出每一帧，再额外加 `--save-png-frames`。

多次录制到同一个 `--output-dir` 会自动追加 episode。录制完即可直接用于训练，无需转换。

### 方式二：录制 raw 格式再批量转换

录制为 legacy raw 格式：

```bash
python3 scripts/dataset/record_cr5a_pi0_dataset.py \
  --robot-ip 192.168.5.1 \
  --d415-serial 841612070371 \
  --d435-serial 801312070525 \
  --camera-width 224 --camera-height 224 --camera-fps 15 \
  --output-dir ./datasets/cr5a_raw \
  --prompt "pick the object" \
  --duration-sec 30 \
  --preview-frame-count 10 \
  --action-source teleop \
  --teleop-action-host 127.0.0.1 --teleop-action-port 5010 \
  --teleop-state-source feedback
```

批量转换为 LeRobot 格式：

```bash
python3 scripts/dataset/convert_to_lerobot.py \
  --input-dir ./datasets/cr5a_raw \
  --output-dir ./datasets/cr5a_lerobot
```

### 方式三：Mock 录制（无遥操作，测试用）

```bash
python3 scripts/dataset/record_cr5a_pi0_dataset.py \
  --robot-ip 192.168.5.1 \
  --d415-serial 841612070371 \
  --d435-serial 801312070525 \
  --camera-width 224 --camera-height 224 --camera-fps 15 \
  --output-dir ./datasets/cr5a_test \
  --format lerobot \
  --prompt "pick the object" \
  --duration-sec 10 \
  --preview-frame-count 10 \
  --mock-action zero
```

### 检查录制的数据

```bash
# 检查 raw 格式 episode
python3 scripts/dataset/inspect_cr5a_pi0_dataset.py \
  --episode-dir ./datasets/cr5a_raw/episode_000000 \
  --save-preview ./preview.png
```

### 验证 LeRobot 数据集

```bash
python3 -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('cr5a_lerobot', root='./datasets/cr5a_lerobot')
print(f'Episodes: {ds.num_episodes}, Frames: {ds.num_frames}')
print(f'State: {ds.meta.shapes[\"observation.state\"]}')
print(f'Action: {ds.meta.shapes[\"action\"]}')
"
```

---

## OpenPI / π0 → CR5A 桥接

`scripts/bridge/pi0_cr5a_bridge.py` 将 PI0 输出通过安全路径发送到 CR5A：

```text
OpenPI policy server: {"actions": action_chunk}V
→ RTC soft-mask inpainting inside PI0 flow sampling
→ RTC scheduler (delay estimate + stale-prefix skip; no action0 reset on chunk arrival)
→ action adapter (units / axis map / base-or-tool delta)
→ CartesianSafetyEnvelope (workspace + displacement + orientation fence)
→ CartesianTargetGovernor (linear and angular rate limit)
→ ServoP(X,Y,Z,Rx,Ry,Rz,t,aheadtime,gain)
```

当前 CR5A 上机脚本默认使用 `--policy-scheduler rtc --rtc-soft-mask`，不是旧的简单异步替换。RTC 调度会在当前 chunk 执行到 `s_min - d_pred` 时后台请求下一次推理；请求里会把上一段剩余 action chunk 作为 `action` 条件传给 OpenPI，OpenPI 在 PI0 flow denoising 内部执行 hard-prefix + exponential soft-mask guidance。新 chunk 返回后，bridge 再按真实经过的控制步数 `d` 跳过 `action[0:d]`，从尚未过期的动作接上，避免“新 chunk 来了又从 action0 开始发”的时间错位。

如果需要回退到只做延迟对齐、不做模型内部 inpainting，可在 bridge 指令里加 `--no-rtc-soft-mask`。修改 OpenPI 代码后需要重启 `serve_policy.py`，否则旧 server 进程不会加载 RTC soft-mask sampler。

### 为什么改成 RTC soft-mask

旧的 `async` 推理逻辑只是在后台请求下一段 action chunk。新 chunk 一回来，如果直接替换旧 chunk 并从 `action[0]` 开始执行，`action[0:d]` 实际上对应的是推理请求发出时的机器人状态；但等推理结果返回时，机械臂已经继续执行了 `d` 个控制周期。这会把“已经过期的动作”再次发给机械臂，表现为动作回拉、停顿、提前闭合或末端对不上物体。

只在 bridge 层跳过 `action[0:d]` 可以解决“过期动作重复执行”，但新 chunk 本身仍然是在不知道旧 chunk 已经执行到哪的情况下独立采样出来的。论文里的 RTC 进一步要求在模型 flow denoising 内部把上一段 chunk 的重叠部分作为条件：前 `d` 个重叠动作 hard mask 固定，后面的重叠区用指数衰减 soft mask 约束，剩余未来动作自由生成。这样新旧 chunk 的连接更连续，不是简单拼接，也不是普通异步替换。

### 具体做了哪些改动

1. OpenPI 模型采样器：`openpi/src/openpi/models/pi0.py`
   - 新增 `_rtc_prefix_weights()`，生成 hard-prefix + exponential soft-mask 权重。
   - `sample_actions()` 新增 `rtc_prev_actions`、`rtc_inference_delay`、`rtc_prefix_attention_horizon`、`rtc_max_guidance_weight` 参数。
   - 在每一步 flow denoising 中计算当前最终动作估计 `x_0 = x_t - t * v_t`，对上一段 chunk 的重叠区做 VJP pseudo-inverse guidance。
   - 因为 OpenPI 的采样方向是 `t=1` 噪声到 `t=0` 动作，积分步长为负，所以 guidance correction 在 velocity 上使用反号。

2. OpenPI policy server 输入层：`openpi/src/openpi/policies/policy.py`
   - 从 observation 中取出 `__rtc` 元信息，不让它进入普通 observation。
   - 复用现有 CR5A transform，把上一段 7D action chunk 按训练时相同的统计量 Normalize，并 Pad 到 PI0 的 32D action 空间。
   - 把处理后的 `actions` 作为 `rtc_prev_actions` 传给 `sample_actions()`。

3. CR5A observation provider：`scripts/bridge/cr5a_observation_provider.py`
   - 普通请求仍然给 `action=0` 占位。
   - RTC 请求时把 bridge 传来的上一段剩余 action chunk 写入 `action` 字段，并附带 `__rtc`。

4. CR5A bridge 调度器：`scripts/bridge/pi0_cr5a_bridge.py`
   - 新增 `--policy-scheduler rtc`、`--rtc-soft-mask/--no-rtc-soft-mask`、`--rtc-max-guidance-weight` 等参数。
   - 记录当前完整 chunk、已执行步数 `chunk_elapsed`、历史推理延迟，并预测下一次 `d_pred`。
   - 当执行到 `s_min - d_pred` 时发起下一次后台推理。
   - 后续 RTC 请求会把当前 chunk 从 `chunk_elapsed` 开始的剩余动作左对齐传给 OpenPI，作为 soft-mask 条件。
   - 新 chunk 返回后按真实 elapsed step 跳过 `action[0:d]`，从未过期动作继续执行。

5. 上机脚本：`scripts/bridge/run_pi0_cr5a_servoj.sh`
   - 默认启用 `--policy-scheduler rtc --rtc-soft-mask`。
   - 当前配置使用 `--open-loop-horizon 33`、`--rtc-min-execution-horizon 33`、`--policy-prefetch-remaining 8`。

推荐直接运行：

```bash
cd /home/jiaotan/AMAS/cr5_tele/dobot_teleop
bash scripts/bridge/run_pi0_cr5a_servoj.sh "pick the object"
```

```bash
cd /home/jiaotan/AMAS/cr5_tele/dobot_teleop

# Dry-run 验证
python3 scripts/bridge/pi0_cr5a_bridge.py \
  --robot-ip 192.168.5.1 \
  --actions-jsonl /path/to/actions.jsonl \
  --action-format cartesian_delta_mm_deg \
  --delta-frame tool \
  --max-actions 10

# 正式执行
python3 scripts/bridge/pi0_cr5a_bridge.py \
  --robot-ip 192.168.5.1 \
  --policy-host 127.0.0.1 --policy-port 8000 \
  --observation-provider /path/to/cr5a_observation.py:make_observation \
  --instruction "pick up the object" \
  --policy-scheduler rtc \
  --rtc-soft-mask \
  --action-format cartesian_delta_mm_deg \
  --delta-frame tool \
  --command-rate 10 --max-linear-speed-mm-s 30 --max-angular-speed-deg-s 15 \
  --clear-error --enable-robot --execute --log-targets
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

## Quest 手柄操作

| 控制器操作 | 功能 |
|-----------|------|
| **RG** (右手柄握持键) | **死人开关**：握住才发送运动命令，松手冻结当前位置 |
| **右扳机** (rightTrig) | **夹爪控制**：0=张开，1=闭合（需 `--enable-gripper`） |
| 手柄移动/旋转 | 控制机械臂末端 6-DOF 位姿 |

---

## 参数调优

### 伺服参数（已调试 ✅）

| 参数 | 调试值 | 默认值 | 作用 | 调优建议 |
|------|--------|--------|------|---------|
| `servo_mode` | `joint` | `joint` | 伺服模式 | `joint`=ServoJ (推荐)，IK 预检查 + 关节步长限制 |
| `command_rate` | **50** | 10.0 | PC 控制循环频率 (Hz) | 50Hz 时 servo_t=0.03s 匹配 |
| `servo_t` | **0.03** | 0.10 | 每步运行时间 (s) | 与 `1/command_rate` 匹配 |
| `servo_gain` | **700** | 500 | PID 的 P 项 | 200~1000 |
| `servo_aheadtime` | **65** | 50 | PID 的 D 项 | 20~100 |
| `collision_level` | **1** | — | 碰撞检测灵敏度 | 0=关闭, 1~5（越高越灵敏）。**遥操作建议 1**，防止误触发碰撞保护 |

### Online governor 参数（已调试 ✅）

| 参数 | 调试值 | 默认值 | 作用 |
|------|--------|--------|------|
| `max_linear_speed_mm_s` | **35** | 30.0 | command target 追 raw target 的最大线速度 |
| `max_angular_speed_deg_s` | **20** | 15.0 | 最大角速度 |
| `max_joint_speed_deg_s` | **35** | 30.0 | ServoJ 模式下每个关节目标的最大角速度（防奇异抖动） |

### 基本映射参数（已调试 ✅）

| 参数 | 调试值 | 默认值 | 作用 |
|------|--------|--------|------|
| `position_scale` | **0.80** | 0.20 | 手柄位移 → 机器人位移比例 |
| `rotation_scale` | **0.30** | 0.50 | 手柄旋转 → 机器人旋转比例 |
| `rotation_mode` | **`frame-delta`** | `frame-delta` | 帧间累积（平滑）；`origin-delta` 直接映射 |
| `filter_ratio_pos` | **0.50** | 0.0 | 位置 EMA 滤波比例 (0=关闭, 0.5=中等平滑) |
| `filter_ratio_rot` | **0.40** | 0.0 | 旋转 EMA 滤波比例 (0=关闭, 0.4=中等平滑) |

### 安全限幅

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `target_deadband_mm` | 2.0 | 增量 < 此值不发送命令 |
| `target_deadband_deg` | 1.0 | 旋转死区 |
| `max_step_mm` | 0.8 | 单周期最大位移（raw mapper 内部） |
| `max_step_deg` | 0.50 | 单周期最大旋转（raw mapper 内部） |
| `max_total_translation_mm` | 500.0 | 从复位点最大总位移（governor 版本默认更大） |
| `max_total_rotation_deg` | 90.0 | 从复位点最大总旋转 |

### 工具偏移（gripper_center）

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `--use-gripper-center-pose` | true | 将实时 TCP 位姿通过工具偏移映射到 gripper_center |
| `--tool-offset-z-mm` | 160.0 | TCP 局部 Z 轴偏移量 |
| `--controller-tool-offset-already-set` | false | 控制器已设工具偏移则跳过软件变换 |
| `--log-pose-diff` | false | 1Hz 打印 raw_tcp vs gripper_center 差异 |

录制脚本会先通过 Dobot feedback 端口读取实时 `ToolVectorActual/QActual`；如果显式选择 `--teleop-state-source dashboard`，才会改用 Dashboard `GetPose/GetAngle`，但这通常会与遥操作端抢占 `29999`。在默认 `--use-gripper-center-pose` 下，`ToolVectorActual` 返回的 raw TCP/末端法兰位姿会经过 `tool-offset-z-mm=160` 的软件工具偏移，计算出夹爪中心 `gripper_center` 位姿。Legacy raw 格式会把这个 gripper_center 作为主 `tcp_pose` 保存；LeRobot 格式的 `observation.state` 仍只保存 `[j1..j6, gripper]`，其中 gripper 是 PGE 实际位置读回，不直接保存笛卡尔位姿。

---

## 坐标系变换

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

## 链路自检

```
1. Quest App 是否在发送数据?
   → 终端打印 "First Quest pose received"?
   → 否则检查 WiFi / App IP 配置

2. UDP 端口是否被占用?
   → netstat -tulpn | grep 5005

3. 机械臂 TCP 是否可达?
   → nc -z 192.168.5.1 29999
   → 或用 tests/servop_test_direct.py 测试

4. ServoP 返回错误?
   → 查看终端输出 error_id
   → 常见: 需要 ClearError → 按 c
   → 需要 EnableRobot → 加 --enable-robot
```

---

## 参考

- lerobot_franka_teleop: OculusRobot._compute_delta_pose() — 帧间差分 + 坐标系变换
- Dobot CR5 官方 SDK: `TCP-IP-Python-V4/dobot_api.py`
- LeRobot 数据集格式: https://github.com/huggingface/lerobot
