# 自然语言仿真数据后台接口

平台通过 [HTTP v1](docs/platform.md) 提交作业，也可直接调用下方的 Python 入口。团队资源的实验室地址、官方下载方法和踩坑记录见 [下载指南](docs/downloads.md)。

全部仿真、模型调用、采集和导出在实验服务器进行。当前仅实际验证了 `g2crsb90d`；多机器人扩展接口不代表第二种机器人已验证。

## 后台入口

在 A100 上进入项目目录并载入项目专属 Docker 路径。

```bash
cd /home/vipuser/by/genie_sim
export PATH="/data/by/genie_sim/recovery-tools/docker:$PWD/.venv/bin:$PATH"
export DOCKER_HOST=unix:///data/by/genie_sim/dockerhub-sidebuild-20261119/docker.sock
```

保存 UTF-8 请求文件 `requests/example.json`。

```json
{"instruction":"用g2crsb90d完成手机装盒、盖笔帽和积木插入，每个任务采5条完整成功轨迹，场景0三条、场景1两条。"}
```

准备真实 Qwen 计划、Embedding 检索，不启动仿真：

```bash
.venv/bin/python natural_backend.py --request requests/example.json --robot-id g2crsb90d
```

生成计划后启动有配额和预算限制的执行器：

```bash
.venv/bin/python natural_backend.py --request requests/example.json --robot-id g2crsb90d --execute
```

可分别指定 `--task insert_building_block`、`--scene 0`。用户语言、任务、场景和机器人 ID 分开传入，显式任务必须符合语言目标。当前配额策略固定为每任务 5 条、两场景 3+2；其他配额尚未实现。

Python 接口：`natural_backend.submit(instruction, robot_id=..., task_ids=..., scenes=..., execute=...)`。

目前三个已注册任务可进入开发执行：`phone_packaging`、`put_the_cap_on_the_pen_new`、`insert_building_block`。尚未通过物理验收的任务/场景属于开发候选，不代表已有合格轨迹。未注册的机器人、资产、技能或任务返回 `status=unsupported` 及缺项，不用相近任务替代。

## 模型与执行边界

1. `qwen-plus` 判断需求范围并提取资产搜索词。`text-embedding-v4` 批量检索本地官方 USD 索引。按场景核对装配绑定、资产落盘、包围盒和交互标注可用性，将可用资产及场景信息传给技能计划。执行时继续校验抓取、可达性、避碰和装配。
2. `StructuredPlanner` 提供机器人能力、任务几何契约和技能注册表，归档模型原文、调用回执和计划 SHA256。模型输出受约束 JSON，精确位姿由几何模块计算。
3. `SkillExecutor` 按计划逐条执行；`Geometry` 根据当前场景计算位姿；官方 RobotAPI/cuRobo 规划并通过 PhysX 执行；独立验证器决定成功。

旧 `run_language_job.py` 转入同一结构化入口。整任务基线脚本只保留作调试参考。正式验收要求实际执行顺序与真实模型计划一致，单纯选择任务名称无法通过。

技能注册表包括 `grasp`、`move_held`、`align_precision`、`insert_axial`、`release`、`retreat`、`wait_stable`、`verify_relation`、`sync_assembly_proxy`、`verify_task`。注册技能与每个机器人/任务组合已验证是不同状态。

## 恢复和难点检查

优先使用其他有效抓取/运动候选，再基于实测状态局部纠正；仍失败时，由模型重规划剩余技能。每回合最多三版模型计划；规划器尝试次数、回合时间和场景尝试次数也有限制。终止保留失败证据。

正式数据要求从明确初态连续执行，不能包含回滚。搜索调试可以回滚，但不能通过训练数据验收。

每个新配方先运行一条。自动检查覆盖全程，深入画面检查集中在抓取、释放、对齐和接触装配。`certify_recipe.py` 保存技术复核记录，供同版代码、机器人模型和计划补齐配额；这是技术检查，不是用户审批。更改配方后不能继续沿用旧版证书。

已验收回合可作为不可变版本补采：

```bash
.venv/bin/python batch_collect.py --job jobs/<job>/job.json \
  --tasks insert_building_block --scenes 0 \
  --source-run /data/by/genie_sim/collection-runs/<accepted-run>
```

## 数据与读取

每回合位于 `/data/by/genie_sim/collection-runs/<run>`。

- `source/`：实际加载代码、机器人 URDF、校准、请求、种子、能力描述与准备计划。
- `raw/`：不可变原始 MCAP、首次提取数据、三路视频和逐文件校验。
- `raw-aligned/`：官方分块提取漏帧时，从同一 MCAP 重提取；原始记录保留。不插帧、不插值、不重放物理。
- `evidence/`：实际下发控制、实测状态、120Hz 装配证据、执行步骤、模型原文与 API 用量、相机和源代码核对。
- `acceptance.json`：完整独立验收。实际穿孔、套合或放置关系与相应稳定性判据、时间同步和命令来源共同决定结果。

全部场景配额通过后，`batch_collect.py` 自动将合格回合导出为 `/data/by/genie_sim/exports/<job_id>-v2-1`。独立 CPU 读取器核对数值和三路视频后，作业返回 `status=complete` 及 `dataset.dataset_path`。导出或读取异常时，作业保留轨迹和诊断目录，状态为 `dataset_finalization_failed`，重跑同一作业可接续。

正式数据集验证完成后，可以按同一回合编号集中保存原始轨迹。

```bash
.venv/bin/python bundle_raw_trajectories.py /data/by/genie_sim/exports/<job_id>-v2-1
```

集中目录为 `/data/by/genie_sim/exports/<job_id>-raw-trajectories`。`manifest.json` 对应正式数据集的 `episode_index`，列出原始 MCAP、同步状态、实际动作、相机视频和 SHA-256。每个回合保存首次原始记录和已使用的同步版本。同步版复用目录内同一份 MCAP，整份原始包可以独立读取。

采集服务器读取环境的路径为 `/data/by/genie_sim/lerobot-reader-0cf8648`，锁定团队使用的 LeRobot 提交 `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`。作业中的 `success_counts` 和 `attempts` 可从 `job.json` 读取，实际运行进程需同时检查项目专属 Docker socket。

导出 LeRobot：

```bash
.venv/bin/python export_lerobot.py --episode /data/by/genie_sim/collection-runs/<run> \
  --output /data/by/genie_sim/exports/<new-dataset>
```

沿用团队实际读取格式 `v2.1`。当前 G2 原生状态和位置动作各为 42 维，按归档关节名称排列，全部为转动关节弧度。速度、力矩、有效值掩码和实际驱动增益/最大力矩单独保存。`action` 来自真实控制目标，不从测量状态伪造；零增益关节的对应目标不代表控制生效。

每回合的导出溯源目录保存实际执行的场景配置、任务加载脚本、记录元数据、第一条同步关节观测和模型执行前的实测初态。`control-spec.json` 按关节列出状态、控制与驱动参数的单位；每份数据集使用一种机器人和相机的数据描述。

视频保持原始帧，当前原生编码 HEVC；三路分辨率为 640×400、1280×1056、1280×1056。数据视频时间、物理时刻、实际命令时刻分别保存，检查原始三路图像、关节、TF 是否有精确对应的消息时间戳。

直接压缩采集使用原生相机帧和原物理时标，在发布回调内编码 JPEG。隔离 SDK 的 `server/command_controller.py` 只在 `GENIE_DIRECT_COMPRESSED_CAMERA=1` 时跳过旧的 `image_transport` 转发。SDK 改动前后哈希与原文件保存在服务器 `logs/phone-coupled-release-20261121/`，实际加载哈希逐回合记录在 `evidence/runtime-source.json`。旧配方运行时仍使用其自己的发布实现。

原生录制结果最多等待 1,200 秒。如果封装进程先超时，独立验收要求真实技能程序已执行成功、归档中的原生任务结果确实成功，并重新核对三路原始时标、动作与物理判据。任何缺失的源相机帧仍导致拒收。归档保留封装报告和原生任务结果。

`verify_lerobot_reader.py` 使用团队现有官方 `LeRobotDataset` 读取全部数值和每回合首、中、末三路图像。原生数据可读取，不等于 π0.5 动作归一化和执行适配已验证。训练侧还需明确策略控制维度和夹爪接口，不能直接套用旧 ALOHA 归一化。

## 新增机器人

在 `pipeline/robots/<robot_id>.json` 提供能力、手臂角色、关节/单位/限位、TCP、夹爪控制、规划模型和相机描述。在 `robot_adapters.py` 的 `ADAPTERS` 注册实际实现，将标定文件与规划模型挂载集中到适配器。仅有配置时入口返回缺少执行适配器。

新增机器人还须提供场景任务绑定和对应的原生控制、夹爪及相机运行时实现。对照 `G2ParallelAdapter`、`project_runtime_hooks.py`、`run_collection_server.py` 中 G2 的实现，校验新机器人的 URDF/USD/FK、抓取、避碰、持物代理、接触技能和录制时间轴。计划通过 `operator`/`assistant` 声明角色，各机器人独立求解轨迹。需要双臂的技能会检查 `bimanual` 能力。

## 并行采集

主槽在 A100 GPU1 使用 `127.0.0.1:50051` 与 ROS 域 77。`collection_control.py --slot secondary` 在 GPU0 使用 `127.0.0.2:50051` 与 ROS 域 78，并按回合隔离缓存和录制。两个客户端都已实测连接各自的服务器。兼容性由每回合 `source/execution-profile.json` 记录。

独立槽可用 `watch_isolated_run.py /data/by/genie_sim/collection-runs/<run>` 等待模拟器退出，并调用与主作业相同的原始录像归档和验收函数。之后由统一作业串行归入 `job.json`，同一时刻由一个进程持有主作业锁。独立槽的已验收配方复采时继续传入 `--slot secondary` 与 `--source-run`。

`queue_isolated_trial.py` 可等待独立槽空闲并使用同一个作业已验证的 Qwen 计划启动一次复测。每条回合保留自己的录制与验收报告。采用 `--source-run` 按配方补采时需要对应的物理难点复核记录。

## 复现替换后的积木回合

在主槽空闲时，使用正式回合的冻结配置、Qwen 计划和种子。新运行标签应在项目内唯一。

```bash
run=/data/by/genie_sim/collection-runs/insert-building-block-s0-iso-distinct-aperture-approach-20261122041208
label=reproduce-distinct-block-041208
.venv/bin/python collection_control.py insert_building_block --robot-id g2crsb90d \
  --scene 0 --slot primary --label "$label" --seed 1783165249 \
  --execution-mode structured --prepared-plan "$run/source/prepared-plan.json" --source-run "$run"
.venv/bin/python watch_isolated_run.py "/data/by/genie_sim/collection-runs/$label"
```

正式作业冻结在 `jobs/recovered-20261121-structured-ed63df63/job.json`。它按每任务场景 0 三条及场景 1 两条发布，要求导出前的全部运动指纹不同。原始回合和替换证明分别从 `finalization.json` 与对应的 `provenance/` 查阅。

## 限制

- 当前证据只覆盖已记录的任务/场景；15 条筛选成功不代表高泛化成功率。
- 笔帽使用官方 `PenAttachJoint` 装配近似，以部件实测相对位姿为触发条件。应用层保留此原生行为。
- 简化网格只服务规划；不通过缩物体、扩孔改变装配难度。
- API 实际 token 和请求记录保留；货币费用需实际服务账单或已确认价格。
- H20 服务器目前只用于 CPU 数据读取验证，尚未验证其 Isaac 渲染。
