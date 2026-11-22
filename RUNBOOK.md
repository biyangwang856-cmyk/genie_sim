# Genie Sim 服务器采集入口

更新：正式入口、结构化技能执行、不可变版本补采和当前数据语义见 [PIPELINE.md](PIPELINE.md)。下方保留早期基线的路径说明；新作业使用 `natural_backend.py`，不能将旧任务选择器视为 LLM 技能组合的验收。

实际仿真服务器工作区：`/home/vipuser/by/genie_sim`。
全部仿真和 API 请求在服务器执行；本目录是对应脚本的本地编辑副本。

## 从用户语言创建作业

```bash
cd /home/vipuser/by/genie_sim
.venv/bin/python run_language_job.py \
  '用g2crsb90d生成手机装盒、盖笔帽和积木插入三个任务，每个任务5条完整成功轨迹，覆盖两个场景'
```

入口实际调用 Qwen `qwen-plus`，用 `text-embedding-v4` 检索已注册资产，再绑定官方已有任务。
首批固定为每任务场景 0 三条、场景 1 两条。
密钥由 `secrets/qwen.json` 私有读取，不写入作业、视频或日志。

首次执行每对任务/场景需要在操作难点核验技术证据；其余阶段由自动判据覆盖。
`recipe-validation/<任务>/<场景>.json` 是执行程序版本的技术验收记录，包含源码和机器人模型哈希，不是用户审批。
未验证的程序最多先运行一个回合；只有完整成功且难点核验通过，才累加配额并继续同版程序。
这个门槛目前仍在调试，不能据入口存在宣称三个任务或15条数据已完成。

## 继续现有作业

```bash
.venv/bin/python batch_collect.py \
  --job jobs/20261119-185730-ed63df63/job.json
```

`--tasks phone_packaging put_the_cap_on_the_pen_new --scenes 0` 可只推进对应子集。
运行单回合可用 `collection_control.py <任务> --scene 0 --label <新回合名>`。
它拒绝同时启动两个项目采集容器，且拒绝覆盖已有回合目录。

## 产物和验收

- 每回合：`/data/by/genie_sim/collection-runs/<label>`，含源码、校准 URDF 快照、日志、阶段物理证据、实际控制命令和验收 JSON。
- 原始录制：`/data/by/genie_sim/runtime-collection/recording_data/<回合名>`，含 MCAP、`state.json`、`aligned_joints.h5` 和元数据。
- 原始三路视频：原始录制目录的 `observations/videos/{head,hand_left,hand_right}_color.mp4`。
- `audit_episode.py <回合目录>` 检查完整任务判据、真实抬升、无回滚、有限控制量、持物规划几何、相机更新异常、视频全解码、帧数一致和空白画面。
- 一次失败后的回滚探索可保留为调试记录，不能算作连续成功的训练回合。成功额度按新进程完整重执行并通过验收的回合计数。
- 当前原生 HDF5/MCAP 不是已经验证的 LeRobot 或 π0.5 输入；仓库提供 AgiBot→LeRobot v2.1 转换器，仍需针对本次模型版本和动作语义验证后再交付。

物理运行使用官方资产和 PhysX；规划 attachment 只描述持物碰撞几何，不将物体焊到机器人。
笔帽资产自带 `PenAttachJoint` 锁合近似，必须在数据元信息中披露。
本项目使用独立 Docker socket 和 GPU1 CDI，禁止启动到其他实验的 GPU0。

当前三个任务都是刚体任务；为排查GPU物理结果与图像不同步，默认使用CPU PhysX，cuRobo和渲染继续使用A100。
启动清单和 `source/execution-profile.json` 记录实际后端；`--physics-backend gpu` 只用于独立诊断。
