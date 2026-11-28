# Genie Sim 自然语言仿真采集后台

用户提交操作指令，后台调用 Qwen 理解需求并检索已有资产。结构化计划由技能执行器逐步交给 RobotAPI、cuRobo 和 Isaac Sim。独立验收通过后，系统归档实测状态、实际下发动作和三路视频，直接生成 LeRobot Dataset v3.0。历史 v2.1 包仍可按原版本读取。

## 团队从哪里开始

| 工作 | 入口 |
| --- | --- |
| 下载资产、采集镜像、原始回合、训练集，排查下载或构建问题 | [下载与排错](docs/downloads.md) |
| 接入平台，提交自然语言作业，轮询进度和数据位置 | [平台接口、示例与扩展协议](docs/platform.md) |
| 改变机器人、技能、规划、采集判据或重现某条轨迹 | [采集运行手册](PIPELINE.md) |
| 查阅来源、验收数字、外部依赖版本 | [版本清单](release/reference.json) · [第三方来源](release/NOTICES.md) |

## 已验收的范围

`g2crsb90d` 的三个任务各有两个预注册场景。首批正式采集从 50 次已结束的尝试中验收了 15 条不同轨迹。当前初始场景精简包按每个任务和场景各留一条，共 6 条、18,654 帧。状态和动作均为 42 维，LeRobot 官方读取器验证了数值和三路图像。下载口径见 [数据下载说明](docs/downloads.md)。

| 任务 | 平台任务 ID | 场景 0 | 场景 1 | 训练帧 |
| --- | --- | ---: | ---: | ---: |
| 手机装盒 | `phone_packaging` | 3 | 2 | 19,675 |
| 盖笔帽 | `put_the_cap_on_the_pen_new` | 3 | 2 | 17,144 |
| 积木插入 | `insert_building_block` | 3 | 2 | 9,899 |

这组证据覆盖表内六个组合。笔帽沿用官方 `PenAttachJoint` 套合近似。跨场景作业已验证官方场景接入和库内桌面、背景与摆放重组，逐任务和场景的物理结果见验收清单。[跨场景用法](docs/cross-scene.md) · [跨场景验收清单](pipeline/cross-scene-acceptance.json)

## v0-4-crossscene-skill-reuse

本版以团队已上传的 `92b5fef` 为基础。

- 跨场景编排按场景数与轨迹总数分配配额。默认检查布局和实测运动是否重复，显式复执行可沿用已验收回合的冻结配方。两种不同手机装盒布局和一次同轨迹复执行共验收 3 条、4,278 帧。[跨场景与重复策略](docs/cross-scene.md)
- 后台按技能效果查找 RobotAPI 能力。独立物理验收通过后按机器人、资产与运行版本发布证书。G2 右臂已验证 +15° 手机转腕，以及先复用转腕再由 Qwen 计划装盒的连续操作。证书按实际运行版本匹配，新版运行环境通过单项验证重新发布证书。[技能与接口](docs/platform.md)
- 目录外 `place_in_container` 复用抓取、搬运与释放技能，从资产几何计算盒口、抓持和放置条件。白色 4B 橡皮入盒在场景 0 成功采集 1 条、1,444 帧，原始数据校验和 LeRobot v2.1 读回通过。当前物理证据覆盖该物体组合与场景。[自然语言入口](PIPELINE.md#后台入口) · [数据与证据](docs/downloads.md)
- 轨迹工作台提供自然语言提交、作业进度和已验收三路视频预览。服务绑定实验室主机的回环端口，成员使用各自的 OpenSSH 配置连接。

多机器人与其他采集引擎保留平台扩展接口。当前物理结果来自 `g2crsb90d` 与 Isaac Sim。微波炉开门候选已验证接口发现和失败分类，门板净空仍限制该技能的物理验收。

| 代码范围 | 团队维护内容 |
| --- | --- |
| `natural_backend.py`、`prepare_skill_plans.py`、`asset_retrieval.py`、`structured_planner.py` | 自然语言路由、Embedding 检索、资产绑定、受约束技能计划与重规划 |
| `robot_adapters.py`、`pipeline/robots/`、`skill_registry.py`、`skill_executor.py`、`task_geometry.py` | 机器人能力和角色、技能前置条件、执行顺序、任务几何 |
| `container_goal.py`、`container_scene.py`、`pipeline/task_contracts/` | 目录外入盒意图、对象几何与独立接触验收 |
| `scene_composer.py`、`scene_admission.py`、`trajectory_*.py` | 场景重组与准入、轨迹去重和复执行 |
| `skill_capabilities.py`、`skill_onboarding.py`、`skill_recovery.py` | 原生能力映射、范围证书与实测状态恢复 |
| `collection_control.py`、`run_collection_*.py`、`project_runtime_hooks.py`、`batch_collect.py` | 隔离采集、源码冻结、实测动作与三路图像、批量配额与恢复 |
| `audit_*.py`、`review_hardpoint.py`、`certify_recipe.py` | 从记录重算成功判据、检查动作重复、复核抓取与释放等难点 |
| `finalize_job.py`、`export_lerobot.py`、`lerobot_v3.py`、`verify_lerobot_reader.py`、`bundle_raw_trajectories.py` | v3.0 分片、正式数据发布、官方读取器验证和原始回合归档 |
| `platform_api.py`、`platform_jobs.py`、`genie_extensions.py` | 版本化 HTTP 接口、持久作业队列和可信实现注册 |
| `tasks/`、`configs/`、`runtime_inputs/` | 六份实际运行场景、各任务描述、G2 标定、几何证据和两份 SDK 补丁 |

## 新同学本地自检

在仓库根目录使用 Python 3.11 或更新版本。下面两项只读校验与接口测试可在 Windows 或 Linux 上运行。

```bash
python manage_runtime_inputs.py
python -m unittest discover -s tests -v
```

启动接口后可先访问 `http://127.0.0.1:8786/v1/capabilities`。服务和 Worker 命令见平台接口文档。正式仿真沿用实验室 Linux、项目专用 Docker、资产和私有 Qwen 配置。新主机需要按下载指南准备同版资源并适配挂载与设备。

实验室运行环境就绪后，在仓库根目录运行 `bash trajectory_studio/start_server.sh`。可通过 `GENIE_STUDIO_ENV` 指定已有的部署环境脚本，并通过 `GENIE_STUDIO_RUNTIME` 指定已部署的仓库目录。成员在本地运行 `python trajectory_studio/open_studio.py --host genie-lab`，其中 `genie-lab` 为各自在 SSH 配置中的别名。多版并行时在两端使用相同的 `GENIE_STUDIO_PORT`。

Git 管理源码、场景描述、校准、补丁和小型校验清单。资产本体、镜像、录制和模型凭据由运行环境提供。原始上游源码按提交固定，团队可从同版源码继续开发。早期环境摸底记录保存在 [初始记录](docs/initial-environment.md)。
