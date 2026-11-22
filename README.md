# Genie Sim 自然语言仿真采集后台

用户提交操作指令，后台调用 Qwen 理解需求并检索已有资产。结构化计划由技能执行器逐步交给 RobotAPI、cuRobo 和 Isaac Sim。独立验收通过后，系统归档实测状态、实际下发动作、三路视频和 LeRobot v2.1 数据。

## 团队从哪里开始

| 工作 | 入口 |
| --- | --- |
| 下载资产、采集镜像、原始回合、训练集，排查下载或构建问题 | [下载与排错](docs/downloads.md) |
| 接入平台，提交自然语言作业，轮询进度和数据位置 | [平台接口、示例与扩展协议](docs/platform.md) |
| 改变机器人、技能、规划、采集判据或重现某条轨迹 | [采集运行手册](PIPELINE.md) |
| 查阅来源、验收数字、外部依赖版本 | [版本清单](release/reference.json) · [第三方来源](release/NOTICES.md) |

## 已验收的范围

`g2crsb90d` 的三个任务各有两个预注册场景。正式作业取得 15 条不同轨迹，46,718 帧，45 路回合视频。50 次已关联且结束的尝试中，18 次通过物理与录制检查，剔除 3 次精确重复后交付 15 条。训练集的实际状态和动作均为 42 维，官方读取器已核对数值并读取三路图像。

| 任务 | 平台任务 ID | 场景 0 | 场景 1 | 训练帧 |
| --- | --- | ---: | ---: | ---: |
| 手机装盒 | `phone_packaging` | 3 | 2 | 19,675 |
| 盖笔帽 | `put_the_cap_on_the_pen_new` | 3 | 2 | 17,144 |
| 积木插入 | `insert_building_block` | 3 | 2 | 9,899 |

这组证据覆盖表内六个组合。笔帽沿用官方 `PenAttachJoint` 套合近似。新场景自动发现与准入、第二种机器人和其他采集引擎均留有接口，尚待验证。平台 HTTP 接口的本地测试与上述服务器物理验收分别记录。

## 这次提交了什么

基线提交 `f97e948` 的名称为 `v0-3work2changjing`。它冻结了 32 个文件的采集实现。其后单独提交了平台入口、上游扩展接口、实验室实际输入和交接文档。`git show --stat f97e948` 可查看基线差异，`git diff --stat f97e948..HEAD` 可查看接口交付差异。

| 代码范围 | 团队维护内容 |
| --- | --- |
| `natural_backend.py`、`prepare_skill_plans.py`、`asset_retrieval.py`、`structured_planner.py` | 自然语言路由、Embedding 检索、资产绑定、受约束技能计划与重规划 |
| `robot_adapters.py`、`pipeline/robots/`、`skill_registry.py`、`skill_executor.py`、`task_geometry.py` | 机器人能力和角色、技能前置条件、执行顺序、任务几何 |
| `collection_control.py`、`run_collection_*.py`、`project_runtime_hooks.py`、`batch_collect.py` | 隔离采集、源码冻结、实测动作与三路图像、批量配额与恢复 |
| `audit_*.py`、`review_hardpoint.py`、`certify_recipe.py` | 从记录重算成功判据、检查动作重复、复核抓取与释放等难点 |
| `finalize_job.py`、`export_lerobot.py`、`verify_lerobot_reader.py`、`bundle_raw_trajectories.py` | 正式数据发布、官方读取器验证和原始回合集合 |
| `platform_api.py`、`platform_jobs.py`、`genie_extensions.py` | 版本化 HTTP 接口、持久作业队列和可信实现注册 |
| `tasks/`、`configs/`、`runtime_inputs/` | 六份实际运行场景、各任务描述、G2 标定、几何证据和两份 SDK 补丁 |

## 新同学本地自检

在仓库根目录使用 Python 3.11 或更新版本。下面两项只读校验与接口测试可在 Windows 或 Linux 上运行。

```bash
python manage_runtime_inputs.py
python -m unittest discover -s tests -v
```

启动接口后可先访问 `http://127.0.0.1:8786/v1/capabilities`。服务和 Worker 命令见平台接口文档。正式仿真沿用实验室 Linux、项目专用 Docker、资产和私有 Qwen 配置。新主机需要按下载指南准备同版资源并适配挂载与设备。

Git 管理源码、场景描述、校准、补丁和小型校验清单。资产本体、镜像、录制和模型凭据由运行环境提供。原始上游源码按提交固定，团队可从同版源码继续开发。早期环境摸底记录保存在 [初始记录](docs/initial-environment.md)。
