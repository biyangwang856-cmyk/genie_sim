# 平台接入与 Genie Sim 扩展

`GET /v1/capabilities` 可在无 GPU、无 Qwen 密钥的开发机上查询接口。平台提交作业后立即得到可轮询的 ID。单独的 Worker 在实验室执行已有采集后台并转交原生状态，HTTP 进程只使用 CPU。

## 启动与调用

在同一仓库根目录、同一用户的两个终端使用同一个本地磁盘目录。实验室 Worker 使用项目 `.venv/bin/python` 和 [采集手册](../PIPELINE.md) 中的 Docker 环境。服务监听回环地址，平台可在已有鉴权网关后挂载 `create_app(state_dir)`。

```bash
python platform_api.py --state-dir .platform-state serve --port 8786
# 第二个终端，运行在采集服务器
.venv/bin/python platform_api.py --state-dir .platform-state work
```

```bash
curl -s http://127.0.0.1:8786/v1/capabilities
curl -i -X POST http://127.0.0.1:8786/v1/jobs -H 'Content-Type: application/json' -H 'Idempotency-Key: phone-scene0-preview-01' --data '{"instruction":"在场景0采集手机装盒轨迹","simulator":"isaac_physx","robot_id":"g2crsb90d","task_ids":["phone_packaging"],"scenes":["0"],"execute":false}'
# 从上一个响应的 Location 复制路径
curl -s http://127.0.0.1:8786/v1/jobs/<id>
```

`execute=false` 为默认值，真实调用语言模型和 Embedding，准备计划。正式申请将它设为 `true` 并使用新的幂等键。默认双场景采集在场景 `0` 目标为 3 条，在场景 `1` 目标为 2 条，可只选择其中一个场景。未显式填写 `task_ids` 时由 Qwen 识别任务目标。每次新配方仍使用原有物理验收与难点复核机制。

| 路径 | 输入 | 响应 |
| --- | --- | --- |
| `GET /v1/openapi.json` | 无 | [OpenAPI 3.1 接口描述](openapi.json)，可导入联调工具 |
| `GET /v1/capabilities` | 无 | `operations` 列出 ID、上游目录、所需输入、返回字段、可执行状态和已接通的采集范围 |
| `POST /v1/jobs` | 必填 `instruction`，选填 `simulator`、`robot_id`、`task_ids`、`scenes`、`execute` | `202`，返回 `id`、`status_url`，响应头含 `Location` |
| `GET /v1/jobs/{id}` | 平台 ID | `state`、`native_job_id`、可用时的 `native.success_counts`、`attempt_count` 和 `result` |
| `POST /v1/extensions/{operation}` | 按能力清单传入该操作的必填字符串参数 | 已接入的扩展进入同一队列。预留扩展返回 `422 unsupported`，列明缺项 |

工作队列使用 SQLite。`queued` 等待 Worker，`running` 已被领取，`finished` 表示原生调用已返回。请读 `result.status` 判断业务结果。只有 `complete` 并带已验收的 `result.dataset` 才能发布训练数据。`skill_plans_prepared` 只有计划。`needs_hardpoint_check` 等待同版配方难点复核。`failed` 保留错误类别。Worker 在物理执行前持久化原生作业 ID，查询时可获得收集计数和服务端数据路径。该路径供平台有权限的文件服务读取。

相同 `Idempotency-Key` 加相同请求返回同一 ID，重用键并改变参数返回 `409`。无效 JSON 或字段返回 `400`，未登记的机器人、任务、场景或后端返回 `422`。每个状态目录同时允许一个 Worker。意外中断后，重新启动 Worker 将上次运行中的平台作业标为 `interrupted`，保留原生 ID。可在实验室据此检查实际进程、原生日志和已入库回合，再按采集手册接续。`work --once` 适合已有调度器轮询。

## 轨迹配额与重复策略

`trajectory_request.total_episodes` 是一个任务的总条数。`successes_per_scene` 是每个场景的配额。两者同时给出时，数量须与请求的场景数相符。`distinct_dimensions` 指定场景之间必须有差异的维度。`lineage_id` 用于在同一训练数据谱系下比较不同作业的动作。

```json
{
  "instruction": "在两个不同的现有场景各采一条手机装盒轨迹",
  "task_id": "phone_packaging",
  "scene_request": {"mode": "existing", "scene_ids": ["0", "1"]},
  "trajectory_request": {
    "total_episodes": 2,
    "duplicate_policy": "distinct",
    "distinct_dimensions": ["layout"]
  },
  "execute": true
}
```

`distinct` 是默认策略，按实际执行的场景和物理运动判重。`allow` 允许独立实跑的运动相同，每次仍需有新的运行回执。`replay` 需要 `reference_run`，从已验收的冻结配方复执行，并比较新的实测运动与参考回合。参考回合需保留原场景、源码、首次计划、镜像、种子、运行槽位和独立验收回执。配额不足时，`native.trajectory_outcome` 显示已发布数和缺额。

## 技能缺口与自动验证

自然语言可提出较窄的目标，例如“只把手机放进盒底，内托和盒盖留在原位”。后台通过已有装盒资产和受支持的物理关系生成技能计划，原始记录会核查手机包含关系和附带的保留条件。

对于目录外的桌面抓放目标，例如“采集一条把优盘放进白色中空包装盒的轨迹”，可省略 `task_id`。后台检索刚性工件和开口容器，从容器的原始碰撞几何计算开口、深度和支撑面。随后复用抓取、持物移动、释放和撤手技能。验收从原始记录复核松爪后工件整体位于盒内并保持稳定。

`ASSET_NOT_FOUND` 表示索引中没有找到匹配资产。`ASSET_GEOMETRY_INCOMPATIBLE` 表示现有候选未满足抓放几何要求。`missing_assets` 返回对象或几何缺口。涉及开盖、多工件、工具等操作时，后台检查对应任务技能。

原生技能可通过自然语言或 `skill_goal` 提交。铰链微波炉的候选映射使用 `RobotAPI.open`。下面的请求准备能力映射和试验场景。`execute=true` 时，服务器执行隔离物理验证，并在测量通过后发布与代码、机器人和资产对应的证书。自然语言中的物理开启端由资产绑定解析成关节的有向限位。

```json
{
  "instruction": "抓住把手，将微波炉门打开到开启限位",
  "skill_goal": {
    "skill": "open_hinged_lid",
    "actor": "benchmark_microwave_oven_003",
    "target_limit": "min"
  },
  "execute": false
}
```

状态查询返回候选接口、缺少的技能或观测、验证阶段和回执。没有可用技能时返回 `SKILL_NOT_REGISTERED` 和 `missing_skills`。已有接口需要满足的装配关系、控制方式和物理观测由能力映射声明。

`rotate_held_about_tool` 通过 `RobotAPI.rotate_actor` 验证持物旋转。当前证书适用于 `g2crsb90d`、右臂、已标注的手机和绕世界 +Z 的 +15° 旋转。使用 `completion: "pack_phone"` 时，运行时先核对同一范围的证书，物理转腕通过后，从实测夹持状态生成后续装盒计划。证书、续执行见证和失败原因可从该作业的 `runtime_capabilities` 与 `validation` 读取。

```json
{
  "instruction": "拿起已标注的手机，保持工具位置绕世界 +Z 转动 +15 度，然后只把手机放入盒底，内托和盒盖留在原位。",
  "skill_goal": {
    "skill": "rotate_held_about_tool",
    "actor": "benchmark_phone_000",
    "angle_deg": 15,
    "completion": "pack_phone"
  },
  "execute": false
}
```

## 后续怎样接另一个后端

`genie_extensions.Registry` 按 ID 连接可信实现。`isaac_physx` 的实现直接调用 `natural_backend.submit`。仿真后端共享请求字段，各适配器自行声明它能处理的机器人、任务及场景。团队在适配器内调用相应引擎的规划、控制、录制与验收。独立验收合格后，再由部署代码在 Web 和 Worker 两侧注册同一适配器。

```python
from genie_extensions import Registry

registry = Registry()
# collect_with_mujoco 由团队实现，签名为 (payload, on_native_job) -> dict。
# 原生作业建立后调用 on_native_job(native_id)，返回至少包含 status、job_id 的字典。
registry.register("mujoco", collect_with_mujoco, scope={
    "robot_ids": verified_robots, "task_ids": verified_tasks, "scenes": verified_scenes,
})
# Web 传入 create_app(state_dir, registry)，Worker 使用 jobs.run_next(registry)。
```

| 已预留的范围 | 操作 ID | 接入时应返回 |
| --- | --- | --- |
| 原生 MuJoCo、Newton（含上游 MuJoCo-Warp 路线）、独立 libuipc | `mujoco`、`newton`、`libuipc` | 原生作业与任务状态，自行验证控制、接触、时钟和录制 |
| 场景编排、已有新场景准入、场景重建、全景世界生成 | `scene_authoring`、`scene_admission`、`scene_reconstruction`、`world_generation` | 对应场景、验收或世界资源引用 |
| VR 遥操作、ROS 控制、MoveIt 规划 | `teleoperation`、`ros_control`、`moveit_planning` | 录制、会话或轨迹计划引用 |
| 基准评测、Agent2Act、Dream、RLinf | `benchmark`、`agent2act`、`dream`、`rl_training` | 评测、动作计划、预测或检查点引用 |

`/v1/jobs` 的 `scene_request` 接受已有官方任务场景和库内资产组合，并进入自动试采与验收。独立的 `/v1/extensions/scene_admission` 仍保留作扩展接口。完整参数、资源和复现命令见 [跨场景采集](cross-scene.md)。
