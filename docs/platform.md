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

`execute=false` 为默认值，真实调用语言模型和 Embedding，准备计划。正式申请将它设为 `true` 并使用新的幂等键。当前每个任务在场景 `0` 目标为 3 条，场景 `1` 目标为 2 条，可只选择其中一个场景。未显式填写 `task_ids` 时由 Qwen 在已登记任务中选择。每次新配方仍使用原有物理验收与难点复核机制。

| 路径 | 输入 | 响应 |
| --- | --- | --- |
| `GET /v1/openapi.json` | 无 | [OpenAPI 3.1 接口描述](openapi.json)，可导入联调工具 |
| `GET /v1/capabilities` | 无 | `operations` 列出 ID、上游目录、所需输入、返回字段、可执行状态和已接通的采集范围 |
| `POST /v1/jobs` | 必填 `instruction`，选填 `simulator`、`robot_id`、`task_ids`、`scenes`、`execute` | `202`，返回 `id`、`status_url`，响应头含 `Location` |
| `GET /v1/jobs/{id}` | 平台 ID | `state`、`native_job_id`、可用时的 `native.success_counts`、`attempt_count` 和 `result` |
| `POST /v1/extensions/{operation}` | 按能力清单传入该操作的必填字符串参数 | 已接入的扩展进入同一队列。预留扩展返回 `422 unsupported`，列明缺项 |

工作队列使用 SQLite。`queued` 等待 Worker，`running` 已被领取，`finished` 表示原生调用已返回。请读 `result.status` 判断业务结果。只有 `complete` 并带已验收的 `result.dataset` 才能发布训练数据。`skill_plans_prepared` 只有计划。`needs_hardpoint_check` 等待同版配方难点复核。`failed` 保留错误类别。Worker 在物理执行前持久化原生作业 ID，查询时可获得收集计数和服务端数据路径。该路径供平台有权限的文件服务读取。

相同 `Idempotency-Key` 加相同请求返回同一 ID，重用键并改变参数返回 `409`。无效 JSON 或字段返回 `400`，未登记的机器人、任务、场景或后端返回 `422`。每个状态目录同时允许一个 Worker。意外中断后，重新启动 Worker 将上次运行中的平台作业标为 `interrupted`，保留原生 ID。可在实验室据此检查实际进程、原生日志和已入库回合，再按采集手册接续。`work --once` 适合已有调度器轮询。

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

`scene_admission` 是团队预留协议。后续实现需绑定场景与资产版本、抓取和装配几何、机器人可达性与避碰，并验收一条完整物理轨迹。当前用户提交 `scenes:["2"]` 会收到缺失场景。上游源码中可独立启动的某模块，其在本平台的状态仍以 `capabilities` 为准。
