# 用已有资产采集跨场景轨迹

当前平台入口支持已有任务场景和库内资产组合。作业沿用 Isaac Sim、PhysX、cuRobo 和 g2crsb90d。角色、装配件和相应接触判据由已注册任务契约提供，场景发行与代码验收结果以作业回执为准。

## 调用

在已有的 POST /v1/jobs 请求中加入 scene_request。省略 scene_request 时，后台可从自然语言提取场景要求。原有 scenes 参数继续表示六个基线任务场景里的 0/1。明确选择额外的官方任务场景时使用 scene_request.scene_ids。

~~~json
{
  "instruction": "用已有桌子和背景重新摆放星形积木和孔板，生成完整的插入轨迹",
  "task_id": "insert_building_block",
  "robot_id": "g2crsb90d",
  "scene_request": {
    "mode": "compose_from_library",
    "background_id": "common__hdr_studio__background",
    "support_id": "benchmark_table_000",
    "variation": ["background", "support", "layout", "distractors"],
    "scene_count": 1,
    "seed": 20261126
  },
  "successes_per_scene": 1,
  "execute": true
}
~~~

用 background_query 和 support_query 描述场景分类时，后台通过 Qwen Embedding 对库内候选排序。显式资产 ID 约束固定。候选需通过实际 OpenUSD 组合、水平碰撞面和放置检查。现成场景模式从已准备的官方任务包读取 USD，核对工件/接收件配对与任务角色。场景缺少适配件时会得到具体拒绝原因。

~~~json
{
  "instruction": "在官方场景2完成手机装盒，并采一条完整成功轨迹",
  "task_id": "phone_packaging",
  "scene_request": {"mode": "existing", "scene_ids": ["2"]},
  "successes_per_scene": 1,
  "execute": true
}
~~~

GET /v1/jobs/{id} 提供原生作业 ID、当前阶段、已验收数量、尝试记录和发布路径。场景搜索或物理尝试用尽预算时，作业返回失败原因和已保存回合。请求未固定已有场景 ID 或布局种子时，后台最多预备两个替代场景。初态稳定性检查未通过，或一个没有合格回合的场景连续两次在同一物理步骤失败时，可以试用已完成静态预检的候选。替换沿用原资产限制和配额，并在作业回执中记录前后场景 ID 和原因。

## 轨迹数量和重复策略

`trajectory_request.total_episodes` 设置单任务总条数。请求不同场景时同时指定 `scene_request.scene_count`。例如两个不同场景共两条，每个场景分配一条。`successes_per_scene` 适用于明确的每场景配额。固定场景及预检失败的名额会保留在配额结果里。

`duplicate_policy` 默认使用 `distinct`。编排根据实际资产和布局检查场景差异，录制后独立核对实测初态、动作哈希与按路程重采样的关节和腕部路径。`distinct_dimensions` 可指定必须变化的场景维度。`lineage_id` 让后续作业与同一数据集已登记的回合一起查重。正式发布回执提供 `trajectory_diversity`，作业的 `trajectory_outcome` 提供请求、已验收、已发布和缺少的条数。

允许重复时传入 `duplicate_policy: allow`。每个名额仍需一段新录制。要求按某一条参考轨迹复验时，传入以下字段，其中 `reference_run` 使用已验收回合的运行 ID。

~~~json
{
  "instruction": "在原场景重新执行参考回合两次",
  "trajectory_request": {
    "duplicate_policy": "replay",
    "reference_run": "scene-reference-run-id",
    "total_episodes": 2
  },
  "execute": true
}
~~~

复验会在参考回合的槽位重新启动隔离仿真，固定已封存的场景、初始技能计划、程序、种子和镜像。完成物理后，验收器比较回合实测轨迹及实际下发的关节位置目标，按既定容差确认与参考一致。回执的方法是 `immutable_recipe_reexecution`。参考缺少证据或当前资产版本与参考不同，将返回 `REPLAY_REFERENCE_UNAVAILABLE`。

## 物理执行中的自动恢复

| 触发条件 | 后台动作 | 可核查记录 |
| --- | --- | --- |
| 笔帽对接前笔轴发生偏移 | 连续采样确认笔仍被夹稳，以实测笔轴重算对接目标。局部路径不可行时，先试用原生关节空间规划到对接区外侧，再沿实测轴靠拢。若仍无可行路径且零件保持足够间距，有限搜索双臂都可达的不同呈递姿态。 | `pen_alignment_reference`、`cap_preinsert_route_rejected`、`cap_outboard_connector_candidates`、`paired_alignment_workspace_candidates` |
| cuRobo 的局部关节路径中途跳到另一组逆运动学解 | G2 的局部规划优先保留衔接上一点的逆运动学候选。生成路径后再检查相邻关节目标，超过 0.18 rad 的单步变化会拒绝下发。搬运技能可调用原生关节空间规划重新连接，再验证实测夹持。 | `local-joint-plan-continuity.jsonl`、`shape_transport_joint_space_reconnect` |
| 积木搬运中检测到小幅滑移 | 确认积木仍悬空且姿态已稳定，按实测夹持关系刷新 cuRobo 的碰撞代理，执行 6 mm 试提。只有积木实际随手运动且跟踪误差合格才续跑。明显滑移或试提失败结束当前回合。 | `shape_grasp_recovery_probe`、`shape_transport_grasp_slipped` |
| 积木搬运的直线路径被拒或执行后有位姿误差 | 直线路径被拒时，按实测位姿生成抬高或侧移的避碰路径，试图抵达同一个目标。路径可行但跟踪有小幅误差时，在夹持可信的条件下进行一次有上限的实测补偿。通过原路点要求后继续。 | `held_waypoint_detour_applied`、`shape_transport_tracking_correction`、`shape_transport_grasp_tracking` |
| 持物搬运的关节空间重接导致夹持偏转 | G2 在重组场景的积木搬运中使用 RobotAPI 原生降速选项，将重接路径的执行时长设为两倍。每次重接后重新测量工件是否随夹爪运动。 | `shape_transport_joint_space_reconnect`、`shape_transport_grasp_tracking` |
| 闭爪后的长距离抬升在当前腕部分支不可行 | 换桌左臂积木先试提 12 mm，检查工件确实升起且当前手臂的规划代理正确。从实测位姿尝试完整抬升，失败后继续夹住工件，改用不超过 31 mm 的分段抬升。每段复查工件跟随和高度，最终仍需实际抬升超过 55 mm。 | `grasp_lift_probe`、`grasp_probe_payload_attachment`、`grasp_lift_short_segment_completed`、`pick_result` |
| 持物重接的中段可能擦碰装配件 | 从当前实测抓取关系与 cuRobo 正运动学计算候选中每个采样点的工件范围，挑选对登记装配件保持正间隔的路径。积木可以先向外、向上绕行。笔帽可以在外侧连接完成后用同一检查寻找保留装配间隙的对口路径。 | `held-transport-clearance.jsonl`、`guarded_transport_detour`、`cap_preinsert_guarded_joint_space` |
| 积木在精确穿孔时容易被开爪动作带偏 | 先逐步降低夹持力，再低力小幅开爪。实测物体脱离后才完全张开。穿孔仍按原始几何复算完整物理截面。 | `low_effort_gripper_separation`、`hole-crossing-audit.json` |
| 手机装盒的内托容易在开爪时偏转 | 对重组场景的内托使用原生驱动限力卸载与缓慢开爪，逐阶段用盒体、手机、内托和盒盖的实测几何验收。 | `packing_release_gate`、`gentle_release_effort_unload` |
| 台面凸分解改变实际支撑高度，或探针先碰到桌上物品 | 固定台面使用资产原始三角面。九条 PhysX 射线查询全部命中，再按登记路径识别桌面并记录遮挡物。每个桌面命中仍须与源网格高度相差不到 2 mm。 | `scene-physics-preflight.json`、`nearest_occluder` |
| G2 腕部超出原生 1.92 rad 联动限位 | 重组笔帽和左臂积木可在端点 IK 返回前检查最多八个原生候选。批规划在锁定成功候选前检查整段路径，最终选择和发布前也复核。抓取被拒且现场位姿基本未变时可尝试备选抓取。 | `coupled-wrist-endpoint-ik.jsonl`、`coupled-wrist-plan-candidates.jsonl`、`grasp_candidate_rejected` |
| 双臂的呈递候选均因腕角无法执行 | 按登记装配轴额外采样六个共同滚转角，保留装配轴与预插入间隙。用当前持物关系和双臂原生 IK 筛选，执行后重新测量末端到位与握持。 | `cap_presentation_roll_candidates`、`cap_presentation_roll_selected` |
| 笔帽对口后接触前的轴向微进无法规划 | 优先保持笔静止，只移动帽。按同一相对插入目标尝试移动持笔臂，运动候选逐点检查两件的外廓间隔。回复成功前核验实测相对推进、对口和剩余间隙。接触阶段仍用原细步插入。 | `cap_axial_guarded_advance_result`、`cap_fine_advance_measured` |
| 三路相机在空闲启动段晚于状态录制就绪 | 原始消息共同验证首个完整帧。只有缺帧都在首次运动指令前、关节和工件保持静止时，才从首个完整帧重提取。提取与同步审计使用相同时间精度，逐帧检查保留段并重新解码录像。 | `capture-header-audit-before-idle-trim.json`、`idle-capture-prefix.json`、`aligned-archive.json` |

每次候选路径、实测偏差和恢复失败都会进入回合证据。局部恢复用尽后，结构化重规划从当前状态和已验收阶段继续。超出已支持的资产、能力或恢复范围时，作业返回失败事件和已验收回合。最终由独立验证器复算物理关系以及三路录像与控制指令的同步。

跨场景 G2 作业还会根据归档 URDF 的关节速度限制，核查原始驱动目标及按 30 Hz 对齐的训练动作。未通过的回合保留物理证据，后台会继续采集训练配额。跨版本接续时，`prepare_cross_scene_remainder.py` 可以核对已有 LeRobot 读取回执与这一动作标准，去重后生成剩余配额。

共享数据盘低于启动水位时，工作进程记录 `waiting_for_storage`，每 30 秒复查空间，在原预算内恢复。等待期间保持预留回合及原有数据。

完整发布包括 LeRobot 读取回执和集中原始轨迹清单。新的集中归档中，同盘视频和 MCAP 使用硬链接节省空间。通过 rsync -aH 下载时可保留文件关系。

## 实验室启动

使用已有采集镜像、专用 Docker socket、官方资产和 Qwen 专用账号。在独立项目目录配置 GENIE_PROJECT_ROOT。将相同修订的三个任务模板放入 tasks，额外的官方任务场景由固定上游源码的 benchmark/config/llm_task 复制到 official-task-scenes/<task>/<source-id>，每份至少包含 scene.usda、scene_info.json、instructions.json。场景编排的宿主机 Python 需要 OpenUSD。实验室此次选用与 Python 3.12 兼容的 usd-core 25.11。采集容器继续使用镜像内原有的 Isaac SDK。

~~~bash
python natural_backend.py --request /path/to/request.json
python batch_collect.py --job /path/to/job.json --slot secondary --wait-for-worker
~~~

第一次可设 execute=false 只预备 Qwen 计划与场景。已开始的原生作业使用同一 job.json 续接。平台 Worker 重启后依据持久化的原生作业 ID 接续，匹配已有容器归属和已验收回合。作业执行期间固定代码版本、场景摘要、初始计划与实际启动回执。

用以下命令在已配置的实验室运行冻结验收矩阵。加 --prepare-only 可以检查预检结果，后续使用同一 --run 接续。每组发布独立的 LeRobot 数据集和原始轨迹目录。

~~~bash
python validate_cross_scene_release.py pipeline/cross-scene-acceptance.json --run cross-scene-g2 --slots primary secondary gpu0-extra gpu1-extra
~~~

当前实验室两张卡的槽位按 GPU UUID 登记，GPU0 有 secondary 和 gpu0-extra，GPU1 有 primary 和 gpu1-extra。各槽使用独立 gRPC 回环地址和 ROS 域。每次采集拥有自己的场景副本、模型响应记录、录制目录和缓存。新增槽的共享源码、运行环境与资产按只读挂载，只有当前回合的工作目录可写。

新增槽的容器内存上限为 16 GiB。启动前须留出该槽的内存预算、3 GiB 系统余量和其他新增槽可能继续增加的内存，并确认目标卡至少有 10 GiB 空闲显存。主机最近十秒的全局内存停滞比例达到 5% 时也暂停新启动。后台记录 waiting_for_host_memory、waiting_for_host_memory_pressure 或 waiting_for_gpu_memory，并在原预算内接续同一预留回合。

## 适配一个新任务

在 pipeline/task_contracts 下新增可信的任务描述和同名 Python 适配器。加入经过物理验证的 tasks/<机器人场景前缀><任务ID>/0 样例，必要时加入第二个兼容模板。适配器给出物体绑定、目标和物理判据。已有抓取、搬运、释放、稳定性检查与局部运动恢复由共享执行器提供。适配器提供以下函数。

| 函数 | 责任 |
| --- | --- |
| bind(scene, robot_id) | 绑定角色、工件、配对几何、可用抓取、目标、技能要求和执行约束 |
| resolve_target(geometry, ref, obj) | 从实测场景求末端或工件目标，返回候选和接触阶段所需规划信息 |
| validate_steps(plan, spec, held, achieved) | 验证任务动作顺序和恢复时可继续的阶段 |
| verify_relation(executor, parameters) 与 verify_task(executor) | 在实际执行中检查关系和最终物理状态 |
| audit(run, raw, frames) | 从原始实测数据重新计算成功判据，返回布尔检查项与度量 |

部署端只加载经过代码审核和物理验证的契约，实际任务文件会随运行冻结。涉及新接触控制时先注册并验收新的技能，再用于新场景。

当前支持域是水平刚性台面、已经登记的装配件组合与现有夹爪。新增装配件组合、工作区高度及场景变体需要通过独立验收。盖笔帽使用官方 PenAttachJoint 近似，需随数据保留该来源。
