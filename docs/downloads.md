# 资产、采集镜像和轨迹怎么下载

## 选资源

| 资源 | 实验室可取得的版本 | 官方获取方式 |
| --- | --- | --- |
| 3D 资产库 | `/home/vipuser/by/genie_sim/GenieSimAssets`，20,435 个发布文件，43,177,589,868 字节 | ModelScope 整库固定修订，或按官方 CLI 下载任务子集 |
| 采集镜像 | 项目专属 Docker 内的 `geniesim-by-collection:a100-20261119`，镜像展开大小约 35.28 GB | 获取上游采集镜像，或用固定版源码构建。团队镜像另有依赖与镜像源适配 |
| 初始场景原始回合 | `/data/by/genie_sim/exports/initial-scenes-20261126-one-per-task-scene-raw-trajectories`，6 条，回合数据约 15.52 GB | 由团队在实验室采集，每个任务的场景 0 和 1 各留一条 |
| 初始场景 LeRobot 训练数据 | `/data/by/genie_sim/exports/initial-scenes-20261126-one-per-task-scene-v2-1`，442 个文件，约 0.76 GB | 保留原动作、观测和录像，已通过 LeRobot v2.1 官方读回 |

只做模型侧读取的同学可先取 LeRobot。需要看 MCAP、实际控制命令、原始视频或物理验收证据的同学取 raw。已有服务器运行权限的成员可就地使用资产和镜像，节省数十 GB 传输。镜像归档、Docker 加载后的镜像和构建缓存可能同时占用磁盘，按实际可用空间选择下载项。

## 从实验室取团队版本

在**接收机**的 Linux 或 WSL 进入本仓库。`genie-lab` 代表成员自己配置的 SSH Host，替换为有权限的实际别名。需要 SSH、rsync 和 Python 3.11+。只取镜像时使用 SSH 和 Python 即可。服务器镜像通过只读 `docker image save` 流出，归档直接写入接收机。

```bash
LAB=genie-lab
DEST="$HOME/genie-sim-downloads"
bash download_from_lab.sh assets "$LAB" "$DEST"
bash download_from_lab.sh lerobot "$LAB" "$DEST"
bash download_from_lab.sh raw "$LAB" "$DEST"
bash download_from_lab.sh image "$LAB" "$DEST"
```

前三条分别完成可续传的目录复制和逐文件大小、SHA-256 核验。使用 Linux 文件系统（WSL 可用 home 目录）能保留 raw 包内部的硬链接。资产传输会跳过本机可重建的下载缓存。脚本从 Git 包内固定清单校验资产和 LeRobot，先核对 raw 的 `manifest.json` 摘要，再核对 882 个回合文件和 4 份验收回执。下载镜像中途断网可以重跑镜像命令，流式 tar 需要从头传。成功时才发布最终文件名。

省略作业 ID 时下载上表的历史样例。下载新作业时，从发布回执取得作业 ID。训练包用 `dataset-files.json` 校验，原始包用 `manifest.json` 校验。

```bash
JOB=lerobot-v30-verification-20261128
bash download_from_lab.sh lerobot "$LAB" "$DEST" "$JOB"
bash download_from_lab.sh raw "$LAB" "$DEST" "$JOB"
```

训练侧安装 [采集手册中的 LeRobot v3.0 读取环境](../PIPELINE.md#数据与读取)，使用整个目录读取。

```python
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

data = LeRobotDataset("local/geniesim_native", root=Path("/path/to/job-v3-0"), video_backend="pyav")
sample = data[0]
print(sample["task"], sample["observation.state"].shape, sample["action"].shape)
```

在**接收机**加载、验证团队镜像。上传到 GitHub 的协作包里只保存镜像 ID、补丁与步骤。

```bash
docker image load -i "$DEST/geniesim-by-collection-a100-20261119.tar"
docker image inspect geniesim-by-collection:a100-20261119 --format '{{.Id}}'
```

加载后期望 ID 为 `sha256:21b42560fa351db97478fa77851199fcfabb7af01ad3d35af1a8a80e05ac6732`。`verify_download.py image` 会检查 tar 结构、标签和配置摘要。Docker 加载成功并返回该 ID 后再用于采集。

如只需单个回合，先取索引得到任务、场景、帧数与实际目录，然后只下载该目录。单回合查看与完整包校验是两种操作。

```bash
RAW=/data/by/genie_sim/exports/initial-scenes-20261126-one-per-task-scene-raw-trajectories
mkdir -p "$DEST/one-episode"
scp "$LAB:$RAW/manifest.json" "$DEST/one-episode/manifest.json"
python3 -c 'import json,sys; print([(e["directory"],e["task"],e["scene"],e["frames"]) for e in json.load(open(sys.argv[1]))["trajectory_index"]])' "$DEST/one-episode/manifest.json"
rsync -aH --partial "$LAB:$RAW/episode_000002/" "$DEST/one-episode/episode_000002/"
```

`episode_000002` 对应手机装盒的初始场景 0。原始包、镜像、训练集和官方读取器的版本与校验值见 [锁定清单](../release/reference.json)。

每条留存轨迹的运行源码保存在对应原始回合中。扩展场景的留存条目与各自的 LeRobot、原始录制路径登记在实验室的 `/data/by/genie_sim/experiments/cross-scene-20261124/quota1-20261126/delivery.json`。

## 最新的物理验证样例

下表路径相对于实验室的 `/data/by/genie_sim/exports/`。每个训练集有 `official_loader_receipt.json`，原始包有包含 SHA-256 的 `manifest.json`。有服务器访问权限的成员可按需用 `rsync -aH --partial` 复制目录。

| 样例 | 回合与帧数 | LeRobot 目录 | 原始轨迹目录 |
| --- | ---: | --- | --- |
| 手机和 4B 橡皮入盒，原生 v3.0 读回验证 | 2 条，2,870 帧 | `lerobot-v30-verification-20261128-v3-0` | `lerobot-v30-verification-20261128-raw-trajectories` |
| 手机入盒，两种不同布局 | 2 条，2,852 帧 | `20261126-152211-skills-f805ead3f9e74334-v2-1` | `20261126-152211-skills-f805ead3f9e74334-raw-trajectories` |
| 手机入盒，显式复执行 | 1 条，1,426 帧 | `20261126-164344-skills-7b3f917abbac4a1c-v2-1` | `20261126-164344-skills-7b3f917abbac4a1c-raw-trajectories` |
| 目录外 4B 橡皮入盒 | 1 条，1,444 帧 | `20261127-104856-skills-d3a5d4cf1c7e4b16-v2-1` | `20261127-104856-skills-d3a5d4cf1c7e4b16-raw-trajectories` |

橡皮入盒的运行记录在 `/data/by/genie_sim/collection-runs/scene-8b9a75e9d89a185f78ff468d`，其中的 `independent-task-geometry.json` 记录盒内包含关系和稳定性，`plan-execution-audit.json` 记录模型计划的执行步骤。

## 从官方取得源码、资产和镜像

以下命令在自己的新目录执行。先固定与实验室相同的 Genie Sim 3.6.0 源码，供外部 SDK、benchmark 和暂未接通的模块使用。[官方固定提交](https://github.com/AgibotTech/genie_sim/tree/847bb9ebc002f7b0d2f21364970b3290c02e5f44)

```bash
git clone --no-checkout https://github.com/AgibotTech/genie_sim.git genie_sim-main
git -C genie_sim-main checkout --detach 847bb9ebc002f7b0d2f21364970b3290c02e5f44
git -C genie_sim-main rev-parse HEAD
```

**全量资产**用 ModelScope 的数据集模式和本次发布修订。在独立 Python 环境执行，`verify_download.py` 的路径指向团队仓库。Hugging Face 也提供官方资产。两个托管站使用各自的提交历史，应以固定清单确认所取字节。[官方资产库](https://modelscope.cn/datasets/agibot_world/GenieSimAssets) · [ModelScope 下载参数](https://github.com/modelscope/modelscope/blob/master/docs/source/command.md) · [Hugging Face 资产卡](https://huggingface.co/datasets/agibot-world/GenieSimAssets)

```bash
python3 -m pip install modelscope
modelscope download --dataset agibot_world/GenieSimAssets --revision f77c220cdf5d2e73fff77e1a2a0a9c7602384c44 --local_dir "$DEST/GenieSimAssets"
python3 verify_download.py assets "$DEST/GenieSimAssets" --restore-asset-receipt
```

**只下载某些 benchmark 依赖**可走上游 `geniesim assets`。固定源码附带的按任务索引使用另一份修订 `dfa5864361ec79188cfcc5795a8ea60f02f7b508`，用专门目录保存。subset 适合上游预览。复现本项目整库检索使用上一段全量版本。[官方按任务下载与断点续传](https://github.com/AgibotTech/genie_sim/blob/847bb9ebc002f7b0d2f21364970b3290c02e5f44/source/geniesim_cli/docs/assets-download.md)

```bash
python3 -m pip install -e genie_sim-main/source/geniesim_cli
geniesim assets plan finemanip
geniesim assets download finemanip --dest "$DEST/finemanip-subset" --workers 8
geniesim assets verify finemanip --dest "$DEST/finemanip-subset"
```

**官方采集环境**可直接拉取发布标签，或在固定源码目录用官方 CLI 单独构建。官方 `latest` 会变化。需要对照本批原始回合时使用实验室镜像 ID。实验室保留了自定义构建 Dockerfile、依赖记录及构建前后哈希，位置为 `/data/by/genie_sim/dockerhub-sidebuild-20261119/` 的 `source/data_collection/docker/Dockerfile`、`build-patches.json`、`completion-receipt.json` 和 `HANDOFF.md`。[官方采集镜像与构建说明](https://github.com/AgibotTech/genie_sim/blob/847bb9ebc002f7b0d2f21364970b3290c02e5f44/source/data_collection/README.md)

```bash
docker pull registry.agibot.com/genie-sim/geniesim4-data-collection:latest
# 另一条路线，在已安装上游 CLI 且已准备 6.2 父镜像的开发机运行
(cd genie_sim-main && geniesim data-collection build --china)
```

**官方公开示范数据**有自己的任务和采集版本。前述 raw 和 LeRobot 路径提供本项目留存的初始场景 G2 回合。下载公开集时可在官方数据集页面查看文件清单和说明。[官方示范数据](https://modelscope.cn/datasets/agibot_world/GenieSim3.0-Dataset)

## 实验室已经遇到的问题

| 现象 | 实测原因与处理 |
| --- | --- |
| 临时下载成功，却只有部分 USD 或缺少装配件 | 曾先取 task subset。整库恢复时固定 `f77c220…`，按 20,435 个发布文件核对字节数和 SHA-256。服务器目录多出的缓存、回执不计入发布文件数。检索程序读出 1,564 个已注册 USD 条目，计数范围与 5,000+ 物体和整库文件数各自不同。 |
| 网络中断或模型镜像跨源校验失败 | 对目录传输重跑相同命令。上游按任务 CLI 支持文件级 SHA 和源内固定修订，可在相同清单下换源。遇到已存在但校验失败的同大小文件，用 `rsync -aH --checksum --partial` 重取相应文件，再运行校验。 |
| 大镜像层下载中断、默认拉取不稳定 | 实验室当时按 manifest 校验分层下载，再导入项目 daemon。基础 manifest 为 `sha256:586f509cb924b02dc2fbdeacad0c0c8ce547e09c1e4625b4f14e3f429aac4ad5`。已有成员直接流出构建完成的镜像即可。构建证据见 `verified-image-origin.json`、`verified-local-import.log`。 |
| 镜像构建中 ROS 镜像源返回 404 | 镜像站缺少 `deb-src` 的源包索引。项目构建快照使用保留签名校验的二进制 ROS 源和有界超时。具体差异见 `build-patches.json`。 |
| Triton 的大 wheel 下到一半，或者依赖求解反复回退 | 实验室预取 Linux x86_64、CPython 3.12 的 Triton 3.6.0 wheel，核对 188,266,850 字节与 SHA-256 `74caf5e34b66d9f3a429af689c1c7128daba1d8208df60e81106b115c00d6fca` 后按本地文件安装。Viser 固定 `0.2.10`，与镜像内固定的 websockets 版本吻合。 |
| 镜像显示不存在或容器看不到 GPU | 实验室用项目专用 Docker socket 和 NVIDIA CDI，GPU UUID 与本机绑定。使用下方只读命令确认镜像和设备。其他机器要重新选本机 GPU、挂载路径并做相机与物理验收。 |
| 镜像相同，三路录制仍丢帧或几何校准失效 | 工作目录另有宿主机 SDK 挂载。已归档相机发布和 rosbag 提取两份补丁，以及 G2 TCP 校准、真实六份场景和积木插入几何。新工作区用下方脚本复核。已验证回合的版本在各自 `source/` 和 `evidence/` 内。 |

```bash
# 实验室只读核实镜像
DOCKER_HOST=unix:///data/by/genie_sim/dockerhub-sidebuild-20261119/docker.sock /data/by/genie_sim/recovery-tools/docker/docker image inspect geniesim-by-collection:a100-20261119 --format '{{.Id}}'
nvidia-smi -L
```

## 在独立工作区接回采集

以团队仓库为项目根，另取固定上游的 `source/data_collection` 作为专用的 SDK 工作副本，并保留 `genie_sim-main` 供 benchmark 几何代码读取。`SDK_ROOT` 指向这份可写副本。先验包，再按需填入小型输入并应用两处补丁。匹配的已有文件会复用，检测到内容分歧会报告具体路径。脚本仅在 `--install` 下写入目标。

```bash
python3 manage_runtime_inputs.py
python3 manage_runtime_inputs.py --project-root "$PWD" --sdk-root "$SDK_ROOT" --install
python3 manage_runtime_inputs.py --project-root "$PWD" --sdk-root "$SDK_ROOT"
```

现行启动器在实验室用 `/home/vipuser/by/genie_sim`、`/data/by/genie_sim/runtime-collection` 和项目 `.venv`。同机优先复用已有布局。新机需先提供对应挂载、Docker/CDI 设备、项目 Python 依赖和独立 LeRobot 读取器。在服务器为 Qwen 配置私有 `secrets/qwen.json`（字段 `base_url`、`api_key`）。资产索引可由具有项目读取权限的账号复用 `cache/text-embedding-v4/normalized-index.npz`。路径不同时请按目标机器重建索引，重建会调用计费的 Embedding 接口。详细运行与验收见 [采集手册](../PIPELINE.md)。
