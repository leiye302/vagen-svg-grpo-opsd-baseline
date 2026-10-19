# VAGEN-SVG: GRPO + gated OPSD baseline (50 updates)

在 VAGEN 的 **目标图片 → reasoning → 完整 SVG → 渲染反馈** 环境中，用
Qwen2.5-VL-3B-Instruct 直接进行 GRPO + gated OPSD。默认跑 **50 次 trainer update**。

本仓库只发布 baseline。**不需要 Route-SFT checkpoint、私人轨迹、标注库、
Analyzer、API key、任何作者电脑或服务器上的文件。** 所需代码、模型和图像数据
均从固定版本的公开来源取得。它不是 VAGEN-Full PPO/bi-level-GAE 的原样复现；
是 VAGEN 多模态环境与 rollout + SDAR 公开损失的适配版本。

## 1. 一眼看清训练配置

| 项目 | 默认值 |
| --- | --- |
| 起点 | `Qwen/Qwen2.5-VL-3B-Instruct`，未做额外 SFT |
| 数据 | `starvector/svg-icons-simple`；train 500，固定 dev 64 |
| 总步数 | 50 trainer/rollout updates |
| 每步 | 16 个任务槽 × 8 条轨迹 = 128 条轨迹 |
| 单轨迹 | 最多 2 次 SVG 提交；每轮最多 750 个生成 token |
| PPO | minibatch 32，1 epoch，microbatch 1/GPU |
| 学习率 | `1e-6` |
| OPSD | `gate_beta=5.0`，`sdar_coef=0.01`，token-mean |
| Reference KL | `low_var_kl`，系数 `0.01`；不同于 OPSD Teacher |
| Entropy | 系数 `0.001` |
| 验证 | 0、10、20、30、40、50 步，同一 dev64，greedy |
| 保存 | 20、40、50 步；只保留最近两个完整可续训断点 |
| 资源配置 | 单 GPU，TP=1，参数和 optimizer CPU offload |

50 次 update 通常对应 **200 次 optimizer.step**，共采样 6,400 条训练轨迹，
而不是 50 个样本或 50 次单独梯度更新。验证轨迹不参与梯度更新。

### OPSD Teacher 是谁？

Teacher 是 **当前 Actor 权重的无梯度评分视角**。Teacher 的 prompt 比 Student
多一段公开、固定的 SVG 绘制 skill；对 Student 已生成的相同 token 计算概率。
不另外生成一个 Teacher 答案，不使用冻结 Teacher，也不给 Teacher 目标 SVG 源码。

Student 在 rollout 和验证时只看原生图片、历史、渲染反馈，**看不到 skill**。
完整公式和梯度路径见 [方法与实现](docs/METHOD.md)。

## 2. 机器要求

- Linux x86-64，Python **3.12**，Git，网络可访问 GitHub/Hugging Face/PyTorch。
- 默认面向单张 **80–96 GB NVIDIA GPU**；优先 RTX PRO 6000 96GB，或 A100/H100 80GB。
  具体峰值仍需在实际机器上运行 smoke 确认，不能只凭卡名保证能装下。
- CUDA 12.8 兼容驱动。若从源码构建 CUDA 扩展，还需要 CUDA 12.8 toolkit。
- 建议可用 CPU 内存 **至少 96 GiB**，最好 128 GiB；CPU offload 会使用主存。
- 建议准备 **180 GiB 可用磁盘**，用于依赖、模型、数据及保存时的临时峰值。
  小磁盘请先运行 preflight，不要删 optimizer 来伪装“完整断点”。
- 系统依赖（Debian/Ubuntu）：

```bash
sudo apt-get update
sudo apt-get install -y git python3.12 python3.12-venv build-essential \
  libcairo2 libglib2.0-0 libgl1 libgomp1 libffi-dev pkg-config
```

默认使用 CUDA 12.8 / Torch 2.7.0 / vLLM 0.9.2 / Transformers 4.51.3，
显式启用 **vLLM V0**。旧 VAGEN 的 FSDP 权重同步不能直接套用 vLLM 0.11 的 V1。
不要在原有训练环境中随意降级；安装脚本建立独立 `.venv`。

## 3. 从零准备：不需要作者的任何本地文件

克隆本仓库，进入仓库根目录，然后执行：

```bash
git clone https://github.com/leiye302/vagen-svg-grpo-opsd-baseline.git
cd vagen-svg-grpo-opsd-baseline
bash scripts/install.sh
source .venv/bin/activate

# 工作目录可以位于任意有足够空间的数据盘；不要求特定用户名或路径。
export SVG_WORK_DIR="$PWD/work"

python scripts/prepare_assets.py --work-dir "$SVG_WORK_DIR"
python scripts/prepare_data.py --work-dir "$SVG_WORK_DIR"
```

安装脚本会从 `sources.lock.json` 指定的公开 commit 获取 VAGEN/VERL，并验证
原文件与 overlay 的 SHA-256；不会读取本地其他项目。资产脚本下载官方模型、
公开 SVG 数据、DINO 和 DreamSim 权重。首次下载需要网络，之后训练不需要付费 API。

数据脚本从公开数据独立生成 500/64 个任务和 SHA manifest。它不会读取私人轨迹、
旧标注或已训练模型。数据选择、渲染和固定任务验证见 [数据与复现](docs/DATA.md)。
**整个 work 目录搬到另一台机器后，重新运行 prepare_data 以重建该机器的绝对路径**；
不要直接使用包含旧机器路径的 Parquet 文件。

## 4. 检查、smoke，再运行 50 步

先执行不占用训练 GPU 的 CPU 检查：

```bash
python scripts/verify_loss.py
python -m pytest tests -q
python scripts/preflight.py --work-dir "$SVG_WORK_DIR"
```

`verify_loss.py` 将三个损失函数的可执行 AST 与固定公开 SDAR 版本比较。
CPU 测试覆盖 mask、Teacher detach、真实 Actor update hook、任务分组、固定验证集等；
**CPU 通过不等于 GPU 端到端训练通过**。

先跑一个独立 smoke：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run.py --work-dir "$SVG_WORK_DIR" --smoke
```

smoke 使用 1 个任务 × 8 条轨迹、1 次更新，保留两轮交互和 750-token 生成预算，
不先跑整个 dev64。它检查 rollout → SVG 渲染评分 → 当前 Actor Teacher 评分 →
backward → 完整保存。**smoke 的模型不作为正式 50 步起点**。

正式从基模训练：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run.py --work-dir "$SVG_WORK_DIR"
```

默认脚本会创建独立后台会话；关闭 SSH 不会主动终止训练。返回 PID 只代表已提交，
**不是已经开始训练或训练成功**。查看 `work/runs/baseline50/status.json` 和该目录下的
`logs/`；smoke 对应 `work/runs/smoke1/`。需要由 `tmux` 或作业调度器托管时，可加
`--foreground`。不要在同一工作目录重复启动训练实例。脚本自动管理自己的本地 SVG
服务，不需要填写外部服务器地址，也不会停止别人的 GPU/Ray 进程。

正式训练前必须确认 smoke 的 `status.json` 同时包含 `phase="complete"` 和
`training_complete=true`。本脚本不会未经确认自动重启失败任务。

中断后的恢复：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run.py --work-dir "$SVG_WORK_DIR" --resume
```

恢复只接受本轮完整断点；保留模型、optimizer、scheduler、RNG 和 dataloader 状态。
不是拿某个权重目录“重新从第 0 步训”。没有完整断点时不要自动假装恢复成功。

## 5. 看哪些输出？

默认输出在 `work/runs/baseline50/`，以运行时实际 resolved config 为准：

- 控制台/训练日志：reward、entropy、KL、GRPO、SDAR gate/gap/loss、耗时。
- `validation/dev64_step_*.json`：固定 64 个 task_id、每任务输出及各项指标。
- `global_step_*/`：完整可续训 checkpoint，以及完成/完整性记录。
- 数据和运行 manifest：来源版本、文件 SHA、解析后的配置、运行环境。

SVG 主要看 **DreamSim 相似度、DINO 相似度、综合 reward 和协议有效率**。
这里不是 ALFWorld，没有把 reward 擅自换成“任务成功率”。
比较不同方法时，使用同一份 task manifest、同一 dev64、相同 update 位置。
官方 test split 不参与梯度训练和周期验证；仅允许在数据准备时用于去重保护。

50 步结果整理：

```bash
python scripts/summarize.py --run-dir "$SVG_WORK_DIR/runs/baseline50"
```

## 6. 文件导航

| 路径 | 内容 |
| --- | --- |
| `configs/` | 完整公共配置、50 步覆盖项、Teacher 静态 SVG skill |
| `sources.lock.json` | 上游代码、官方模型、公开数据的固定版本 |
| `overlays/` | VAGEN/VERL 修改文件及 preimage/replacement SHA |
| `svg_opsd/` | Teacher 打包、固定任务、协议、安全和验证证据 |
| `scripts/` | 安装、公开资产获取、数据构造、预检、启动、恢复和汇总 |
| `tests/` | CPU 合同与梯度测试；另有真实 processor 打包检查 |
| `docs/` | 方法、数据协议、运行和故障排查 |

## 7. 验证范围和公开内容

当前发布包含可审计的接线和 CPU 测试，**不随代码声称已经复现某个 50 步成绩**。
具体已完成和待完成的检查见 [验证记录](docs/VALIDATION.md)。
GPU 内核、FSDP/vLLM 生命周期与显存占用必须由运行方先执行 smoke 验证。
smoke 未通过时应保留日志排错，不能降低 batch、关闭 OPSD 或更换数据后仍声称同配置。

仓库不含作者机器路径、服务器账号/地址、密钥、历史训练日志、标注库、私人图片或模型权重。
运行方新生成的日志与 checkpoint 默认被 `.gitignore` 排除；请勿将凭据写入配置或提交。

归属和许可证见 [NOTICE](NOTICE.md)、[LICENSE](LICENSE)、`licenses/`。
