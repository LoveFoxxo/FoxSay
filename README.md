# FoxSay

FoxSay 是一个可阅读、可修改、可端到端实验的双脑语言模型项目。它以 Andrej Karpathy 的 [nanochat](https://github.com/karpathy/nanochat) 为基础，保留分词、预训练、SFT、RL、评测和推理工具链，并加入时序右脑、胼胝体融合、稀疏 MoE、数值精度控制与权重分片。

> **边界说明**：双脑结构、HiPPO 和更高精度可以改善表达能力或数值稳定性，但不等价于生物人脑，也不保证任意任务上的误差趋零。本文中的“收敛”主要指浮点误差、梯度稳定性和优化过程；验证集误差仍取决于数据、优化、训练量和任务本身。

## 主要组成

| 模块 | 文件 | 作用 |
|---|---|---|
| 左脑 | `FoxSay/gpt.py` | Transformer、GQA、RoPE、值残差、可选内部验证回环与稀疏专家 |
| 右脑 | `FoxSay/right_brain.py` | HiPPO-LegS 状态空间通道、Tustin 离散化、因果深度卷积、门控残差及流式状态 |
| 胼胝体 | `FoxSay/corpus_callosum.py` | 在推理 Engine 中把右脑时序表示转换成 logits 偏移，与左脑输出相加 |
| 稀疏扩容 | `FoxSay/moe.py` | Top-K 路由、无 token 丢弃、负载均衡辅助损失、router z-loss、完整梯度集合 |
| 数值底座 | `FoxSay/numerics.py` | RMSNorm 在低精度输入下使用 fp32 累加，也支持 fp64 |
| 优化与保护 | `FoxSay/optim.py` | Muon + AdamW、全局梯度裁剪、NaN/Inf 检查及分布式同步跳步 |
| 权重分片 | `FoxSay/fsdp/` | ZeRO-2 之上的 FSDP 权重分片、主权重与优化器状态管理 |
| 可选能力 | `FoxSay/gan/`、`FoxSay/mind/` | 内部验证器、循环门控、谱方法、记忆及其他实验模块 |
| 输入与输出 | `FoxSay/inputs/`、`FoxSay/outputs/` | 文本、网页、多模态学习入口与 CLI/Web/媒体接口 |
| 规模规划 | `FoxSay/scaling.py` | 在 meta 设备上统计双脑参数，不实际分配万亿参数显存 |

### 双脑怎样协同

1. 左脑产生最终隐藏状态和语言模型 logits。
2. 右脑读取左脑隐藏状态的历史窗口，通过 HiPPO 与因果卷积提取时序特征。
3. `CorpusCallosum` 将右脑表示投影为 logits 偏移，再加到左脑输出。
4. 胼胝体的两个投影采用零初始化，因此未经训练的桥接层不会改变原有左脑输出。

**当前 `scripts/base_train.py` 仍训练左脑 GPT，并没有自动联合训练左右脑。** 右脑提供独立的 `setup_optimizer()`、辅助损失和 FSDP 接口；Engine 中的胼胝体是推理融合路径。正式双脑训练还需要显式构造任务、组合损失与保存相应模块。

## 环境安装

要求 Python 3.10 或更新版本，依赖由 [uv](https://docs.astral.sh/uv/) 管理。

```bash
git clone https://github.com/LoveFoxxo/FoxSay.git
cd FoxSay

# CUDA（PyTorch CUDA 12.8 源）
uv sync --extra gpu --group dev

# 或 CPU / MPS；Windows 可使用 .venv/Scripts/activate
uv sync --extra cpu --group dev
source .venv/bin/activate
```

`cpu` 与 `gpu` 是互斥选项。开发组提供 pytest 等测试依赖。需要严格使用锁文件时：

```bash
uv lock --check
uv sync --frozen --extra cpu --group dev
```

## 从数据到聊天

### 1. 下载数据并训练分词器

```bash
# 下载 170 个 ClimbMix 数据分片；可按磁盘和训练规模减少
python -m FoxSay.dataset -n 170

# 训练 32768 词表的 BPE 分词器并计算 token_bytes
python -m scripts.tok_train
python -m scripts.tok_eval
```

下载和训练会消耗大量网络、磁盘与内存；请先用较小 `-n`、`--max-chars` 验证流程。完整流水线也可参考 `runs/speedrun.sh`，该脚本面向多卡 GPU，不应直接用于资源有限的电脑。

### 2. 左脑预训练

小规模 CPU 冒烟实验：

```bash
python -m scripts.base_train --depth=4 --max-seq-len=512 \
  --device-batch-size=1 --total-batch-size=512 --num-iterations=20 \
  --eval-tokens=512 --core-metric-every=-1 --sample-every=-1
```

8 卡训练及 FSDP 示例：

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --depth=24 --device-batch-size=16 --run=d24

torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --depth=24 --device-batch-size=16 --fsdp --fsdp-master-dtype=float64
```

当单卡放不下时，优先减小 `--device-batch-size`、序列长度或模型深度。FSDP 的优化器检查点依赖恢复时的卡数；更多限制见 `FoxSay/fsdp/README.md`。

### 3. SFT、评测与推理

```bash
python -m scripts.chat_sft
python -m scripts.chat_eval -i sft
python -m scripts.chat_cli -p "你好，FoxSay！"
python -m scripts.infer_bench
```

多卡 SFT/RL 的启动方式见各脚本顶部说明。`--run=dummy` 可关闭 wandb 记录；非 dummy 运行前需自行配置 wandb。

## 低误差与鲁棒训练开关

```bash
python -m scripts.base_train \
  --moe-experts=8 --moe-top-k=1 --moe-expert-ratio=4 \
  --loss-dtype=float64 --loss-chunk-size=128 --logit-softcap=0 \
  --grad-clip=1.0
```

- `--moe-experts=0`：保留原始稠密结构；大于零时在指定层启用稀疏专家。
- `--moe-top-k`：每个 token 激活的专家数；当前实现不丢弃 token。
- `--loss-dtype=float64`：提高 logits 与交叉熵的精度，会增加计算和内存开销。
- `--loss-chunk-size`：分块计算语言模型头与损失，缓解完整 logits 的瞬时内存。
- `--logit-softcap=0`：关闭 tanh logits 饱和；关闭后需依靠梯度保护与训练监控控制异常值。
- `--grad-clip`：全局梯度范数裁剪；检测到非有限梯度时同步跳过更新。

还可通过 `NANOCHAT_DTYPE=float64` 提高整条计算链精度。它不是免费升级：GPU 吞吐、显存和硬件 fp64 能力都可能成为瓶颈，且更高浮点精度不能消除数据噪声、局部最优或泛化误差。

## 近万亿双脑预设

```python
from FoxSay.scaling import trillion_dual_configs, dual_parameter_report

left_config, right_config = trillion_dual_configs()
print(dual_parameter_report(left_config, right_config))
```

该预设使用 64 层、8192 维宽度，两侧均配置 14 个专家、Top-1 激活；元设备统计结果为：

| 口径 | 左脑 | 右脑 | 合计 |
|---|---:|---:|---:|
| 总参数 | 492,318,002,586 | 485,409,423,360 | 977,727,425,946 |
| 单 token 激活参数上限 | 45,641,403,802 | 38,732,824,576 | 84,374,228,378 |

“激活参数上限”排除了未被 Top-1 路由选中的专家参数，不代表实际显存占用。此预设是**架构与分片规划，不是单机可训练承诺**：真正落地仍需专家并行、流式初始化、通信与检查点设计，以及 Linux/NCCL 集群上的系统验证。当前 FSDP 会分片存储，但专家前向仍有 gather/广播成本。

直接构造左、右脑的接口：

```python
from FoxSay.gpt import GPT, GPTConfig
from FoxSay.right_brain import RightBrain, mirror_left

left_config = GPTConfig(n_layer=4, n_embd=256, n_head=4, n_kv_head=4)
right_config = mirror_left(left_config, ssm_state=16, conv_kernel=7)
# 在目标训练脚本中显式初始化、组合损失并保存两个模型。
```

## 包改名与兼容性

核心 Python 包已从 `nanochat` 改为 **`FoxSay`**，脚本、测试、notebook 和 Shell 中的导入路径已同步更新：

```python
from FoxSay.gpt import GPT, GPTConfig
from FoxSay.engine import Engine
```

旧 `import nanochat` / `from nanochat ...` 不再作为包名提供，外部代码需改为 `FoxSay`。包名区分大小写，Linux 上尤其如此。

为避免已有数据与 checkpoint 失效，以下接口暂时保留旧名称：

- `NANOCHAT_DTYPE`、`NANOCHAT_BASE_DIR` 环境变量；
- 默认缓存目录 `~/.cache/nanochat`；
- 既有 checkpoint 中的模块字段结构（本次是包目录改名，不是模型参数重命名）。

历史文档和指向上游 Karpathy 仓库的 URL 保留 `nanochat`，它们不是失效导入。

## 测试与已知限制

```bash
python -m compileall -q FoxSay scripts tests tasks
python -m pytest -q

# 未安装 MSVC 的 Windows 环境可禁用 Dynamo 后运行非沙箱、非多进程集合
TORCHDYNAMO_DISABLE=1 python -m pytest -q \
  --ignore=tests/test_execution.py -k "not test_fsdp_multi_rank"
```

Windows 上未配置进程沙箱或多进程 Gloo 时，相关测试可能失败，应修复运行环境，而不是跳过失败后宣称通过。GPU、FSDP 多 rank、真实数据训练及万亿预设需要对应硬件与后端；小规模单元测试通过不代表这些场景已经完成验收。

提交前建议运行 `git diff --check`。虚拟环境、缓存、本地 dry-run 和构建产物由 `.gitignore` 排除。

## 目录速览

```text
FoxSay/                 核心包（双脑、MoE、优化器、分片、I/O、实验模块）
scripts/                数据、训练、评测与推理命令
runs/                   可配置的 Shell 实验流水线
tasks/                  CORE/SFT/RL 任务
tests/                  单元、数值、分片与接口测试
dev/                    上游历史、实验记录和分析 notebook
pyproject.toml          项目、依赖和 setuptools 包发现配置
uv.lock                 锁定依赖
```

## 来源与许可

FoxSay 基于 [karpathy/nanochat](https://github.com/karpathy/nanochat) 改造，感谢原作者及相关开源项目。项目继续采用 MIT 许可，完整条款与上游版权声明见 `LICENSE`。`dev/LOG.md`、`dev/LEADERBOARD.md` 保留上游实验历史，不应视为 FoxSay 当前硬件上的复现结果。
