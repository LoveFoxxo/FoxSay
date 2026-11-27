# FoxSay FSDP（权重分片）设计文档

## 解决什么问题

FoxSay 自带的 `MuonAdamW`（`FoxSay/optim.py`）是 **ZeRO-2**：梯度 reduce-scatter、
优化器状态分片，但**权重本身每卡复制**（fp32 矩阵 + bf16 embedding）。权重复制直接
卡死单卡能训的参数量上限。本文件夹在 ZeRO-2 之上补齐 **ZeRO-3 / FSDP 权重分片**，
使每卡持久显存变成 **O(参数量 / 世界大小)**，目标支撑 100 卡以内任意规模。

## 分片策略

| 参数类别 | 归属 | master 存储 | forward 取数 | optimizer.step |
|---|---|---|---|---|
| Muon（块内矩阵 c_q/c_k/c_v/c_proj/c_fc/c_proj/ve_gate） | 按"参数整只"分片（与优化器 reduce-scatter 分组完全同一套数学，见 `shard_math.py`） | owner 持完整 fp32/fp64 master | owner 广播 bf16 副本给所有 rank | owner 直接更新自己的 master；**删除原版 step 末尾的 all-gather 回写** |
| AdamW 大组（wte、value_embeds、lm_head——显存大头） | 整组摊平 + zero-pad + 均匀切块（任意 world size 都成立，32768 行词表切 100 卡也没问题） | 每卡只持 1/N 的 master chunk + 1/N 的 exp_avg/exp_avg_sq | 一次 all-gather 重建 flat buffer，各参数成为其 view | 梯度拷入 flat buffer → reduce-scatter(AVG) → 同一个融合 AdamW kernel 原地更新本地 chunk；**永远不 all-gather** |
| AdamW 小组（resid/x0/smear 标量，<4096 元素） | 复制 | 原地 fp32 | 不需要 | 走原版 all_reduce 路径（`MuonAdamW._reduce_adamw/_compute_adamw` 原样复用） |

## 机制（不改模型结构）

- 不包装模块、不改 state_dict 键名。每个"分片单元"（wte、value_embeds[i]、block[i]、
  lm_head）注册一个 **forward pre-hook**（已 `torch.compiler.disable`，dynamo 在此
  graph-break、hook 内 eager 执行）：把本单元参数 `.data` 指向新分配的 compute-dtype
  buffer 的 view，并**预取下一个单元**的通信。
- 每个参数注册 **post-accumulate-grad hook**：整个单元 backward 完成后立即释放
  gather buffer、恢复空/master 存储。梯度张量本身保留到 `model.zero_grad`。
- master 存放：Muon owner 的 master 就是 param 的"home storage"（fp32 时零拷贝）；
  AdamW flat 组的 master chunk 是独立张量，param `.data` 只在前向窗口内是 view。

## 通信量 / 显存账（每卡，N=参数量，W=世界大小，fp64 master 为例）

- 持久：Muon master+动量 ≈ 18B·N/W，AdamW master+两个动量 ≈ 24B·N/W → 约 **42B·N/W**
- 瞬时峰值：bf16 gather buffer（前向末端全模型 bf16 ≈ 2B·N）+ bf16 全尺寸梯度 ≈ 2B·N
- 对比原版：fp32 权重 4B·N 恒驻 + fp32 梯度 4B·N。W=8 时约省一半，**W=64/100 时
  权重部分省 30~100 倍**，10B 模型在 8×80GB 上从"训不动"变成宽裕。
- fp64 master 的额外代价只有 4B·N/W（分片摊掉了），这正是"魔改精度到 64"的正确姿势：
  bf16 计算 + fp64 master，速度≈bf16，精度预算拿到 fp64。

## 用法

```bash
# 8 卡，fp32 master（默认）
torchrun --nproc_per_node=8 -m scripts.base_train --fsdp ...

# 64 卡，fp64 master（"魔改精度到64"：bf16 计算 + fp64 master/动量）
torchrun --nproc_per_node=64 -m scripts.base_train --fsdp --fsdp-master-dtype float64 ...
```

`scripts/base_train.py` 已接线（`--fsdp` / `--fsdp-master-dtype`）。checkpoint：
- 模型权重：rank0 照旧写 `model_<step>.pt`（`consolidated_state_dict()` 重建标准
  全量 state dict，原始 dtype 还原），**任何 FoxSay 脚本都能加载**；
- 优化器状态：照旧按 rank 写 `optim_<step>_rank{r}.pt`，但内容是 FSDP chunk 布局，
  **恢复训练必须用相同卡数**（与原版 rank 分片约定一致，加载时强校验并报错）。

## 测试

```bash
python tests/test_fsdp.py          # CPU + gloo，多卡 vs 单卡参考逐权重对拍
# 或 pytest tests/test_fsdp.py -v
```

验证矩阵（Linux x86 + gloo 实测通过；"跨 rank 逐位一致" = 训练若干步后所有 rank 的
master 权重 bit-exact 相等，这是"万卡不跑偏"的直接证据——rank 间只要逐位一致，
卡数增加就只是 SUM 里多几项，不会产生新的不一致源）：

| 场景 | 口径 | 实测结果 |
|---|---|---|
| ws=1（无进程组退化）vs 原版单进程 | 逐元素 | **max\|dw\| = 0，逐位一致** |
| ws=2 fp32 vs 原版 + 跨 rank | 逐元素 | **max\|dw\| = 0；2 rank 逐位一致** |
| ws=2 fp64 vs ws=1 fp64（同精度轨迹） | 逐元素 | **max\|dw\| = 0；2 rank 逐位一致** |
| ws=4 fp32 vs 原版（含 rank 零 ownership） | 逐元素 | **max\|dw\| = 0**（(4g)/4 逐位精确） |
| ws=3 fp32（奇数卡，原版 ZeRO-2 跑不了） | 1 步紧对拍 + 3 步混沌界 | 1 步 2.4e-7；3 步 6.7e-4（<5e-3，全 finite） |
| **ws=3 fp64（SUM+div 精确均归约）** | 跨 rank 逐位 + 漂移不放大 | **3 rank 逐位一致；1 步 0.0 → 3 步 3.3e-6 不放大**（fp32 路径同条件放大 ~2800 倍到 6.7e-4） |
| ws=2 fp32/fp64 + bf16 梯度（真实 GPU 链） | vs ws=1 同梯度模式 | **max\|dw\| = 0；全 rank 逐位一致**（bf16 噪声一次性进入、不随卡数放大） |
| 超参 0-D 张量 dtype | fp64 0-D CPU | 10 个张量全部 fp64 |
| **ws=3 fp64 回环激活（stage 9）**：loop_gate.readout/adapter 各自成 unit；gate 非中性参数下每前向 3 个 latent cycle | 跨 rank 逐位 + 漂移相对界（回环放大梯度，绝对界失效） | **3 rank 逐位一致（49 张量）；drift 2.0e-5 ≤ 回环位移 4.888 的 5%；loss 零漂移** |

回环激活下的两条新不变量（stage 9 修复记录）：
- gate 的 `readout`/`adapter` 是以**子模块调用**进入前向的，必须各自建 FSDP unit；
  挂在 `loop_gate` 自身上的 hook 永远不触发，非 owner rank 会拿空权重前向 →
  rank 间崩溃分歧 → all-gather 死锁（表象即"单卡过、多卡挂"）。
- `_free_unit` 会把被 one-ahead prefetch 提前 relaunched 的 unit（回环里 lm_head 之后
  的 readout/adapter 每步都被反复进出）的 `launched` 标志与残留 handles 一并复位，
  否则下一次激活跳过 launch、把参数绑到已置空的 `cur_buf` 上。

**数值口径说明（重要）**：
- 换 master dtype = 换轨迹。fp64 master 相对 fp32 参考的 ~6e-4 轨迹差是 AdamW
  `m/√v` 对低位差异的混沌放大，不是分片 bug；同 dtype 跨 world size 对拍才是
  分片正确性判据。
- fp32 下非 2 的幂卡数的 `(Σg)/N` 每 step 有 1~2 ulp 舍入，多步会被 `m/√v` 放大
  （ws=3 实测 2.4e-7 → 6.7e-4，~2800 倍）—— 这是 fp32 本身的容量问题，**在 fp64
  reduce 下不再存在**（SUM+div 每 rank 同一舍入，跨 rank 逐位一致，见下）。
- 判定真分片 bug 的特征：≥1e-2 量级偏差或 NaN。
- fp64 与 ws=1 的 3.3e-6 残差来自 fp32 compute-cast 的 ulp 翻转（bf16/fp32 计算
  的固有代价），它在 fp64 优化器状态里**不放大**；NANOCHAT_DTYPE=float64 可把
  计算链也升到 fp64，连这个残差也归零。

## 精度根因审计（"卡越多精度越差"的 7 处根因与修复）

逐处根因、所在文件、修法（f64 万卡标准 = 全 rank 逐位一致 + 漂移不放大）：

| # | 根因 | 位置 | 修复 |
|---|---|---|---|
| 1 | `ReduceOp.AVG` 内部按非精确缩放分块求和，非 2 幂卡数/万卡下各 rank 舍入不同 | `fsdp/core.py` `_reduce_scatter_avg`、`optim.py` 小参组 all_reduce | **SUM + 整数 N 一次除法**：每 rank 同一归约树、同一单次舍入 → 跨 rank 逐位一致（`_ScaledMean` 包装异步 wait） |
| 2 | 梯度累加/归约缓冲 = compute dtype（GPU 上 bf16），每步丢失低位 | `fsdp/core.py` `AdamWFlatGroup.grad_flat/grad_chunk` | 缓冲改为 **reduce_dtype**（fp32 起步；fp64 master 时 fp64），bf16 梯度在求和前上抬 |
| 3 | Muon 堆叠梯度按 grad dtype 建 buffer（GPU 上 bf16 直接跨卡平均） | `fsdp/core.py` `_reduce_muon` | 堆叠/归约 buffer 按 **reduce_dtype** 重建，copy_ 逐参上抬 |
| 4 | 复制小组用 stock `ReduceOp.AVG` all_reduce | `fsdp/core.py` `FSDPMuonAdamW._reduce_adamw` | override：**SUM + `_ScaledMean`**，ws=1 走原版 |
| 5 | 超参（lr/beta/eps/wd/step）0-D 张量 fp32，fp64 下每步截断高位 | `optim.py` `MuonAdamW.__init__` | 10 个张量全部 **fp64 0-D CPU**（compile 不重编），核内 round 到累加 dtype（fp32 值语义不变） |
| 6 | Muon 核硬编码 `COMPUTE_DTYPE==bf16 时把梯度砸回 bf16` + `g.float()`/`step_size.float()` 降采样，fp64 master 下 NS 迭代全程 bf16/fp32 | `optim.py` `muon_step_fused` | **按 master dtype 门控**：仅 fp32 masters 走 bf16 快路（原行为不变）；fp64 时 NS 迭代/方差归约全程 fp64，`gf` 模式拒绝下cast |
| 7 | 微批次梯度累积在 bf16 `p.grad` 上累加（base_train 每 step grad_accum 次） | `scripts/base_train.py` + `FoxSay/common.py` | `NANOCHAT_DTYPE=float64` 新档：计算/累积/归约/master **全链 fp64**（CPU 友好，GPU 需 full-fp64 卡）；默认 bf16 计算时靠 #2/#3 的上抬消掉跨卡部分 |

fp32 master 下这些修复全部零语义变化（SUM+div 与 AVG 在 2 幂卡数逐位相同，
ws=1/2/4 对原版仍逐位一致）；fp64 master 下它们共同构成"万卡不跑偏"的证据链。

## 已知限制 / 后续可做

1. 梯度在每步 backward 期间仍是全尺寸（bf16 约 2B/参数）；可再做单元级梯度
   reduce-scatter + 提前释放（FSDP1 的 backward prefetch 方向）。
2. 每 unit 每次 micro-batch 一次 all-gather + k 次广播；可按形状跨层堆叠成单次
   大广播（同形状 c_q 全层一次发完）进一步降低小包开销。
3. torch.compile 下每个 unit 有一次 graph break（hook 被 dynamo 跳过），正确性
   不受影响，融合度略降。
4. 不支持 fp16（GradScaler）与 fp8 组合。
5. 优化器 checkpoint 与卡数绑定（与原版约定一致）。
