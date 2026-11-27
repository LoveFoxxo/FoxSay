# FoxSay/gan — 内部验证回环 + 对抗输出门（GAN）包

## 为什么存在

原架构"想更久"的唯一途径是多生成 token（test-time），且 RL 在 GSM8K 上
只给二值正确奖励、rollout 256 token 截断即 0 奖励——训练结果必然是
"少想快答"。本包提供两条互补的修复路径：

1. **内部验证回环（latent recurrence）**：Transformer 主干在一个位置上
   循环执行多个 cycle，"想"发生在隐空间，不产生额外 token。
2. **对抗输出门 + 过程奖励**：GAN 判别器在推理时充当产出门槛，在 RL
   时充当过程奖励来源，训练与推理用同一判据。

## 组件

| 文件 | 组件 | 职责 |
|---|---|---|
| gate.py | `LoopGate` | ACT 停机门（p_stop/混合权重）+ cycle 适配器 + mix_lambda。**全部零初始化** → 未训练时前向与旧模型逐位一致 |
| gate.py | `neutral_state()` | 中性张量的唯一权威来源，`GPT.init_weights` 和 `_patch_missing_keys` 都引用它 |
| verifier.py | `VerifierModel` | 判别器/PRM：读整个 turn，输出逐块质量分 + 整 turn pass logit |
| verifier.py | `VerifierLoss` | LSGAN 目标（正样本→+1，负样本→-1，排名一致性项） |
| verifier.py | `OutputGate` | **最终输出门槛**：采样到 `<|assistant_end|>` 时判分，P(pass)≥阈值放行；被拒→遮蔽结束符继续想+回环预算+1；`max_rejects` 次后强制放行（无死锁） |
| verifier.py | `VerifierTrainer` | 在线对抗训练器（AdamW + LSGAN），随 policy 滚动更新 |
| reward.py | `RewardConfig` / `shape_reward` | 组合奖励：结果分 + PRM 过程分 + 对抗分 + 循环效率塑形 + 截断小惩罚（**不清零**） |
| reward.py | `grpo_advantages` / `token_level_advantages` | GRPO 组相对优势（r−μ，无 z-score）+ DAPO token 级归一 |

## 接线位置（"在需要的地方引用、插入"）

- `FoxSay/gpt.py`：`GPTConfig(loop_max_cycles, loop_stop_threshold)`；
  `GPT.loop_gate`；forward 的 `run_trunk/read_kv/kv_capture` 回环段；
  `setup_optimizer` 的 loop 参数组（Muon×矩阵 + AdamW×标量）。
- `FoxSay/engine.py`：`Engine(model, tokenizer, output_gate=...)`；
  采样 `<|assistant_end|>` 时的门控分支；`row_cycles`（供 RL 效率项）。
- `scripts/chat_rl.py`：`--loop-cycles`、`--use-output-gate`、奖励权重 CLI；
  rollout 时在线训练判别器；`shape_reward`+`grpo_advantages` 替代旧均值法。
- `scripts/chat_cli.py`：`--verifier-path` 加载训练好的判别器开启门槛。
- `FoxSay/checkpoint_manager.py`：`_patch_missing_keys` 给旧 checkpoint
  嫁接中性 loop_gate（旧权重无损加载、行为零变化）。

## 关键不变量（全部已测试）

1. `loop_max_cycles=1` 或未训练 gate：前向与旧模型**逐位一致**（含 KV
   cache prefill/decode 路径）。
2. 精化 cycle 不推进 KV cache 位置；激活门后每个 token 仍恰好 +1。
3. 未训练 verifier（p=0.5≥阈值 0.5）时输出门等价于无门；`max_rejects`
   上限保证任何 verifier 都不会造成死锁。
4. mix_lambda 是唯一"点火开关"：中性初始化下第一步只有它拿到梯度，
   回环由数据信号唤醒（鲁棒性设计）。
5. 回环深度受 `min(_loop_depth, max_cycles)` 硬约束。

## 与 FSDP/fp64 的兼容

loop 参数都是普通 nn.Parameter，走既有 MuonAdamW 参数组；FSDP 分片、
fp64 master 对它们一视同仁。判别器在 RL 进程内独立训练（AdamW），
不进 FSDP 树。

## 测试

`tests/test_gan.py` 覆盖：包冒烟、零初始化等价、KV-cache 一致性、
点火梯度、优化器覆盖、门控行为（未训练/全拒/全收）、奖励塑形排序、
GRPO 优势、对抗分离收敛、checkpoint 嫁接。
