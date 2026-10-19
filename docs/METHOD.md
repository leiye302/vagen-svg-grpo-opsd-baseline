# 方法与代码对应

## 任务与闭环

策略观察目标图像及已有渲染反馈，先输出显式 reasoning，再提交完整 SVG：

```text
<think>...</think><answer><svg ...>...</svg></answer>
```

SVG 在整段生成结束后交给环境解析、渲染、评分。下一轮看到渲染图及反馈，再输出
一份完整 SVG，而不是 SVG patch。默认最多两轮。非法协议、外部资源或脚本不执行。
目标图是普通观察；目标 SVG 源码仅供环境生成目标图，不进入 Student/Teacher prompt。

## GRPO

同一个任务采样八条轨迹，用 VAGEN 的轨迹回报和 group-relative advantage；
环境观察 token 不参与 Actor 损失。使用公开 SDAR clipped PG，包括 dual clipping。
每个 update 128 条轨迹，minibatch32，PPO epoch1，共四次 optimizer.step。

## Gated OPSD

设 Student 的已采样 token 为 $y_t$，普通历史为 $h_t$，固定 SVG skill 为 $s$。

$$
\ell_t^S=\log\pi_\theta(y_t\mid h_t),\qquad
\ell_t^T=\log\pi_\theta(y_t\mid h_t,s).
$$

Teacher 使用当前 Actor 权重的 `no_grad` forward，在本 update 的 optimizer 更新前
对相同 token 评分。这里没有额外的 Teacher 参数优化，也没有固定 Teacher。

$$
g_t=\operatorname{stopgrad}\left[
\sigma\left(5\left(\operatorname{stopgrad}(\ell_t^T)
-\operatorname{stopgrad}(\ell_t^S)\right)\right)\right].
$$

$$
\mathcal L_{\mathrm{OPSD}}
=\frac{\sum_t m_t g_t\left(\operatorname{stopgrad}(\ell_t^T)-\ell_t^S\right)}
{\sum_t m_t}.
$$

$m_t$ 是 Actor token mask，不包括图片/环境反馈。gate 和 Teacher 均 detach，只有
Student log-prob 得到这项梯度。这是公开实现的 sampled-token 目标，不应描述成
显式遍历完整词表的 KL。

$$
\mathcal L=\mathcal L_{\mathrm{GRPO}}
+0.01\mathcal L_{\mathrm{OPSD}}
+0.01\mathcal L_{\mathrm{reference\ KL}}
-0.001\mathcal H.
$$

reference KL 使用固定的初始策略和 `low_var_kl` 估计器；这个 reference 不是 OPSD Teacher。
技能文本固定在 `configs/svg_static_skill.txt`；它不是每条轨迹生成的 hindsight。
baseline 不生成局部 hint、不使用 Analyzer、不调用 LLM API。

## 实际接线

1. VAGEN reset 按数据行固定 task_id，采集 image-conditioned rollout。
2. `svg_opsd.teacher.attach_teacher_scores` 为每个 decision 构造因果 Teacher prefix。
3. 使用原生 Qwen multimodal processor 重新计算图片 token、grid 和三轴 RoPE。
4. 当前 Actor 的 `compute_log_prob` 无梯度 RPC 评分；逐段检查 Student/Teacher 的
   Actor token ID 完全相同，scatter 到 `teacher_log_probs`。
5. `DataParallelPPOActor.update_policy` 消费这些概率，计算 PG + OPSD + reference KL
   + entropy，然后 backward。不是单独写一个未接入训练的损失函数。

## 相对原版的明确差异

- 采用轨迹 GRPO + 公开 SDAR 损失，而不是 VAGEN-Full 的 PPO/bi-level GAE。
- 固定 train500/dev64、seed2026；50 update 是本 baseline 的预算，不是论文原始预算。
- 上下文8192、超长报错，不使用原始示例的3600/左截断。
- 协议校验更明确，保留特殊 token，阻止“非法格式中偷偷提取 SVG”的回退。
- 绑定环境身份和输入顺序，避免环境池复用后 GRPO uid 对错任务。
- 修复 update 计数、重复末次验证和保存完整性；运行兼容修复都有文件 hash 记录。
- 单卡资源配置只改变执行并行度，不改变每 update 的任务/轨迹数和 loss。

`scripts/verify_loss.py` 检查三个公开损失函数的可执行 AST 一致性。
这不意味着整个 VAGEN 多模态训练器与 SDAR 文本训练器字节相同。
