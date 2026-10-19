# 发布检查范围

本仓库不附带训练成绩，也不把静态检查当作训练完成。

## 已执行

- `python -m pytest tests -q`：**58 passed, 1 skipped**。
  本次 CPU 检查使用 Python 3.13 / PyTorch 2.8 CPU；正式训练环境另固定为
  Linux Python 3.12 / PyTorch 2.7 CUDA 12.8。
- 跳过项只是在本地没有额外复制 SDAR reference checkout 时的重复 AST 测试；
  `python scripts/verify_loss.py` 已单独联网验证三个函数与公开固定版本的可执行 AST 一致。
- 已从公开 GitHub 获取固定 VAGEN/VERL 源码，应用 overlay 与运行补丁；
  `bootstrap_sources.py --verify-only` 和运行补丁幂等/篡改拒绝测试通过。
- CPU 覆盖实际 Actor update hook、Teacher/gate detach、反向梯度、mask、
  token 对齐、GRPO 分组、任务身份、完整 checkpoint 登记/校验/保留、
  50 步与 smoke 配置合成，以及数据路径迁移的合成测试。
- 发布文件隐私扫描和人工审查：不包含私人路径、服务器凭据、原始运行日志、
  私人数据或模型权重。提交历史从公开包单独新建。

GitHub Actions 在 Linux 上重复 CPU 检查，状态以该 commit 的实际 CI 结果为准。

## 尚须运行方执行

- 完整的 Linux/CUDA 依赖安装与全部公开资产下载。
- 使用公开源实际构建 train500/dev64，保留首次生成的身份 manifest。
- `--smoke` 的 GPU rollout、图像渲染评分、Teacher 评分、反向传播和完整保存。
- 正式 50-update 训练、固定验证与性能比较。

尚未用该公开发布包完整跑完 GPU smoke 或 50 updates，因此不提供未经测量的
显存峰值、耗时或最终分数。CPU 测试通过不能替代上面的验收。
