# 故障排查与运行边界

## vLLM / CUDA

- 本入口要求 V0 生命周期：`VLLM_USE_V1=0`。不要换成 vLLM0.11 后继续使用旧
  `driver_worker` / FSDP 权重同步路径。
- Blackwell 使用 CUDA12.8/Torch2.7 构建；旧CUDA12.4轮子不等于支持新卡。
- FlashAttention 轮子必须同时匹配 Torch、Python 和 C++ ABI。
  安装脚本选择官方2.8.3对应轮子；不能把导入成功当作CUDA backward成功。
- `no kernel image` / `invalid device function`：检查GPU架构与torch/FA编译支持，
  不要重复启动相同失败任务。

## 显存与主存

默认microbatch1、Teacher评分batch1、生成并发16、参数/optimizer offload。
先检查实际 cgroup 内存额度，而不是只看宿主机 `free`。降低生成并发是资源调整；
减少每任务8条轨迹、改reward、关闭OPSD则是科学配置变化，不能冒充同一baseline。
默认只保证配置明确，不声称所有80GB卡都已实测通过。

## 网络

首次准备需要访问公开模型/数据和Github源码。失败可重跑准备命令并复用下载缓存。
不要在训练中悄悄替换模型 revision 或使用随机初始化的 DINO/DreamSim。
只使用自己的 Hugging Face 登录（如公开服务要求），不要向代码填写或提交他人的key。
baseline不需要OpenAI/OpenCode/Analyzer服务。

## SVG

脚本、外部文件/URL引用、非法XML或不完整SVG不执行。CPU渲染依赖Cairo，
GPU负责策略及图像reward模型。协议失败是需要统计的结果，不强制补reasoning、
不替Actor生成SVG、也不把非法响应交给宽松的全文提取器执行。

## 恢复与保存

只在完整 checkpoint 的完成标记、文件和身份检查通过后恢复。文件写了一半时，
回到上一个完整断点；不要把 `latest_checkpointed_iteration.txt` 单独当完整性证据。
保持训练起点、数据、总步数、配置和代码版本一致；恢复不应额外跑第51步。
保留日志和 validation 文件，不因只保留两份模型而删除评测历史。

## 已完成的验证不等于什么

公开损失源码AST对照、CPU梯度/实际Actor hook/固定任务测试，证明接线合同可检查；
并不能证明在任何新驱动/新卡上GPU训练一定成功，也不构成50步学习效果报告。
运行方应先跑仓库提供的1-update smoke，保留日志后再跑正式50步。
