# 数据、模型与评测身份

所有资源来自 `sources.lock.json` 指定的公开来源，运行前自行下载；不需要私人备份。

## 构造原则

- 使用 `starvector/svg-icons-simple` 的固定 revision。
- 训练候选来自官方 train；dev 候选来自官方 val。
- seed2026 确定候选顺序；选择500个train和64个dev任务。
- 排除不安全/不能正常渲染的 SVG，防止外部链接、脚本、文件读取。
- 按源码与渲染后 RGB hash 去重；保护官方 val/test，避免重复图像进入训练。
- 官方 test 可以被 CPU 读取/渲染以建立去重集合，**不进行 test rollout、不以 test 成绩
  筛选超参数、不作为周期验证集**。不要把这种去重读取写成“完全未读取 test”。
- 目标图按固定渲染流程生成；任务身份包含源 split、索引、文件名和 RGB SHA。

不依赖已有轨迹或 Good/Bad 标签，本仓库也不做 Route-SFT 数据构造。
只发布公开资产获取与环境任务构造逻辑，不发布私人采样/标注库。

## 数据产物

`prepare_data.py` 输出 train500/dev64 Parquet 和 dataset manifest。每一行把 task_id
与固定 source_index、source_filename、target_pixels_sha256 绑定。环境 reset 必须
命中同一条公开源记录，渲染 RGB hash 不一致则报错，不能静默随机抽一个新任务。

Parquet 的绝对文件路径依赖运行方自选的 work 位置，因此**不同机器的文件 SHA 可能不同**。
跨机器比较应同时核对 source revision、选中 task_id 顺序、RGB hash，以及各自
解析后的路径，而不能把 Parquet 二进制完全一致作为跨路径必要条件。
同一机器上比较不同方法则直接共用这份已生成的数据及 manifest。

## 评测

- 初始 step0 和 step10/20/30/40/50：同一 dev64，greedy，每个任务1条两轮轨迹。
- dev不参与梯度；不生成 Teacher skill/hint，不更新参数。
- 保存每个任务的 SVG 输出、图像相似度、协议/格式指标及整体均值。
- 指标以代码实际输出为准：DreamSim 相似度、DINO 相似度及加权 reward；
  不转换成无定义的“成功率”。
- 保留最后模型及开发集曲线，不能把开发集最优分数伪称最终独立 test 分数。

渲染库和系统 Cairo 版本可能导致像素差异。必须保留 asset/data/run manifests，
并使用相同系统镜像/依赖比较不同实验臂；运行方不能悄悄重建另一份 dev 集。
