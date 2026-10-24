# R3 实施状态

2026-10-24。本文件记录开发事实，不替代实验规约。

## 已实现并做CPU检查

- `r3.common`：原子JSON写入、协议种子、代码指纹和过期门槛拒绝。
- `r3.prepare`：官方仓库commit解析与源码锁、模型配置解析；未发出可生产运行的resolved_spec。
- `r3.scoring`：保留单位与标点的最后合法FINAL ANSWER解析、完整boxed回退、NLS、R10、原始token增量解码边界映射。
- `r3.sampling`：逐请求PCG64随机流、稳定top-k/top-p与逆CDF；GPU函数尚待设备对照。
- `r3.manifest`：预先通过原生输入可容纳性检查后的图像去重、任务归属、确定性取样和标签分离。
- `r3.build_jobs`：主实验／对照／机制条件的dry-run注册；完整合成清单校验23160条。production模式故意拒绝，未伪装为已完成的runner。
- `r3.rescue`：V/B同预算救援与随机对照的匹配移除集合。
- 原辅助检查8项与新增核心检查6项合计14项通过。这不是模型或GPU推理验收。

## 尚未完成

五方法原生模型适配与完整推理runner、实际输入审计、各任务官方评分集成、native/engine完整门槛、连续批处理生产后端、全套统计、数据与模型权重准备。不可正式运行；不得据本文件宣称下一轮代码已全部完成。GitHub仍未上传本轮未完成代码。

## 来源锁

本次解析的源码版本：FastV d1659729b5bf1be225e99ee15783deeea80f63b1；PriorTR e442a4c14af1b333067a6b8310c9111cad922033；AgilePruner 6829283ded10bdec58361512a32676c71bbd17d3；VisionZip 8f86b55c6f000eb033e6912538af2dd7dcb30502；LLaVA-Mini 47da1137f7b503be685ab5faa9f64a88b102e786。

完整本地锁及原始服务器检查回执在`.local-preparation/`，该目录不进入GitHub。源代码存档中的未解析符号链接另有记录，不能把文件存在当作递归import验证已通过。

## 第三台服务器与网络

- 第三组连接确认：RTX PRO 6000 Blackwell **Server Edition**，97887MiB，SM120；Python3.12.3，torch2.12.1+cu130，CUDA13.0。没有加载模型或做性能认证。
- 用户选择扩容保留旧数据；随后检查数据盘200GB，总可用约147GB。旧数据未删除。
- 直接连接Hugging Face失败；按用户提供的[AutoDL官方文档](https://www.autodl.com/docs/network_turbo/)，仅在检查进程中执行`source /etc/network_turbo`后，三个主模型的官方API与冻结revision的config均读取成功。未打印代理变量，未改全局shell配置。
- 官方源确认版本：LLaVA7B `4481d270cc22fd5c4d1bb5df129622006ccd9234`；Qwen8B `92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b`；Mini8B `690d4c0b40c12207561d84507787b61030bfd306`。
- Mini config：compressor_size=1，prefusion_layer_num=4，resolution_ratio=1；配置值不替代实际视觉/文本路径计数，仍须原生前向验收。
- 本机曾通过hf-mirror读取配置，记录transport来源；后续模型准备优先使用已经连通的官方源，不能把镜像来源标成直接官方访问。

## 下一步

先实现并对照作者原生路径，冻结模型及CLIP分片/processor/tokenizer；完成原生输入清单后再接入执行与验收。只有完整代码检查后才按用户顺序上传仓库。M60还需要真实双人标注与一致性记录，不伪造第二标注员。
