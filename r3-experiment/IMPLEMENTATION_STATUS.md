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

完整五方法原生模型适配与正式runner、实际输入审计、各任务官方评分集成、native/engine完整门槛、连续批处理生产后端、完整分析报告、数据与模型权重准备。不可正式运行；不得据本文件宣称下一轮代码已全部完成。

## 本次继续开发与部署

- 已上传私有仓库 https://github.com/wh1sperain/compress_exp 的 `r3-experiment/`；凭据、模型、数据、轨迹与大日志不在提交清单中。上传服务器采用不可变哈希目录 `Compress/r3-releases/`，逐文件SHA256校验。
- 新增 `native_sources`：从固定作者源码加载未改写函数，严格校验来源；VisionZip公开参数dominant=243/122（包含CLS），contextual=45/22。
- 目标RTX PRO6000上完成BF16合成组件检查：PriorTR A/B查询范围分别保留、Agile两个预算、VisionZip实际输出288/144及CLS、64路CPU/GPU采样token相符。**不是完整native gate**；FastV真实模型/cache与Mini预融合未通过该组件检查认证。
- CPU测试目前16项。新增逐题三重复平均后bootstrap、缺配对/重复拒绝测试。
- `reference_stream`已实现请求独立uniform、128步前缀保存、恢复一致性检查；尚未经过真实模型生成验收。
- `native_smoke`已写入作者模型加载、BF16、原生模板、层输入预算与8步合成检查入口；这8步只用于独立冒烟，不能当研究轨迹或替代正式输出上限。
- `environments`建立三套隔离环境，实测transformers4.37.2/4.43.1/4.39.0均保留torch2.12.1+cu130。FastV运行将读取冻结作者的4.39.0.dev0源码；Qwen原生环境尚待接入。
- `analyze`已实现完整job拒缺/拒重复、种子配对、三重复先平均、任务内图像簇bootstrap、A/B分族区间。未在R3真实数据上运行，尚不能替代全部报告与机制分析。
- 模型下载首次被默认PyPI镜像阻断，已保留失败记录并改官方PyPI；随后HF Xet大文件传输失败，保留原文件和失败记录，改为独立HTTPS下载目录。不得把较早的失败回执或组件成功当作当前全流程成功。
- 后台准备与preflight协调器只执行下载、环境和合成模型检查，**不会自动伪造门槛或启动未完成的正式runner**。剩余开发/认证必须继续，M60双人标注仍未就绪。

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

完成模型与CLIP下载校验，运行作者完整模型检查，继续Qwen路径、输入清单及正式runner开发；完成各门槛后再启动正式研究任务。代码迭代同步仓库及不可变服务器release。M60还需要真实双人标注与一致性记录，不伪造第二标注员。
