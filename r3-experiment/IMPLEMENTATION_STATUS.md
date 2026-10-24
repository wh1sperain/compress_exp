# R3 实施状态

2026-10-24。本文件记录开发事实，不替代实验规约。

## 最新：用户暂停正式推理，复用R2分页方案

2026-10-24并发调优续跑：按用户“测试6路，若还能提升继续提高，最快完成”要求，保存4条5701/1673/696/3 token前缀并保留14491成功结果。4/6/8/10路均通过短数值、释放补槽与原32768上限预留；含原GPU采样器的合成混合长短KV吞吐分别180.37/239.44/284.55/320.28 token/s，短缓存238.62/353.60/456.60/561.99，全长缓存145.17/181.47/206.08/224.11。10路在测试范围内最快，普通峰值83.59GiB；校准集中prompt最长的10图PriorTR25预填充另测PASS，峰值84.28GiB、余量10.69GiB，11路增加6GiB池后低于8GiB预留，因此不继续增槽。上述为合成缓存测速，不当作自然整体倍率。

正式Qwen8/4现为10路、60GiB池，LLaVA/Mini仍4路。release `0acd866fcaeb62a7`、coordinator55222、恢复回执`tuned-recovery-1792850925841791201.json`；旧恢复状态和原KeyboardInterrupt均保留，继续原21900任务。新结果标注对应恢复来源，输出上限/采样/分辨率未改，仍不解析评分。

2026-10-24后续：用户明确“完成后续，继续实验”。恢复专项10项CPU测试、四槽原32768上限预留/补槽/页面归还、Qwen8 PriorTR50与Qwen4 PriorTR25短数值检查通过。已用release `fb60dddd6256713c`启动恢复协调器54472、Qwen8 Full worker54486；4路分页KV/Graph，缓存池24GiB。旧14397成功结果全部按原绑定验证并跳过，3条原前缀逐token强制重放、不重抽，PCG64 uniform游标以原生成长度续接。原暂停失败及旧控制状态保留；新状态看`recovery_status.json`，旧暂停父进程已退役。正式仍只生成原始输出，不解析评分。后端迁移来源通过独立恢复回执及每条新结果的`execution_recovery_id/recovery_code_sha256`标记，不能将混合后端视为严格逐token等价。

- 正式运行已暂停，14397条成功结果原位保留；三个未完成请求的29091/965/34个原token已落盘，父协调进程停止，原KeyboardInterrupt记录保留，r3定时任务暂停。
- 候选`r3/qwen_paged_decode.py`适配旧`Compress/src/methods_qwen3.py`分页KV/FlashInfer/CUDA Graph。保留R3原生prefill和mRoPE；普通视觉注意力改SDPA，作者请求权重路径保留。原生参考先执行，候选后构建，避免参考被候选设置影响。
- Full与PriorTR25各3图32步数值检查PASS：KL最大0.002831/0.007909，TV最大0.032169/0.052142；KV一致，argmax不同0/1处，不能声明逐token严格等价。
- release `8d7abae5955b2de3`完成候选测速：3请求、重复KV模拟约30000/1100/200长度，16步纯解码，旧dense 9.82 token/s、新paged 161.65 token/s，16.46倍。排除prefill/采样，非自然完整吞吐，不据此承诺整轮加速倍数。
- 尚未接入正式调度/原prefix恢复，未恢复正式实验。需完成页面复用/补槽、原token和RNG游标恢复、来源与新后端版本绑定后才可恢复。原OOM与诊断驱动失败回执均保留。

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

2026-10-24 09:07检查：Mini与CLIP分片下载校验已完成，Qwen4B仍在下载。已逐项读取7个模型smoke回执：Full、PriorTR50/25、AgilePruner50/25、VisionZip50/25权重加载无missing/unexpected/mismatched keys，层输入数量符合预算，短合成生成完成；这不是完整模型等价门槛。FastV50在外层LLaVA.forward收到cache_position时报TypeError。新增显式兼容包装，将cache_position原样传给作者LlamaForCausalLM；不丢参数、不修改作者磁盘源码。CPU18项通过。新检查输出进入唯一attempt目录，旧成功/失败/prefix记录不覆盖；恢复只检查FastVFull、FastV50/25、MiniNative。

2026-10-24 08:52定时检查：LLaVA7B与Qwen8B分片下载及校验已成功；Mini元数据请求遇到HTTP503，原准备和等待进程失败记录保留。新增已下载模型逐文件重新校验后无网络复用（损坏即拒绝，不覆盖），CLIP优先准备，各方法仅等待自身所需模型；17项CPU测试通过。基础设施恢复不会生成或重抽研究答案，正式实验仍未启动。

完成模型与CLIP下载校验，运行作者完整模型检查，继续Qwen路径、输入清单及正式runner开发；完成各门槛后再启动正式研究任务。代码迭代同步仓库及不可变服务器release。M60还需要真实双人标注与一致性记录，不伪造第二标注员。

2026-10-24 09:22：全部五套模型/CLIP权重准备完成。FastVFull、FastV50/25短合成生成通过，前两层675、后续分别675/387/243 token，加载无缺失/多余参数；仍不是完整KV/logits/长轨迹认证。Mini严格加载失败：391个unexpected keys全部属于checkpoint内vision tower，零missing/mismatched。作者延迟构造未建立该子模块；使用作者已有unfreeze_mm_vision_tower构造开关提前建立模块，使checkpoint权重正常接收，随后明确冻结全部参数。未忽略任何不匹配键，未改作者算法，原失败保留。18项CPU测试通过，Mini独立新attempt待GPU验证。

2026-10-24 09:37：Mini独立smoke协调器完成、退出0（完整认证仍false）。新增Qwen作者原生模块接入：进程内按原transformers包名载入冻结作者qwen3目录，不改安装目录、不执行作者setup删除逻辑；独立qwen环境使用作者README要求的transformers f8f2834e1a并检查pip解析的完整commit来源，保留Blackwell torch。新增Full/PriorTR50/25合成图检查入口、严格加载回执及逐层输入长度记录。18项CPU检查与语法检查通过，Qwen运行时尚待验证。正式runner、五任务输入审计与长/自然门槛未完成。

2026-10-24 09:52：Qwen隔离环境安装成功，但Full模型加载将vtr_config作为HF kwargs传入，触发GenerationConfig序列化错误。已按冻结作者from_pretrained_vtr的原顺序改为加载完成后设置VTRConfig、strategy和text model；继续保留严格loading_info验收，不给VTRConfig伪造序列化方法。原失败保存，新attempt复验三组，未产生研究答案。

2026-10-24 10:07：Qwen已完成权重载入，写加载诊断时HF5返回set导致JSON序列化失败。新增保留全部键/shape的递归转换，不吞掉缺失或不匹配；新增回归测试，19项CPU通过。原失败保留，独立新attempt继续Qwen三组检查；正式研究未开始。

2026-10-24 10:22：QwenFull/PriorTR50/25合成短生成全部完成、协调器退出0；这不是长轨迹或后端认证。数据盘仅发现MathVista及旧Doc/Info池，ChartQA/GQA尚待官方来源准备。新增500/100清单冻结CLI：要求逐模型输入审计字段与上下文预算、拒绝重复UID/覆盖、生成侧字段白名单、嵌套审计白名单和评分答案隔离，记录清单SHA与排除记录。20项CPU测试通过（含600题合成清单完整性/标签不泄露/不可覆盖）；尚未冻结真实样本。

2026-10-24 10:37：有界检查无新模型失败。新增ChartQA官方vis-nlp/ChartQA test_human获取器：冻结commit/tree，逐Git blob验证JSON和全部human-test涉及的原图，生成未审计候选而不提前选100/20；单进程锁、进度及独立失败回执。不下载train/augmented，不替换为社区数据。GQA官方页面当前访问超时，尚未获取。21项CPU测试通过；真实native输入审计仍待执行。

2026-10-24 10:52：ChartQA raw官方地址TLS超时，原失败与锁保留；有界网络诊断确认加速环境下相同地址200，直连超时。显式恢复原固定来源。GQA官网页面503但Stanford官方questions1.2.zip返回200（1498616372字节）；新增官方问题/图像ZIP后台准备，文件SHA与ZIP CRC校验，只解出balanced-val引用图像，保留原压缩包，不预抽样；盘空间门槛32GiB。21项CPU和语法检查通过；数据获取不是输入验收或正式研究。

2026-10-24 11:07：ChartQA官方源1250题/625图完成（固定044eabfc306abfe9340c5741f0093aefc5973d06），GQA仍正常下载问题包。目标Qwen处理器实测size shortest_edge262144/longest_edge4194304、patch16/merge2。新增CPU原生输入审计：图像哈希核对、实际processor展开prompt、grid/token数一致性、8192门槛、处理器/tokenizer/template/prompt来源哈希、无答案审计输出；不生成答案，不预抽样。22项CPU测试通过。A/C输入审计与正式runner、长轨迹门槛仍未完成。

2026-10-24 11:22：ChartQA全部1250题Qwen原生输入审计退出0。GQA长连接在163577856字节后ChunkedEncodingError，原partial与失败回执保留。新增4路8MiB HTTP Range传输：服务器206/Content-Range/ETag或Last-Modified验证、分段SHA、不可变成功块、完整组装SHA及双份空间预留；不做无界重试，不改来源/采样。24项CPU测试通过，含模拟传输组装与拒绝错误字节范围/变化来源。

2026-10-24 11:37：GQA分段返回非206，保留失败；定向小范围与8MiB含If-Range探针证实206且ETag未变，网络不稳定。增强异常HTTP回执与stream模式，避免异常整包响应被读入内存；显式恢复，未无界自动重试。新增LLaVA原生pad336/CLIP14 patch/tokenizer输入审计，检查单图占位、实际像素张量和Full展开2048上限；Mini真实预融合审计仍待实现。24项CPU/语法检查通过。


2026-10-24 12:08：Mini全部3250题实际压缩/四层prefusion输入审计退出0。逐文件SHA及记录完整性复核9750条（三模型×3250题）通过，全部原生输入合格。独立图像数：DocVQA390、InfoVQA223、ChartQA625、MathVista972；尚未跨任务去重或冻结最终500/100。ModelScope问题包已校验完成，图像包检查时下载1.64/21.82GB；无新失败。正式runner与完整后端/长自然轨迹认证仍未完成，正式研究未启动。


2026-10-24 用户外部复核意见落实：确认并修复恢复时较短前缀覆盖旧检查点风险（未完整重放不发布，更长前缀发布前保留旧版本），并在Agile原生与作者函数入口拒绝DIST_THRESHOLD环境覆盖。26项CPU测试通过。新增真实图像1025步teacher-forcing逐层KV增长/位置/mask形状与选定logits探针，再按原上限自然生成；独立协调器PID14515/release0b6ab58ff2fbfbc9正在执行，优先FastV。诊断不是独立作者等价证明，所有方法总状态保持INCONCLUSIVE，正式生成禁止。仍需独立作者QP/索引/特征、FastV官方完整入口与当前组合路径、DeepStack/mRoPE、Full跨环境差异及恢复GPU证据。Mini已有3250候选真实预融合输入审计完成，不能代替原生前向数值对照。


2026-10-24 13:49：五任务135312候选×三模型405936条输入审计全部完成并按完成/退出码/SHA/UID×模型覆盖复核。冻结正式500题（各100）及校准100题（各20），600幅原图互不重复、标签单独保存、生成清单不含答案。main SHA9dc50dac4a5da8c8c1682c5fe3dff67fbb03303559c992ae52ace5856692439a，cal SHA b14ad4478904a91b7500171970b694fa16a25010b30fde4084c9169eb5f4da58。输入清单验收PASS不代表native gate。现有16条件单图1025步缓存/自然诊断、8探针Full比较、同激活作者选择器重放、合成GPU token/RNG恢复已完成；独立作者端到端对照、真实模型恢复与完整长/自然backend认证、正式runner仍未完成，正式研究未启动。M60双人标注仍缺。

2026-10-24 持续开发：16条件真实模型前缀恢复诊断已完成（同已加载模型，非跨进程KV反序列化）；新增带native/engine/输入/源码锁门槛的B1 worker、持久模型session、成功结果不可覆盖与失败不自动重抽。35项CPU回归通过。服务器运行13条件×五任务校准图的session输入一致性检查；FastV发现仅HF版本序列化_valid_processor_keys元数据差异，保留原失败，新增严格限定归一化并同时记录原始hash，其余图像变换字段仍须全字典一致。独立FastV官方HF完整入口对照已编码（全张量映射+Full/50/25预填充和1024步解码），待当前GPU检查结束启动。主运行器仅实现B1主条件，控制臂扩展、完整数值/自然认证与正式注册仍未完成；正式研究未启动。

2026-10-24 14:44：FastV官方HF完整入口独立对照通过（同一校准图，逐张量映射核对；Full/50/25各预填充+1024步，最大KL0.003682、TV0.037382，argmax与逐层KV无不一致）。只覆盖该图/长度，不宣称全native gate。FastV输入复测通过，累计13/13 session条件通过。新增完整阶段注册与串行隔离调度：先全量核对门槛/注册哈希/重复job/总数，再启动worker；成功不覆盖、失败不重抽，控制未实现时启动前拒绝。阶段计数测试发现AgilePruner正式名与诊断别名差异，已按正式名修复。37项本地测试通过；尚未启动正式研究。

2026-10-24 用户明确要求并行后：三路并行实现控制适配、Qwen加载入口对照、Mini加载入口对照；主线程完成集成。49项CPU测试全部通过（d2l解释器，无skip）。Qwen4补充输入审计600/600完成，原500/100清单保持不变，新增旁路审计哈希绑定。FastV75/37p5与Random25_L2已接入，随机控制仅替换冻结作者两处top-k表达式，GPU门槛仍待验收。release abfbaf6ffd3f2146/64文件校验，集成验证协调器PID32293已启动：控制输入、Qwen4输入、Qwen作者/local各三条件、Mini作者/session及比较，单GPU串行自动接续。加载入口对照共用冻结作者模型类，不称算法独立重实现；正式研究未启动。

2026-10-24 14:59周期：有界检查显示集成验证仍正常运行，已完成控制与Qwen4输入检查，推进至Qwen PriorTR25作者路径。并行补齐原token分区（7项测试）、比较回执严格判定（4项测试），新增100图×3seed双后端自然统计验收（3项测试，分任务质量避免抵消）；统计工具不发native gate，Qwen思考行为单列待审，不擅自新增阈值。已验证63项CPU测试通过；真实tokenizer接入、离线评分和完整认证仍未完成，正式研究未启动。当前运行旧release不热改，完成后须逐份读取比较JSON，不以进程exit0代替PASS。

同周期并行交付收拢：离线四任务评分及8项测试完成，GQA评分未独立核实保持Blocked；全套71项CPU通过。原token分区/评分尚待真实tokenizer及流水线接入，不能据此宣称正式可运行。

2026-10-24 用户要求30分钟内完成并删除冗余测试：删除集成协调器中的服务器重复CPU单测，撤回尚未执行的FastV全3072步追加扫测；待执行项仅保留未覆盖门槛，已有成功回执复用，原失败和证据不删除。Mini新release 2b209898ce01080d独立作者恢复及比较均PASS（回执SHA56ed5ba1f2534cee740bf2fbc71a7620e4a22db7ee50595b913febac14a94df2），Qwen三组入口比较PASS；均为单图1025步范围，非全native gate。全量自然认证尚缺两侧worker，不得将相同路径双跑伪装等价。截止UTC07:47:47只据实际完成报告。

2026-10-24 用户进一步授权大幅缩减长程测试，仅保证运行正常：本轮改为同图Qwen Full/PriorTR25 local各4096强制步，检查finite/logits/36层缓存增长，取消32K长程矩阵及作者重复路径；不声称原完整long/backend gate通过。PID36480/release22c15b7d2d480bda已启动。Mini独立恢复已PASS；16份已有自然记录的冻结tokenizer离线分区全PASS，未重新生成。GQA严格exact评分已按冻结作者eval核实；真实prompt IDs与EOS配置现保存到生成记录。

2026-10-24 并行推进：补齐Qwen/Mini/FastV及A其他方法独立作者自然参考worker，仍共享固定作者forward/HF内核，未声称分页引擎实现；共同natural_preparation job_key/种子/原cap不变，记录原prompt与EOS。Qwen Full候选五任务5条及参考5条已完成，自动离线解析评分已通过（10/600，完整自然认证仍INCONCLUSIVE）。Mini参考与评分自动接续；A五方法各任务1题×两侧共50条integration队列PID38514/release7b1bf1f8231574f5按单GPU顺序自动运行，失败保留并继续其他独立条件，不重复旧成功/长测。全阶段仍属集成验证，正式未启动。

2026-10-24 15:47截止复核：当前集成队列全部输出到齐，Qwen Full 10、Mini参考1、A五条件各10，共61条解析评分通过。Qwen Full及A Full/PriorTR25/AgilePruner25/VisionZip25五对原token完全相同；FastV25为4/5对相同，余1对长度均20，不能称严格token等价。规模不足完整自然认证，所有统计仍INCONCLUSIVE。发现LLaVA Full五题4题缺FINAL ANSWER标记，其中InfoVQA一题整段答案影子评分1而V3主评分0；仅诊断，不改提示/主评分，不将格式失败等同于内容错误。原始记录均保留。正式研究仍未启动。

## 2026-10-24 正式推理启动政策
用户明确要求准备正式实验，禁用答案解析和评分，结束后下载全部结果。注册主实验19500+对照2400=21900唯一任务，M60人工阶段单独暂缓。采用显式user_reduced_native_b1许可并保留原完整认证未建立事实；旧严格gate路径不伪造通过。新增唯一流水线、GPU记录、完整结果导出与本地SHA核验。实际启动以formal_inference_launch回执为准。

## 2026-10-24 真实批处理执行更新
按用户强调GPU_EXECUTION_R3后未启动B1正式任务。已实现单模型常驻、原生prefill、逐层独立KV的真实批解码和连续补槽。六个代表条件及Agile/VisionZip/Qwen4补充短数值检查PASS，Qwen4路满cap保守预算INFEASIBLE（未分配、未OOM），独立新release改Qwen3路，其他4路，不改研究输出上限。三family满cap资源检查及LLaVA/Mini五题EOS补槽已PASS，Qwen原cap五题补槽运行中；全部通过后自动21900正式启动，不做答案解析评分。当前release340eebd2e4c82bbd，GitHub214b53e070d6c30c9096713ec816a1c71398061b。

## 2026-10-24 16:38 正式推理已启动
Qwen原cap五题自然EOS/补槽检查PASS，累计1693生成token，无触顶，峰值3路。正式run r3-20261024-batch-inference-v1于16:37:46启动，release340eebd2e4c82bbd，lifecycle42345/coordinator42346；首次回执核对115个真实result文件，当前llava7-Full。总21900，LLaVA/Mini4路、Qwen3路，单份常驻权重及连续补槽。答案解析和评分关闭，原完整native/engine认证仍未建立，采用user_reduced_native_batch并如实保留范围。15分钟定时监督，结束自动导出全部原始结果，本机归档与逐文件SHA验收；M60人工阶段另行暂缓。
