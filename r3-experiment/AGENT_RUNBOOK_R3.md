# R3 执行规约：原生骨干上的视觉压缩与生成长度变化

版本：`visual-compression-r3.0-native-rtxpro6000`；修订日期：2026-10-22。

2026-10-24执行修订（用户最新明确授权优先）：删除冗余测试，原长程缩减为Qwen Full/PriorTR25各一图4096步运行稳定性检查。用户进一步强调GPU规约，未启动B1正式任务；新增单份权重多请求原生prefill+批解码、连续补槽与完整cap显存准入，以 `user_reduced_native_batch` 许可启动main+controls 21900次。新增batch路径须通过必要数值/KV、资源及slot检查；原完整native/engine和6000自然等价认证仍未建立，不写成通过。分页与Graph未启用。正式阶段不运行答案解析与评分，只保留完整原始输出和运行元数据；完成后将全部实验结果下载本地并校验。M60缺双人标注，仅该阶段暂缓。下文研究条件、样本、原生处理、采样和输出上限保持；与本条冲突的旧验收规模与自动评分步骤由本条替代。

本文件完整替代 R2 的实验执行部分。它是开发与运行任务书，不是已经完成的 GPU 实验。文献/仓库依据见 SOURCES.md，硬件执行细节见 GPU_EXECUTION_R3.md。JSON 与正文冲突时停止并报告，不自行选一个执行。

## 1. 本轮变更与研究目标

保留问题：视觉 token 压缩是否增加生成长度、异常重复和未能及时交答案的概率？额外生成是否有效？

修改范围：
- FastV、PriorTR 的 LLaVA 路径、AgilePruner、VisionZip 使用作者支持的 `liuhaotian/llava-v1.5-7b`。这是 Vicuna/LLaMA 系列的多模态模型，不是裸 LLaMA。7B 是这些原生支持型号中最接近目标 8B 的选择。
- 新增 VisionZip（CVPR 2025；免训练选择+合并）与 LLaVA-Mini（ICLR 2025；学习式压缩+模态预融合）。不增加两个相似的注意力剪枝器。
- LLaVA-Mini 使用官方公开 `ICTNLP/llava-mini-llama-3.1-8b`，不把它移植到 Qwen。
- Qwen 仅保留 PriorTR 作者已经支持的原生 Qwen3-VL 路径，作为长思考验证；停止运行 FastV-Qwen、AgilePruner-Qwen 的自定义移植版。
- B1 是数值/行为参考，不再是整轮生产运行的硬约束。通过验收后使用连续批处理和分页缓存。禁止为加速修改研究终止规则。

## 2. 三条主实验轨道

| 轨道 | checkpoint | 条件 | 科学用途 |
|---|---|---|---|
| A | liuhaotian/llava-v1.5-7b | Full；FastV、PriorTR、AgilePruner、VisionZip 各 50%/25% | 同权重、同任务、同提示下的主要配对实验 |
| B | Qwen/Qwen3-VL-8B-Thinking | Full；PriorTR50/25 | 原生长思考模型验证，不承载五方法排名 |
| C | ICTNLP/llava-mini-llama-3.1-8b | 原生已训练压缩配置 | 学习式压缩的架构覆盖与描述性结果 |

C 没有本方案核实过的“相同权重且无压缩”的 Full。不得关闭压缩模块、改变查询数、换另一架构模型，然后称为 Mini 的无混杂 Full。C 不报告与 LLaVA7B 的配对长度膨胀率，不与 A 混成因果排名。Mini 论文某些 Vicuna7B 表格与公开 Llama3.1-8B checkpoint 不是同一个模型。

A 的四种方法尽量统一权重、原生 CLIP、聊天模板、采样器和 reference decode。各作者项目在关闭压缩时必须对公共 Full 通过验收。FastV若用作者HF格式分支，可做原始checkpoint到官方HF命名的无损tensor映射，但必须逐张量验证权重对应、视觉处理和模板相同；这不是允许换成另一个8B骨干。若某个分支关闭压缩后已经不同，先定位其模板、权重、精度、位置或依赖差异；不得把不一致的分支当作公平方法对比继续运行。

## 3. 归档、环境与版本

根目录 `${COMPRESS_ROOT:-/root/autodl-tmp/Compress}`；输出新建 `runs/visual-compression-r3/`，不覆盖 R2 或旧正式实验。

目录：`environment/ manifests/ configs/ source_snapshot/ tests/ jobs/ raw/ metrics/ annotations/ telemetry/ reports/ failures/`。

准备时必须锁定：每个方法 repo 完整 commit SHA；checkpoint revision SHA 和全部分片 SHA；CLIP 权重；tokenizer/processor/chat-template SHA；torch、transformers、CUDA、driver、flashinfer/flash-attn、Python；所有本地递归 import（包含 scoring.py）。方法 repo 中不可解析的 `revision=null` 是待准备字段，不代表允许正式运行使用浮动 main。

PriorTR 先使用 `e442a4c14af1b333067a6b8310c9111cad922033`；AgilePruner 先使用 `6829283ded10bdec58361512a32676c71bbd17d3`。其余方法在准备阶段解析当前官方 commit、核对本文所述代码行为后冻结。若固定 commit 不含所需原生路径，记 blocked 并提出有证据的新版本补丁，不擅自从另一个实现拼装。

不要把多个旧 repo 的 `llava` 同名包混装。为 A、B、C 建立隔离环境或显式 namespaced vendor；每个进程只加载其原生实现。允许现代 torch 与旧作者算法代码的兼容性补丁，补丁必须保留算法并通过作者参考对照；不允许静默把作者方法换成名字相同的第三方实现。

主计算 BF16，参数及 KV 均不量化。LLaVA checkpoint 配置原为 float16，本实验明确统一转 BF16；reference 与 production 使用完全同样的转换，报告注明。若某算子不支持 BF16，阻塞并修复/验收，不自动将某一条件切成 FP16 或 FP32。

RTX PRO 6000 的环境与并行执行按 GPU_EXECUTION_R3.md。不能原样安装旧项目的 torch2.1/cu121 钉死依赖覆盖支持 Blackwell 的 torch。

## 4. 数据与固定抽样

沿用五任务：

| task id | 来源/分割 | 正式/校准独立图像 | 主评分 |
|---|---|---:|---|
| DocVQA | 旧冻结 500 题池 | 100 / 20 | ANLS |
| InfographicVQA | 旧冻结 500 题池 | 100 / 20 | ANLS |
| ChartQA-human-test | vis-nlp/ChartQA，test_human.json | 100 / 20 | 官方 relaxed accuracy + exact |
| GQA-balanced-val | 官方 balanced validation | 100 / 20 | 官方 accuracy |
| MathVista-testmini | AI4Math/MathVista，testmini | 100 / 20 | 按 answer_type/precision 规范化 accuracy |

旧 Doc/Info 为此前筛过的大图题池，不代表两个数据集无条件随机总体。新增三任务不套用原图 2M 像素或 Qwen 2048 个视觉 token 的过滤器。不要按 Full 正确/长度、方法好坏选择主样本。

候选规则：有公开标签、单图、可解码 RGB、问题非空。按图像原始字节 SHA256 去重；跨任务相同图按上表任务顺序归属。先按每题在各主轨道的原生输入检查可容纳性，再按下面规则取样：
1. 每图只保留 `sha256('r2-question|'+uid)` 最小的一题。
2. 图像按 `sha256('r2-image|'+image_sha)` 升序，前100正式、后20校准。
3. 过滤、替补与旧 R2 manifest 的差别全部写入 `manifest_diff.json`，必须发生在读取任何本轮输出之前；已开始推理则不再替补。
4. 候选不足120幅时停止该任务准备，不能偷偷换社区数据或补无标签 test。

输出 `main_manifest.jsonl` 500 行、`calibration_manifest.jsonl` 100 行。每行包含来源 revision、uid/task、原图哈希/路径、问题/选项/答案类型/精度/标签，以及每模型的处理后尺寸、实际视觉数、expanded prompt 长度。参考答案单独保存在评分侧，生成 worker 不读取答案。

原生图像处理：A 使用作者 LLaVA pad+336 输入及 CLIP patch 路径（Full 应为576）；B 使用 Qwen processor min_pixels=262144、max_pixels=4194304；C 使用公开 checkpoint 的官方单图处理路径，读取并冻结 crop/resize、prefusion 与压缩参数，实际计数。不能把 C 的宣传“1 token”当作所有分辨率/裁图路径的实际总数，也不能仅把 A 改成高分辨率而保持同一实验名。

各模型看到同一原始图，但编码分辨率不强行一致。跨模型的差异不能仅归因于参数规模。各任务 Full 的质量、图像文字密度与原生视觉预算必须单独报告。

## 5. 提示、生成与长度定义

A 原生对话模板 `llava_v1`；C `llava_llama_3_1`；B 官方 chat template。保留模板自带系统文本，不额外拼接 Qwen 模板。原问题加以下统一后缀：

```text
After reasoning, end your response with one line in this exact format: FINAL ANSWER: <answer>
Give only the requested answer after FINAL ANSWER:, without extra explanation on that final line.
```

多选在后缀前按原顺序列 `(A) ...`，末尾追加 `For this multiple-choice question, give only the option letter after FINAL ANSWER:.`。

生成设置：sampling=true，temperature=1.0，top_p=.95，top_k=20，repetition_penalty=1.0，presence_penalty=0，min_new_tokens=0。除温度对照外不变。它是本研究的统一采样设置，不冒称各论文原始 greedy benchmark 设置。

| 模型 | 总生成上限 | 上下文与停止规则 |
|---|---:|---|
| A LLaVA7B | 2048 | 原生4096窗口；Full expanded prompt≤2048；原生EOS/模板合法终止符 |
| B Qwen8B / 小型4B桥接 | 32768 | runtime context65536；prompt≤8192；正常EOS；思考闭合与EOS分别记 |
| C Mini8B | 2048 | 所有预融合/LLM阶段必须满足 checkpoint 原生窗口；不自行RoPE外推 |

不得把 A 的窗口强行外推至32K，不得在 A/C 等待不存在的 `</think>`，不得在看到 FINAL ANSWER 后自动停止；答案标记是解析边界，不是本实验额外 stopping criterion。

主长度字段：
- 所有模型都有 `output_tokens=len(generated_token_ids)`，包含所有真正生成的结构 token。
- B 原生 `reasoning_tokens` 按 tokenizer 的 think 边界分区，保留此前的三分区和删失标签。
- A/C 不是原生 Thinking 模型。使用最后一条合法 FINAL ANSWER 行起点前的文本作为 `explicit_explanation`，token 边界通过增量解码/offset 映射到原始生成ID；不得重分词后把结果冒充原生成token计数。记录首个/最后一个 marker 位置、marker数量、marker后是否继续解释。
- A/C 未给标记：`explicit_explanation_tokens=null`，保留完整输出长度、parse_failure，不将整条输出自动称为原生 reasoning tokens。

A 的主要长度终点是 total output；B 的主要终点是 native reasoning，另报 total output；C 仅描述。不汇总三轨道 raw token 均值/触顶率为一个“总体压缩效应”。

种子：`s=(1234+int(sha256(uid)[:8],16)+1000003*j)%(2**31-1)`，j=0,1,2。同模型同题同重复的 Full 与各压缩条件使用相同起始随机流；不同 tokenizer 的相同种子不意味着相同语义采样。

采样必须参数化。CPU参考使用逐请求PCG64；生产预生成足够的 uniform 数组，按每请求已经生成的步数索引，不受槽位迁移影响。GPU执行同样的温度/top-k/top-p/逆CDF规则。并列logit按token_id稳定排序，比较边界统一，禁止每步把整个logits复制回CPU。重启保存生成前缀与已消耗随机步数。

## 6. 方法实现规范

### 6.1 Full
关闭全部压缩，使用完整原生视觉表示。VisionZip 的“设置为576但仍运行CLS/聚合”不是 Full。所有 native reference 的 Full 路径必须对齐。

### 6.2 FastV（A）
使用作者静态 KV 版本，选择在前2个LLM层计算完后应用；实际实现中先以张量流核实 `K/layer_idx` 的0/1基准，不能只改参数名称。根据作者最后查询的头聚合注意力选择视觉token。keep 50%=288，25%=144。

标题标 `FastV-static/native-LLaVA`。它不同于原始无缓存逐步重选的动态版本。缓存、position_ids、前层是否保留全部KV均以固定作者静态代码为reference；不能强迫所有层KV同长来配合旧Qwen decoder。保存每层实际视觉/文本KV数。

### 6.3 PriorTR（A与B）
第3层后剪枝，评分为归一化 P log(P/Q)。A 显式指定 keep_tokens=288/144，防止作者setup默认192优先覆盖 keep_ratio。A 使用 LLaVA1.5 的作者 question query scope（含其原生 null/newline 先验逻辑）；B 使用作者 Qwen3 vision_end 先验路径与 question scope。不得把 Qwen 的特殊 token ID 套到 LLaVA。

在准备阶段记录 P/Q 的真实查询索引、特殊token解码、聚合维度、epsilon和层索引。对同一张量调用固定作者函数与本地优化函数，比较分数和top-k。不是自行把查询改成“更干净的问题词”后仍称复现。若作者分支在图后插入newline/null而普通模板没有，准备阶段将同一分隔序列纳入该轨道全部条件（含Full）的canonical prompt，并冻结hash；不能只给PriorTR额外文本。

### 6.4 AgilePruner（A）
原生 CLIP 倒数第二层 CLS attention，而非旧版Qwen mean-incoming替代。保持作者有效秩、余弦距离、原attention rank、自适应阈值、预算不足补齐及投影阶段。LLaVA参考 `erank_avg=90`、`tau_max=.25`，不沿用813.9009、不用正式题标签重新校准。K=288/144，LLM前选择。CLS用于评分不自动作为额外LLM token；实际输出K需断言。

### 6.5 VisionZip（A；新增）
使用免训练原始权重版本，不使用额外微调权重。CLIP CLS注意力保留dominant patches，再将其余内容按作者算法聚合为contextual tokens。必须保留作者的特征阶段、metric、排序和 `target_hidden + aggregated_hidden` 规则；不能改写成另一种平均后仍称同方法。

当前核对的单图tensor路径包含1个CLS，并在输入LLM时保留。严格按实际输入数量定义预算：`D+C+1=K`；维持论文常用54:10的非CLS配额比例，`C=floor((K-1)*10/64+0.5)`，`D=K-1-C`。

| K | dominant D | contextual C | CLS |
|---:|---:|---:|---:|
| 288 | 242 | 45 | 1 |
| 144 | 121 | 22 | 1 |

这些是本实验同预算配置，不宣称是论文表格的原始budget。冻结commit后再次探测输出长度；与表不符就停止，不能偷偷裁掉CLS。输入统一走已验证tensor路径；原始视觉encoder的Python list分支与tensor分支不相同，批处理不能走错。记录每个聚合token的来源成员、系数和代表项（至少机制子集），不要伪造为“选中原token索引”。

### 6.6 LLaVA-Mini（C；新增）
只运行公开8B checkpoint自身学到的视觉压缩和模态预融合。记录 learned queries、压缩前后 token 数、预融合层与真实文本特征路径。两个模块不可省略其一：把原图信息注入文本的预融合也必须计入成本。

不任意将查询数切到144/288，不用未经训练的新增queries，不能只接压缩视觉embedding而丢掉prefused text。在未核实配置前，JSON保留“读取checkpoint后冻结”，`prepare`必须完成该字段才能运行。这是环境准备约束，不让agent自由选择算法。

## 7. 正确性验收与高吞吐认证

按 GPU_EXECUTION_R3.md 执行。最低要求：
1. 每模型原生Full、图像特征、位置、attention mask、各层KV、logits验证。
2. 每个免训练方法强制identity路径（不做压缩）与Full一致；真实K<N路径对作者参考，不只测绕过压缩的K=N。
3. B1/B>1、左padding、不同长度混批、每层不同KV长度、EOS回收、页边界、槽位重用、prefix COW、断点恢复均测试。
4. VisionZip合并表/输出特征/CLS数量，Mini prefusion文本+视觉特征均验收，不只比较top-k索引。
5. 長前缀 teacher forcing 与自然生成对照并行进行。通过短prefill不代表长轨迹等价。
6. 所有任务、方法、比例、采样参数与输出上限逐行记录实际收到的值。训练、量化、重复惩罚必须关闭。

kernel、编译开关、sampler、page布局、依赖或方法源SHA变化都重新验收。候选高吞吐后端失败不允许放松阈值；切回通过验收的reference后端，并在报告记录未达到性能目标。

## 8. 主实验与固定对照

### 8.1 主实验

| 轨道 | 题数 | 条件 | 重复 | 行数 |
|---|---:|---:|---:|---:|
| A LLaVA7B | 500 | 9 | 3 | 13500 |
| B Qwen8B | 500 | 3 | 3 | 4500 |
| C Mini8B | 500 | 1 | 3 | 1500 |
| 合计 | | | | 19500 |

每重复6500行。三重复都事先固定。生产按一个模型完成其已注册任务再卸载模型；模型内按repeat、task、image hash调度，条件顺序按题序循环位移。允许为了batch选择同长度待处理请求，但不能根据答案或预测正确率改变调度/丢弃任务。每条请求独立随机流。

### 8.2 对照（2400行）

P100：每任务按 `sha256('r3-panel|'+uid)` 最小20题。B100：Doc/Info各按 `sha256('r3-bridge|'+uid)` 最小50题。

| 对照 | 模型/面板 | 新条件 | 三重复新增行 |
|---|---|---|---:|
| 小规模桥接 | Qwen4B / B100 | Full、原生PriorTR25 | 600 |
| 随机选择 | LLaVA7B / P100 | Random25，第2层后，同FastV cache/position策略 | 300 |
| 保留率曲线 | LLaVA7B / P100 | FastV75(432)、FastV37.5(216) | 600 |
| 温度 | Qwen8B / B100 | Full、PriorTR50/25，T=.6 | 900 |

Qwen4B不再跑整个五任务方法矩阵；与Qwen8B在B100的相同条件配对分析。Random mask使用 `sha256('r3-mask|'+uid)` 独立于生成seed，同题三重复mask不变。曲线复用Full/50/25得到五档。温度对照只改temperature，不改后端/模型/图像处理。

主实验+对照21900行。预算允许的主要优先顺序为A→B→C→上述对照；执行全计划时这不是可按结果任意删减的列表。

## 9. 机制实验（保持上一轮目的，改成原生PriorTR）

模型固定Qwen8B，base为**PriorTR25第3层后**，不再使用FastV-Qwen移植版。

M60从正式Doc/Info各30题产生：在读模型输出前，按 `sha256('r3-mechanism|'+uid)` 扫描，人工核验原图与参考。两名标注员标注V（必要数值/实体区域）、B（年份/群体/图例/行列标题/关系区域），冻结一致标注。两人身份可为执行agent+人工复核，但必须标明并保留各自判断，不伪称两个独立人类。

映射到真实processor的视觉token排列，不能假定简单row-major。可行性要求 |V∪B|≤K，S中非证据token足够移除，未选中的非证据token足够构造随机对照。已经覆盖全部证据的题保留作为零干预；不要按是否膨胀挑样本。60题不足时记blocked，不把不足部分随意替换为已观察到异常的题。

四新增arm：
- ValueRescue：补入V\S，从S\(V∪B)按原PriorTR分数升序移除同数量，分数同则索引升序。
- ValueRandom：使用与ValueRescue完全相同的移除集合，补入等量未选中非证据token。
- BindingRescue：补入(V∪B)\S，按相同规则移除。
- BindingRandom：使用BindingRescue移除集合，补入等量随机非证据token。

随机补入种子：`sha256('r3-rescue|'+uid+'|'+arm)`截取前16位转整数；三重复不改mask。K始终不变。每个arm从原始输入重做prefill并在同一层应用mask；不能在异常轨迹缓存里随手补KV。复用主实验Full和PriorTR25基线。4×60×3=720行。

停止实验：同一M60、模型和base，思考预算256/512/1024；若自然think已闭合，引用基线。否则保留真实生成前缀至预算，重建同一压缩状态，强制追加tokenizer编码的 `</think>\n\nFINAL ANSWER: `，再生成最多128个答案token。强制token单独计，答案RNG为原seed+9000007取模。3×60×3=540行。不是直接截断后无答案计零分。

零干预引用不重抽样，保存reused_from_job_key；行数不减少但物理生成可以减少。完整计划23160研究记录；校准/门槛/性能回放另计，不能隐藏为“零成本”。

## 10. 评分与结果文件

保持R2已经确定的V3解析修复：取最后一条合法FINAL ANSWER行；fallback仅完整boxed；不删除末尾句点/货币符号/百分号/单位，不随意取最后一个数字。原始最终文本、prediction_v3、legacy_v2影子结果分别保存。数字语义字段与ANLS分开；不能用ANLS变好定义“推理有效”。

单测必须包含：`11 a.m.` vs `11 a.m.` 得1；`3 Million Users` vs `3 million` 得.6；`34 Million` vs `3 million` 得.9（后两者是字符串指标特性，不修成别的分数）。

最少每行：protocol/job_key/attempt_id；task/uid/image_sha；model/revision/processor/template；method版本与code SHA；generation参数实际值；seed/repeat；backend/kernel/dtype；expanded prompt IDs或可无损恢复的embedding引用及哈希；原视觉数/实际压缩数/每层序列数/每层KV数/位置策略；原始输出ID/文本；总长度/解释长度/native reasoning长度及边界来源；stop_reason/cap/parse_fail；原始答案、V3/legacy预测、任务原生分数；R10；视觉/选择/prefill/decode时间字段；峰值显存；缓存复用；错误/复用来源。

对于VisionZip记录member/weight或其可重建哈希；对于Mini记录所有prefusion相关计数。merged tokens不能只存伪造的原图selected_indices。

错误基础设施任务保留行并标记error，不当模型答错；模型自然不能回答是模型结果，不能重试挑更短/更正确答案。job_key包含协议、模型、uid、repeat、方法、预算、温度、backend profile hash；最终唯一性依研究条件检查，避免后端重试成为额外独立样本。

## 11. 统计与可检验结论

A以完整500题为主，L=output_tokens；B以native reasoning tokens为主，另报output_tokens；C仅质量/长度/重复/成本描述。所有length上限都视为观测预算，不解释为未截断自然完成时间。

每模型任务分层报告：配对均值差、中位差、P90/P95/P99、尾部10%成本；over2=(Lc>2Lf)且Lf>0；severe_over2=over2且ΔL≥256；new_catastrophic=(Lc≥4096且Lf<512)仅用于B，A改用模型内相对窗口指标而非永不可能达到4096的条件；重复R10、触顶率、边界/解析异常、质量。

R10=1−不同10gram数/全部10gram数；lowercase后按 `\w+|[^\w\s]` 切分词与标点，不足10单元设0；严重重复阈值>.5。字面重复标记不等于全部多余token，也不能覆盖换说法的语义循环。

A primary：四种25%方法各自对Full的mean Δoutput、severe_over2概率，共8终点一族。B primary：PriorTR25的mean Δreasoning、severe_over2概率，共2终点另一族。其余50%、温度、随机、模型规模、机制分析为预注册次要分析。不要把severe_over2概率叫“Full同自身的随机风险差”。

先每题平均三重复配对量，再任务内按图整簇bootstrap20000次（seed20261022），保留全部方法/seed，报告普通95%与对应族Bonferroni区间；不跨模型把token/质量分数相加。五任务等权的模型内风险汇总可以作为补充，质量仍按不同指标分别报告。

固定解释边界：LLaVA显式解释不膨胀不等于Qwen原生长思考不会膨胀；Mini更短不等于压缩本身减少推理，因为训练/骨干不同。Full能力低的Doc/Info任务报告floor effect，主总体不能事后只留Full正确题。

人工轨迹分析至少每个可配对方法20例，按预设长度/质量分层随机取，隐藏方法与分数后标注；包含正常短轨迹、增长正确、增长错误、循环、有候选但不提交、仅格式变化。给原文和视觉证据，不能只有自动标签。

## 12. 成本测量与公平性

生产吞吐可复用视觉特征/精确prompt前缀，计时报告必须分为：
- 实验生产吞吐（warm/shared cache）：有效完成请求/秒、生成token/秒、总墙钟、GPU和CPU开销，包含所有长尾。
- 冷请求成本（cold）：不跨条件/seed复用视觉或prefix；同一模型载入后正常预热kernel，再对新请求计图像处理、编码、selector/merge/prefusion、LLM prefill、decode、TTFT、总延迟和显存。
- 固定前缀/固定续写的回放kernel对比，与自然终止后的真实端到端分开，不为凑长度把异常/随机token回放伪称真实回答。

性能面板P100每任务取hash最小5题，共25题；每轨道各条件取其seed0完整真实轨迹，native reference和通过认证的production各预热1次、测量3次完整teacher-forced回放。累计同步CUDA events和墙钟；不产生新的研究答案。Mini成本必须包括预融合，不能只计1个视觉token之后的LLM。

所有数字报告实际测量值，不用官方4000 FP4 TOPS推算本轮BF16吞吐。不把GPU利用率100%等同于高效。

## 13. 断点、顺序与agent接口

先准备/门槛/吞吐调优，冻结 `resolved_spec.json` 与 `engine_profile.json`，才生成主结果。已成功job只读；允许从原token/RNG/cache一致状态恢复。进程崩溃可从原prompt按完整前缀teacher-force重建缓存，记录重算成本；不得重新随机采一条来替代原前缀。改变backend续跑必须先过该job前缀一致性门槛，否则新attempt标识并保留，不按答案选择。

本包实际提供：`preflight.py`（GPU/本地配置检查，不加载权重）、`validate_plan.py`（预算与行数校验）、`round3_spec.json`、本规约和GPU规约。没有伪装已经实现模型推理runner。

执行agent必须实现以下接口，命令名是开发合同：

```bash
python -m r3.prepare --spec round3_spec.json --root "$COMPRESS_ROOT"
python -m r3.validate_native --spec runs/visual-compression-r3/configs/resolved_spec.json
python -m r3.tune_engine --spec runs/visual-compression-r3/configs/resolved_spec.json --calibration-only
python -m r3.validate_engine --spec runs/visual-compression-r3/configs/resolved_spec.json
python -m r3.build_jobs --spec runs/visual-compression-r3/configs/resolved_spec.json --dry-run
python -m r3.run --stage main --spec runs/visual-compression-r3/configs/resolved_spec.json
python -m r3.run --stage controls --spec runs/visual-compression-r3/configs/resolved_spec.json
python -m r3.run --stage rescue --spec runs/visual-compression-r3/configs/resolved_spec.json
python -m r3.run --stage stop --spec runs/visual-compression-r3/configs/resolved_spec.json
python -m r3.analyze --spec runs/visual-compression-r3/configs/resolved_spec.json
```

`analyze`不得触发补跑；`run`检查native/engine gate与spec/code SHA吻合；stage main只19500行，controls2400，rescue720，stop540。缺项不能暗中从分母删掉。Mini字段/来源锁未解析时`build_jobs --production`必须非零退出。

交付：来源/环境锁、500/100清单、预期jobs、所有原始轨迹、每层预算审计、选择/合并机制数据、natural后端验证、GPU调优完整候选日志、cold/warm计时、各任务配对结果、机制干预与至少20个有具体原文的代表案例。报告先回答同模型压缩效应，再谈跨架构覆盖。
