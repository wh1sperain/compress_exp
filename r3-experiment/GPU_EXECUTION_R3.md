# RTX PRO 6000 执行与吞吐规约

2026-10-24最新执行状态：用户强调本规约后，B1正式启动被搁置。新后端 `native-padded-batch` 保留作者逐题完整prefill、选择与位置，在一份权重上进行最多四请求批解码，EOS/上限立即释放请求，下一轮补槽。LLaVA/Qwen逐层独立KV+SDPA，Mini调用原生Llama批forward；使用dense padding，分页/Graph仍未实现。实际KV准入按原完整剩余输出预算和3倍scratch保守预留，留8GiB及总显存10%边界，满cap合成内存检查与自然补槽检查通过后冻结。六条件短数值已PASS，不等于原完整认证；短测试计时含图像prefill且Mini首次网络等待混入，不当作正式吞吐结论。正式阶段不解析或评分，最终完整原始结果下载并逐文件校验。

本方案按单张 RTX PRO 6000 Blackwell 96GB 设计。NVIDIA Workstation Edition 公布96GB GDDR7、1792GB/s、最高600W；不同Max-Q/Server版本功耗和带宽不应混用。计算能力为12.0，不能因为同叫Blackwell就默认支持B200的SM100专用kernel。[S11–S14]

本文所有batch、内存、页大小都是候选起点，不是实测结论。最终值必须由本机校准工作负载选择，保存在engine_profile.json。没有在本会话的环境上测试过该显卡。

## 1. 开机检查：不更改设备权限或功耗

在目标机器运行本包：

```bash
python preflight.py --out environment/preflight.json \
  --config llava7=/path/to/llava-v1.5-7b/config.json \
  --config qwen8=/path/to/Qwen3-VL-8B-Thinking/config.json \
  --config mini8=/path/to/llava-mini-llama-3.1-8b/config.json
```

采集：GPU完整名称/UUID、实际可见总显存、compute capability、driver、torch/CUDA、扩展版本、当前功率限制、温度、时钟、MIG状态、其他进程、CPU核数、系统内存及可用磁盘。存在其他GPU任务、GPU只有48GB、MIG分片或设备不是SM120时暂停性能认证，不能仍声称按96GB整卡运行。不要擅自杀别人的进程、关闭MIG、提高power limit。

脚本只做本地配置与硬件/小型BF16计算检查；不能代替模型、FlashInfer、选中索引和长序列门槛。无GPU时返回blocked，不能打印虚假的性能通过。

## 2. 环境约束

推荐Linux、独立Python3.11环境。PyTorch 2.7开始有CUDA12.8 Blackwell wheel，这是兼容性起点而不是要求使用过时版本。先选一个明确支持SM120的稳定torch/CUDA组合，固定wheel与版本，再安装每个原生repo的非torch依赖。[S13]

FlashInfer目前官方安装文档列出CUDA12.9/13.0支持；首选与已选torch及driver匹配的稳定CUDA13.0组合，通过算子冒烟后冻结。已有cu128环境可使用明确支持它的固定历史FlashInfer版本，但不能把最新FlashInfer、任意cu128 torch和旧nvcc随意混装。保留安装锁及编译输出；不得运行旧setup脚本把torch降回2.1/cu121。[S14]

必须测试实际BF16 prefill/decode kernel，含MHA(LLaVA32KV头)、GQA(Qwen8KV头)和真实head_dim。FA3的Hopper优化或B200专用kernel不能直接认定适用RTX；优先在支持SM120的SDPA fused路径、FlashInfer分页路径之间按实际测试选择。扩展import成功不等于kernel执行成功。[S15]

native LLaVA旧transformers代码和Qwen现代代码可以分环境；每次只启动一个模型worker，不在同一Python路径覆盖多个llava包。原作者算法代码的兼容性补丁必须进入source_snapshot，禁止patch安装目录后不记录。

预编译可能的head_dim/页布局/graph形状，缓存JIT产物；编译、模型载入和预热时间单独记录，不混入稳态TTFT，也不隐藏在整体任务成本之外。

## 3. 生产架构：一份权重、多个活跃请求

推荐流水线：

```text
CPU多进程读图/resize与tokenize → pinned-memory有界预取队列
    → 图像编码batch/可审计特征缓存
    → 按原生方法执行完整prefill与剪枝/合并/预融合
    → 每层独立KV页表 + 请求独立位置/RNG状态
    → 连续批量decode + GPU sampler + CUDA Graph
    → EOS立即释放槽位；异步写原始token/终止/计分记录
```

一个模型常驻一个GPU进程，TP=1。不是启动16个B1脚本复制16份权重。完成A相关任务后卸载，再B、C；4B桥接单独加载。不会通过同时塞三种模型来提升nvidia-smi中的忙碌百分比。

对合并/预LLM方法可以复用正常decode框架；对FastV/PriorTR必须支持中间层序列变长短及**各层可能不同的KV长度**。库支持基础LLaVA，不等于自动执行研究方法。不能直接 `vllm serve` 后以为它已经使用作者剪枝。

优先复用成熟分页attention kernel，但保留作者原生prefill的算法逻辑。旧methods_qwen3.PagedDecoder假设Qwen归一化/3轴位置以及所有层compact，本轮LLaVA不能直接换model_id就用；Mini的prefused文本也不能丢掉。

## 4. 内存预算：按KV算，不只看模型权重

BF16普通decoder每个上下文token的KV存储量：

`bytes_per_token = 2(K,V) × layers × num_key_value_heads × head_dim × 2(bytes)`。

以核对到的官方config为例：[S4,S10]

| 模型 | 层/KV头/head_dim | KV每token | 一个长请求的示意KV |
|---|---|---:|---:|
| LLaVA1.5-7B | 32 / 32 / 128 | 512 KiB | 4096上下文约2 GiB |
| Qwen3-VL-8B | 36 / 8 / 128 | 144 KiB | 32768生成约4.5 GiB，尚未加prompt |

这是KV张量的理论量，不包含weight、激活、graph、padding和页尾碎片。实际每层prompt长度不同则逐层求和；C的配置通过本地读取，不凭“Llama8B”猜实际缓存结构。

先在不预分配大KV池时测出常驻非KV峰值 M_resident_nonKV_peak，包含模型、vision/prefusion模块、最大prefill临时张量、graph池、GPU feature cache。预算：

`M_kv = floor_to_pages(min(0.90*M_total, M_total - 8GiB) - M_resident_nonKV_peak)`。

原生reference的临时KV在测量时单独剥离/记账，避免既计入nonKV又重复计入KV。不要因reserved≠allocated就把全部差值当可用。生产KV池固定后以预填充压力测试复核。候选池40/48/56GiB；只有小于上式且最大输入压力测试通过的值才可参与调优。GPU特征缓存最多2GiB并计入nonKV峰值，其余特征用CPU pinned缓存/本地磁盘。

不能先承诺“batch64一定能跑32K”。48GiB纯KV池理论上装24条满4096的LLaVA；对Qwen32K输出还要加视觉prompt，只能容纳约个位数至十条全长请求。大量短请求可以高并发，但必须有预算准入。

## 5. 默认准入与长尾处理

默认认证模式不做CPU KV换出：对新请求按其**真实各层prompt长度＋完整剩余输出上限**预留KV容量，页尾碎片、dummy槽位与下一步增长纳入预算。条件间长度不同用真实预算，但不能凭更高正确率预测少留空间。EOS马上归还全部预留和物理页；空槽当步/下一调度步接纳新请求，不等整批最慢的请求。

这能明显减少B1的权重读取与Python开销，又没有因爆显存截断长轨迹的风险。可以设置最大slot为32/64，但实际admitted数受预算限制，报告active而不是configured batch。

有较多Qwen长尾时，可额外测试**可选双队列**性能分支，只有与默认一样通过完整门槛且makespan更低才批准写入最终profile：
- 前段队列每请求运行至自然结束或已生成2048个token；到2048但没结束的请求不是停止结果，而是保存无损KV/原始位置/前缀/RNG索引，进入continuation队列。
- continuation以能容纳全部剩余预算的小批继续到同一个32768上限，禁止重抽前段或强制答案。
- 最多缓存64GiB待续CPU KV且不得超过可用主机内存的1/4，达到阈值立即排空continuation队列；不足主机内存时该候选不适用，不悄悄丢任务。
- 每处理完一个前段调度块（最多32个新请求）至少检查一次待续队列，禁止让长请求无限饥饿。模型卸载前必须完成全部待续请求。
- 只有一次受控换出/恢复；禁止频繁thrashing。监控PCIe耗时，如果抵消吞吐收益则选回默认模式。

该分支是执行调度变化，不是新增科研压缩方法。必须记录stage/swap次数和完整wall time；cold timing不使用换出。CPU cache量化、文本KV删除、只保存最后窗口都不允许。

## 6. batch / kernel 调优：有上限的搜索

只使用100幅校准图与固定reference前缀，不看正式实验质量。先通过native正确性门槛，再只对最终候选做完整natural certification。不能每尝试一个batch就把模型质量当优化目标。

固定性能工作负载：100校准题的seed0、各轨道正式条件；使用reference生成的完整前缀回放（包含自然结束和触顶）。同时另做最大上下文synthetic压力检查，后者只做内存/数值检查，不能当真实回答速度。

分阶段搜索，避免无边界笛卡尔积：
1. 固定prefill token budget4096、page32、graphs关闭，试max_live=4/8/16/24/32/48/64；不满足KV预算的自动标infeasible，不算成功batch。每候选1次预热+3次完整回放，保留所有结果。
2. 只保留makespan最好的两个max_live；在两者上试prefill预算2048/4096/8192（相同设置不重复测）。prefill是完整请求的batch预算，单请求过长时单独执行，不截断其prompt。
3. 只对最佳组合试支持的page16/32/64和CUDA Graph开关；graph捕获仅实际需要的decode形状，从1/2/4/8/16/24/32/48/64取上界桶。dummy槽不写进真实请求KV，也不消耗真实RNG。
4. 最后才测试上述可选双队列分支，默认不做swap。每个候选仍预热1+测3，窗口/预算不改。
5. 最小中位makespan胜出；相差≤2%选更小峰值显存，仍同则更少active slots。不以最高GPU-util或最高瞬时token/s胜出。

不要机械使用chunked prefill：FastV/PriorTR要读完整问题注意力再决定剪枝。未实现并验收等价的分阶段look-ahead，不能在一个prompt的早期chunk就剪枝。R3默认关闭prompt内chunking，只做完整请求的分桶批prefill，仍可连续批decode。

## 7. 减少CPU阻塞与冗余计算

读图与tokenize使用4个worker起步，候选2/4/8，CPU线程不超过物理核心可用量；prefetch最多2倍已批准的prefill batch且有内存上限。图像在CPU处理后pinned memory异步传入。随机流预生成并逐请求存放；每步不要 `.cpu().numpy()` 整个logits，不为日志每token同步GPU。

写盘用独立CPU队列：生成token先缓冲，每128步或请求结束写checkpoint；最终完整记录原子提交，按固定间隔fsync。终止检查与KV释放必须及时，不因异步日志遗漏cap/EOS。测量同步只在明确计时边界，生产不逐算子 `cuda.synchronize()`。

可以复用相同图像、同encoder权重和处理配置的原始CLIP/Qwen视觉特征。保留attention/hidden层需求的一致性；VisionZip与Agile的选择输入可能不同，不能只缓存Full最终特征然后用错误层做评分。缓存key包括encoder/processor/dtype/图像SHA/抽取层。

三seed相同prompt/方法的压缩prefill可复用只读prefix KV，写时复制最后未满页；不同方法不共用压缩KV。key还必须包含model、完整prompt IDs、视觉特征、条件、ratio、native position/cache策略、backend与代码SHA。PriorTR是问题条件选择，不能只按图像key缓存选择结果。Mini的预融合文本是问题依赖的，不能把它当成纯图像特征共用。

这些复用只提高研究批量生成吞吐。冷请求报告重新执行视觉encoder、选择、Mini预融合及prefill，不能把缓存命中时间叫算法端到端加速。

## 8. 数值与自然行为认证

### 8.1 硬不变量
每个方法在B1与选定生产batch中对齐作者reference：选中索引/合并组一致，所有文本存在，attention_mask正确，position_ids符合原生规则，KV无跨请求污染，真实K符合预算，identity=Full。VisionZip合并张量与Mini预融合后文本张量都比较。不能只看最终答案一致。

### 8.2 长前缀teacher forcing
固定相同prefix比较reference和candidate，并记录原始logits、完整softmax及经过实际sampler过滤后的分布。LLaVA在128/1024/2048/3072上下文位置（受真实4096窗口限制）测试；Qwen在1024/4096/8192/16384/32768位置；Mini在合法窗口内同类位置。自然轨迹不足时做独立synthetic压力前缀，明确不用于质量结论。

每个探针至少后续16步，覆盖page±1、slot复用、不同长度混批；KL<.02且TV<.08、无NaN/Inf。argmax不一致数量和tie情况全部记录，不能宣称逐token严格等价。测试原生think-close/EOS的原始概率及top-k资格，不只平均logit误差。

### 8.3 自然轨迹比较
校准集100题、3seed：A用Full+四种25%；B用Full+PriorTR25；C用MiniNative；4B桥接若执行也用Full+PriorTR25。两后端使用同一逐请求uniform流。总4800条比较记录（A/B/C），4B另1200，属于认证开销，不计入23160研究结果。

以图像为bootstrap单位，比较paired mean log1p(output length)、触顶概率、任务分数变化；预设95%区间必须分别完全位于±.10、±.05、±.03内。native Qwen另比较reasoning length/think-end行为；各条件分别验收，不把方法间抵消后平均放行。若CI太宽称“未认证”，不是“无显著差异所以等价”。阈值是工程安全门槛，不是理论同分布证明。

失败时定位reference自相重复、batch GEMM、position/cache/采样差异。不得看完结果放宽门槛。可回退通过验证的HF batch，必要时B1；这时说明性能目标未达到，不承诺高吞吐。采用新诊断方案需新版本，而不是用短轨迹挑一个“看起来一样”的子集。

## 9. 冻结与监控

最终engine_profile包含：GPU UUID/edition、software hashes、backend/kernel、dtype、KV池字节、page layout、per-layer policy、max slots及memory admission、prefill budget、graph形状、CPUworker、缓存与swap策略、sampler SHA、校准makespan和正确性门槛文件hash。

运行中每秒记录NVML GPU-util、memory-util、温度、power、clocks/节流、显存占用；每10秒记录active/pending/continuation数、完成数、token总数、prefill/decode比、CPU队列、cached bytes、swap量。SM利用率/DRAM带宽用短诊断profile查看，不把nvidia-smi的单一util当tensor-core占用。

GPU不满载时按原因改：CPU空队列→预取；大量padding→分桶/ragged；每token主机停顿→GPU sampler/graphs；小decode batch→连续补槽；KV满→更合理准入；PCIe过高→减少swap。不能用生成无用token/增加min_tokens来提高util。

正式数据开始后冻结profile。基础设施OOM保留失败状态，只能按事先批准的恢复策略继续，不能缩短max_new_tokens、调低分辨率或给长样本加重复停止。新profile必须新版本+门槛；研究条件和答案选择不变。
