# R2 → R3 的明确变更

| 项目 | R2 | R3 |
|---|---|---|
| 主方法骨干 | Qwen4B/8B上的统一实现 | 作者原生LLaVA1.5-7B上的四方法；不做LLaMA→Qwen移植 |
| 方法数 | FastV/PriorTR/Agile | 增加VisionZip和LLaVA-Mini，共五种 |
| 学习式压缩 | 未覆盖 | Mini官方Llama3.1-8B，单列描述性轨道，不伪造Full |
| Qwen使用 | 三方法都移植 | 只使用PriorTR原生Qwen3-VL路线作为长思考桥接 |
| 模型规模 | 4B/8B全量矩阵 | LLaVA7B与Mini8B主覆盖；Qwen8B桥接，4B仅100题小桥接 |
| 窗口/输出 | 统一32K输出 | LLaVA原生4096窗口、2048输出；Qwen32768输出；Mini2048且原生窗口校验 |
| 长度指标 | 全部按think分区 | A显式解释+总生成；B原生reasoning；C描述，不混合tokenizer |
| Agile | mean incoming Qwen适配，校准erank | 原生CLIP CLS与作者erank=90 |
| VisionZip计数 | 不适用 | D+C+CLS=288/144；实际入LLM张量门槛 |
| KV规则 | 所有层统一compact | 保留每方法原生层级KV/位置语义，后端支持差异 |
| 生产解码 | HF-SDPA batch1 | B1参考；通过自然长轨迹门槛后连续批+分页+GPU采样+graphs |
| 硬件 | 未明确 | 单RTX PRO6000 Blackwell96GB；实机识别edition/SM120 |
| 并发策略 | 固定B1 | 真实KV预算动态准入，EOS补槽，校准makespan选batch；可认证双队列 |
| 视觉与prefix缓存 | 限制复用 | 跑分可精确复用，冷请求端到端单测不复用 |
| 机制base | Qwen FastV25 | Qwen作者原生PriorTR25，第3层后 |
| 主实验行数 | 16500 | 19500 |
| 全计划研究行数 | 20160 | 23160（不含正确性/性能认证） |

沿用：五任务每100正式+20校准、三seed、无重复惩罚、无按异常提前停止、V3评分修复、全配对总体优先、触顶不当自然长度、视觉绑定救援与强制交答案两个机制实验。

需要重新计算而非继承：模型输入长度、视觉预算与合并关系、native cache策略、Mini配置、硬件/软件锁、生产engine_profile。旧Qwen移植分数不拼入新LLaVA Full表。
