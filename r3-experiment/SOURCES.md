# 证据与版本登记

核查日期：2026-10-22。以下为论文、作者仓库、官方模型配置或硬件/软件官方文档。R3的样本数、2048输出预算、VisionZip同预算分配、调优搜索和验收阈值是本方案设计，不是论文原始设定。

## 原方案

基于本对话已生成的 `visual_token_audit/AGENT_RUNBOOK.md`、`round2_spec.json` 修订；保留五任务、三重复、评分修复与两个机制实验的研究目的。旧R2的Qwen统一移植、统一32K、B1生产限制被本轮显式替代。

## 方法与模型

- **S1 FastV**：作者仓库 https://github.com/pkunlp-icler/FastV ，论文 https://arxiv.org/html/2403.06764v3 。ECCV2024；README “Support KV Cache”明确静态缓存版与原动态重选差异。原生LLaVA7B示例。正式准备需冻结commit。
- **S2 PriorTR**：作者仓库 https://github.com/CodeChildCZJ/PriorTR ，论文 https://arxiv.org/html/2606.24156v1 。作者同时支持LLaVA和Qwen3-VL。旧实验已固定commit e442a4c14af1b333067a6b8310c9111cad922033。
- **S3 PriorTR LLaVA配置与环境**：https://raw.githubusercontent.com/CodeChildCZJ/PriorTR/main/image/LLaVA/README.md ，https://raw.githubusercontent.com/CodeChildCZJ/PriorTR/main/image/LLaVA/llava/vtr/config.py 。keep_tokens优先、v1.5默认question聚合；README含SM120/CUDA12.8说明。main仅用于核查，正式按固定commit重核这些行为，不以网页main替代冻结源文件。
- **S4 LLaVA1.5-7B官方配置**：https://huggingface.co/liuhaotian/llava-v1.5-7b/raw/main/config.json 。4096窗口；CLIP336 patch14；pad；32层、32Q/KV头、hidden4096；存储dtype float16。其BF16 KV/token计算是依据配置的算术推导，不是测速。
- **S5 AgilePruner**：https://github.com/cvsp-lab/AgilePruner ，https://arxiv.org/html/2603.01236v1 。ICLR2026；原生LLaVA，作者模型链接7B。固定commit6829283ded10bdec58361512a32676c71bbd17d3；原生选择器在llava/model/llava_arch.py，CLIP来源在multimodal_encoder/clip_encoder.py。不能将Qwen mean-incoming替代称原生CLS。
- **S6 VisionZip**：https://github.com/JIA-Lab-research/VisionZip ，https://arxiv.org/html/2412.04467v2 。CVPR2025；免训练dominant选择+contextual聚合、LLM前压缩、LLaVA1.5支持。正式冻结代码SHA。
- **S7 VisionZip实际计数与聚合**：https://raw.githubusercontent.com/JIA-Lab-research/VisionZip/main/visionzip/clip_encoder.py ，https://raw.githubusercontent.com/JIA-Lab-research/VisionZip/main/visionzip/llava_arch.py 。tensor单图路径保留CLS；D+1个dominant含CLS再拼C个contextual；聚合为target_hidden+aggregated_hidden；LLaVA1.5 flat路径未删除CLS。R3要求再次通过运行张量数核验。
- **S8 LLaVA-Mini论文**：https://proceedings.iclr.cc/paper_files/paper/2025/hash/8400a2ec9ba85bfdb41eb2a584c0876d-Abstract-Conference.html ，https://arxiv.org/html/2501.03895v1 。ICLR2025；学习式查询压缩与模态预融合，预融合把图像内容注入文本。论文比较不是现成的同权重压缩开关。
- **S9 LLaVA-Mini公开checkpoint及代码**：https://huggingface.co/ICTNLP/llava-mini-llama-3.1-8b ，https://github.com/ictnlp/LLaVA-Mini 。官方示例checkpoint为Llama3.1-8B，模板llava_llama_3_1。本次网页获取了model card与代码说明，但未可靠获取该checkpoint完整config.json，因此不伪造其query数、预融合层数、分辨率和上下文细节；prepare必须读取本地config/loader并冻结。论文1-token宣传不替代每次前向的实际token数量。
- **S10 Qwen8B官方配置**：https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking/raw/main/config.json ，https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking 。36层、8KV头、head_dim128；BF16；原生Thinking模板。32768总输出为延续研究设定，不是模型最大上下文。

## GPU与后端

- **S11 NVIDIA RTX PRO6000 Workstation**：https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/ 。96GB GDDR7；1792GB/s；最高600W，仅对应该edition，其他edition以实机信息为准。
- **S12 NVIDIA Compute Capability**：https://developer.nvidia.com/cuda/gpus 。RTX PRO6000 Blackwell Workstation/Server/Max-Q列为12.0；B200为10.0，RTX6000Ada为8.9。
- **S13 PyTorch Blackwell支持**：https://pytorch.org/blog/pytorch-2-7/ 。2.7引入CUDA12.8构建支持Blackwell。R3不要求退回2.7，也不拿旧项目torch2.1作为新卡默认。
- **S14 FlashInfer安装**：https://docs.flashinfer.ai/installation.html 。本次页面为0.7.2，列CUDA12.9/13.0以及预览版本；必须匹配实际torch/driver与SM120 kernel。
- **S15 FlashInfer attention API**：https://docs.flashinfer.ai/api/attention.html 。分页prefill/decode、CUDA Graph等构件。API存在不证明研究方法已集成或自然长生成等价。
- **S16 vLLM multimodal输入**：https://docs.vllm.ai/en/stable/features/multimodal_inputs/ 。原生模型/多模态embedding输入支持，不等于自动保留中间层剪枝语义；R3没有给出未经验证的vllm启动即支持五方法的承诺。

## 数据来源（沿用R2范围）

- **S17 ChartQA**：https://github.com/vis-nlp/ChartQA ，正式使用test_human而非test_augmented。
- **S18 GQA**：https://cs.stanford.edu/people/dorarad/gqa/ ，balanced validation。
- **S19 MathVista**：https://huggingface.co/datasets/AI4Math/MathVista ，testmini；确定性答案提取与官方LLM辅助提取要区分。

## 本次验证范围

已完成：原方案读取、公开论文/官方仓库与模型配置核查、R3规约/配置编制、CPU行数与预算校验、辅助脚本单元测试。

未完成：目标GPU安装、实际权重下载与冻结、Mini config解析、native方法的GPU实现与一致性检查、生产吞吐调优、自然轨迹认证、正式生成。交付包不包含模型权重或伪造的GPU性能报告。
