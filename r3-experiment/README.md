# R3：原生骨干 + 五种视觉压缩机制 + RTX PRO6000

这是R3任务书、作者原生适配与正式推理运行器。

2026-10-24：用户缩减长程验收并授权正式推理，同时强调GPU执行规约。现已实现单份模型、原生完整prefill、最多四请求批量解码和连续补槽，执行政策 `user_reduced_native_batch`。冻结500题×三seed，main 19500 + controls 2400，共21900次；不解析答案、不评分。六条件短数值/KV检查已通过；最后资源与slot检查全部通过才自动启动。使用dense padded KV，尚未启用分页或CUDA Graph；不声称原完整自然等价认证通过。M60双人标注阶段单独保留。实际启动以服务器launch回执为准。

`r3.start_formal` 注册并启动唯一流水线；`r3.formal_lifecycle` 自动衔接各条件、记录GPU使用并在生成完成后导出全部原始结果。`local_collect_formal.py` 下载导出包并逐文件核验。模型、图像数据集不属于结果包；原始回答、token、输入清单、隔离标签、配置、绑定、失败及运行日志属于结果包。

## 阅读与执行顺序

先读 `CHANGELOG.md` 和 `AGENT_RUNBOOK_R3.md`，再读 `GPU_EXECUTION_R3.md`；机器配置为 `round3_spec.json`，依据为 `SOURCES.md`。

已有可运行辅助脚本：

```bash
python validate_plan.py --spec round3_spec.json --out plan_validation.local.json
python -m unittest -v test_helpers.py
# 以下在真正的目标GPU机器执行，路径换成本地checkpoint目录：
python preflight.py --out environment/preflight.json \
  --config llava7=/models/llava-v1.5-7b/config.json \
  --config qwen8=/models/Qwen3-VL-8B-Thinking/config.json \
  --config mini8=/models/llava-mini-llama-3.1-8b/config.json
```

`validate_plan`只检查配置/预算/研究行数，`preflight`不下载模型、不安装包、不改变功耗，不认证算法实现。`plan_validation.json`是本次CPU检查记录，明确production_ready=false。

生产runner接口与门槛详见任务书第13节；必须由执行agent开发并在目标机器验证。`revision=null`等字段在prepare阶段解析并冻结，没解析不能正式生成。

## 核心配置

A：LLaVA1.5-7B，Full+FastV/PriorTR/AgilePruner/VisionZip各两档，13500行。
B：Qwen3-VL-8B-Thinking，仅官方原生PriorTR两档+Full，4500行。
C：LLaVA-Mini官方Llama3.1-8B原生配置，1500行，不做虚假同权重Full比较。

主实验19500；固定对照2400；机制救援720；强制交答案540；总23160研究记录。

单卡执行重点是一份模型权重、多请求连续批处理、分页缓存、按KV预算准入、GPU采样与CUDA Graph；不是常驻三个模型，也不是用量化/缩短生成提高表面吞吐。正式跑分和冷请求算法耗时分开报告。
