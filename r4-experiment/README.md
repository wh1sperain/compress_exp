# R4 扩展验证（当前默认方案）

[具体安排](PLAN.md)｜[用户V1.0原文](SPEC_V1_20261027.md)｜[协议](extension_protocol.json)。旧机制实验在[历史稿](PLAN_MECHANISMS_20261026.md)，`r4`旧入口不是本轮调度入口。

两模型、六任务、26条件，正式46,800次；先导3,840次；鲁棒性按原文所有维度计算是**3,600次**（原文1,800为算术笔误）。不自动开跑。GPU验收、真实六任务清单和人工盲审未完成。

## 本地准备

```powershell
cd r4-experiment
python -m unittest discover -s tests -v
# source-map使用固定revision/SHA、本地官方导出或HTTPS URL、字段映射、原图目录。
python -m extension.import_sources --source-map /data/six-source-map.json --out .local-import
python -m extension.prepare_data --candidates .local-import/candidates.jsonl --source-lock .local-import/source_lock.json --out .local-manifests
python -m extension.plan_jobs --inputs .local-manifests/pilot_inputs.jsonl --phase precheck --out .local-precheck-jobs.jsonl
python -m extension.plan_jobs --inputs .local-manifests/pilot_inputs.jsonl --phase determinism --out .local-determinism-jobs.jsonl
```

源数据适配使用显式字段映射，`source-map`格式示例见`SOURCE_MAP_FORMAT.md`。不猜测MMMU大学科、MathVerse原题ID或AI2D选项索引。数据版本、图像SHA和严格分层不够时拒绝冻结。上传仓库不包含数据/图片/答案。

## 验证与生成

```bash
# 先用独立环境加载锁定作者源代码；官方BF16权重/Tokenizer文件必须有SHA清单。
python -m r4.fetch_sources --help
python -m extension.run_inference --inputs .local-manifests/pilot_inputs.jsonl --jobs .local-precheck-jobs.jsonl \
 --model qwen8 --condition Full --checkpoint /models/qwen8 --weights-lock /data/weights.json \
 --environment-lock /data/environment.json --author-root vendor/PriorTR --diagnostic --out .local-precheck/qwen8
python -m extension.run_inference --collect .local-precheck --out .local-precheck.jsonl
python -m extension.evaluate_answers --records .local-precheck.jsonl --inputs .local-manifests/pilot_inputs.jsonl \
 --labels .local-manifests/pilot_labels.jsonl --official .local-official/runtime_evaluators.json --out .local-precheck-scoring
# 审阅所有循环预警与评分待审项，再生成两模型决策。
python -m extension.validate_backends --records .local-precheck-scoring/scored.jsonl --manifest .local-manifests/pilot_inputs.jsonl \
 --confirmations /data/human-loop-review.json --out .local-decisions.json
python -m extension.plan_jobs --inputs .local-manifests/formal_inputs.jsonl --phase formal --decisions .local-decisions.json --out .local-formal-jobs.jsonl
```

若触发采样替代，另提供绑定决策SHA的`--fallback-plan`，核心五条件三种子，其余单种子，新增9,000条/触发模型。不给回执则阻止调度，不悄悄修改主设置。

正式运行命令与上面相同但移除`--diagnostic`，加`--gate`与`--decisions`，使用正式inputs/jobs；每次指定一个模型/条件，B1。门槛`identity`由实际环境、模型、源代码、协议SHA构成；`conditions[model:condition]`内必须含`extension.validate_backends.CHECKS`全部PASS、实测容差和SHA绑定证据。**代码不提供假PASS模板**。无剪枝诊断用具体方法条件加`--disabled`，Hi-Lo显式绕过选择器与融合；不得将该结果当正常压缩结果。

目前实际长前缀logits捕获可复用`r4.diagnose.probe`/`r4.compare`函数，但必须由新两模型环境产生证据并按新门槛绑定，旧R3回执无效。方法复测使用独立repeat=0/1/2输出目录和相同seed，`--determinism`比较真实位置/序列而非只比较长度。

## 提取、统计、报告

```bash
python -m extension.fetch_evaluators --out .local-official
python -m extension.evaluate_answers --records .local-formal.jsonl --inputs .local-manifests/formal_inputs.jsonl \
 --labels .local-manifests/formal_labels.jsonl --official .local-official/runtime_evaluators.json --out .local-scoring
# 完成盲审后，加--reviewed再次输出到新的目录；永不覆盖原始结果。
python -m extension.aggregate_metrics --records .local-scoring-reviewed/scored.jsonl --inputs .local-manifests/formal_inputs.jsonl --jobs .local-formal-jobs.jsonl --out .local-metrics.json
python -m extension.analyze_trajectories --records .local-scoring-reviewed/scored.jsonl --inputs .local-manifests/formal_inputs.jsonl --out .local-cases.json
python -m extension.generate_report --metrics .local-metrics.json --records .local-scoring-reviewed/scored.jsonl --out reports/expansion-v1
```

推理时直接根据原始ID分区；`extension.parse_outputs`也能离线重建。MathVista/MMMU使用锁定函数；MathVerse保留官方两步来源，等待盲化人工/独立评审回执，不自动调用付费API。盲审回执映射键为blind_id，含sample_id/raw_sha256/blind=true/predicted_answer/correct/judge_type/judge_version/judge_reason；提取队列无gold和方法信息。未解决评分保持null，无答案正常运行按错计，进程失败不当答案。

`results`以每job原始JSON+trace+binding+success-SHA作为不可变源，收集器校验后导出JSONL并附失败清单。前缀恢复不覆盖更长已存前缀；原失败需明确新恢复记录，不自动重抽。图表只有输入真实结果才生成，仓库里的测试夹具不代表实验发现。
