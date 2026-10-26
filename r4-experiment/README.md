# R4：视觉压缩、视觉事实错误与推理长尾

具体任务书见 [PLAN.md](PLAN.md)，方法版本见 [sources.lock.json](sources.lock.json)。

本次交付为可执行的CPU准备/分析工具、作者原生GPU适配与诊断运行器、干预构造与续写组件。**尚未执行R4 GPU验收或正式实验**；没有捏造通过回执。GPU正式运行会拒绝缺少绑定验收的条件。历史R3的缩减测试不豁免本轮验收。

```powershell
cd r4-experiment
python -m unittest discover -s tests -v
python -m r4.prepare --source /data/r3/main_manifest.jsonl --out .local-panel
# 新确认集：必须排除整个R3的图像集合，另加任何其他已使用图像。
python -m r4.prepare --source /data/new_candidates.jsonl --exclude-r3 /data/all_used_images.jsonl --confirm --methods PriorTR FastV HiLo --out .local-confirm
```

`prepare`不读取生成结果来选题；只传递问题、选项和图像绑定，丢弃答案标签。
150题450个Full旧轨迹可用于描述性基线，6750个新条件记录默认全部重跑；没有自动复用不同后端的旧生成。

## GPU准备与执行

1. 按锁定SHA下载作者代码到`vendor/`，不复制其整库到本仓库；保留作者许可证。使用R3同一Thinking checkpoint，`--weights-lock`指向其原始文件哈希清单。权重不下载到Git。
2. PriorTR框架与Hi-Lo需要各自的作者依赖环境，版本见锁文件。HF基准独立进程加载原生类。二者Transformers版本不同，因此环境桥接必须实测，不能将同为SDPA理解为等价。
3. 20道真实问题（五任务各4题）跑关闭剪枝、100%保留及压缩路径；固定前缀比较0/31/1023/4095/32766步logits。近cap前缀可以人工延长作为缓存压力诊断，但须另有自然长轨迹，二者不能混称。
4. `r4.diagnose`接收单个label-free row/job和token数组prefix，记录logits与36层trace；`--native`是未修改HF Full，`--disabled`关闭剪枝，`--keep100`保留所有视觉token。`r4.compare`输出数值差异，不自行设容差或颁发PASS。
5. `r4.run --diagnostic`运行完整自然32K上限；诊断与正式目录隔离。逐项核验后生成`r4.gates.CHECKS`要求的证据回执，绑定权重、代码、协议、环境和每个证据文件SHA。

```bash
python -m r4.run --panel .local-panel/panel.json --jobs .local-panel/jobs.json \
 --condition PriorTR25 --checkpoint /models/qwen8 --weights-lock /data/model_weights_lock.json \
 --author-root vendor/PriorTR --gate /data/measured-gate.json --out outputs/run1/PriorTR25
```

参考运行器为**单权重常驻、B1原生generate**，用于数值/机制对照，不宣称10路分页支持五种方法。正式批量提速需将各方法prefill与保留路径接入同一经过验收的分页decoder并另冻结profile；不能直接复用仅验证PriorTR的R3十路引擎。当前B1也须验收及计时后才可作为正式profile，不自动启动6750条。

## 代码入口

| 模块 | 功能 |
|---|---|
| `prepare/core` | 确定性抽样、独立集去重、6750/5250配对任务、输入隔离 |
| `adapters` | 固定作者FastV/PriorTR/VisPruner/SparseVLM/Hi-Lo；Random与官方像素控制；原索引与36层状态观察 |
| `diagnose/compare/gates` | 原生/包装路径固定前缀数值证据及拒绝不完整验收 |
| `run` | 不覆盖成功、保留失败、原前缀恢复、原token/原文/trace |
| `annotations` | 双人证据、首次错误token位置、M120分层与不足登记 |
| `interventions/continuation/mechanisms` | 同预算证据/随机救援、事实与中性长度匹配、精确前缀修改、Full同样停止干预及独立分支运行 |
| `analyze` | 三重复先按题平均、图像簇bootstrap、删失不剔除、未标注不当正确 |

干预组件必须由真实标注驱动；此包不自动生成“人工核验”证据。已有DeepSeek提取需另固定协议版本并抽查，抽取不能看标签，视觉事实标签不能由答案提取器替代。

`python -m r4.mechanisms --help`列出分支运行入口，recipe须包含父输出SHA和真实证据；停止分支可以复用早结束输出。prefix编辑在独立目录生成`intervention_result.json`，不会混入自然主表。`r4.offline`本地合并原始token、已审提取质量和人工标签，再用`r4.analyze`统计；缺失评分/标注保持null。
