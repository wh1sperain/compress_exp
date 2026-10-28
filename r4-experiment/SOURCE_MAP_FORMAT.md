# 六任务官方导出字段映射

`extension.import_sources`接收一个包含ChartQA/MMMU/MathVista/MathVerse/GQA/AI2D六个键的JSON对象。每个值指定：

- `revision`：固定官方数据版本，拒绝main/HEAD/latest。
- `local_path`：官方JSON导出路径；缺失时仅从可选`url`下载该导出，先核对`sha256`，不自动下载图片。
- `sha256`：原JSON文件的SHA256。
- `records_path`：导出中题目列表/字典的点分路径，根为`""`。
- `image_root`：原图所在目录。
- `fields`：统一字段名到原记录点分路径的映射。
- `constants`：划分共有的元数据（不得放答案的伪造默认值）。
- `value_maps`：可选，显式值映射，如MMMU30学科到6大类、AI2D源索引到选项字母、MathVerse版本标准名。

统一字段至少：original_id/question/choices/gold_answer/question_type/task_type/image_paths/source_id；MathVista另需answer_type/precision/unit（不适用可null）；ChartQA须subset=Human或Augmented；MMMU须discipline为六大类之一；MathVerse须variant=Vision Dominant。字符串形式choices先在官方导出层转有序数组，不能eval执行。

示例单个条目（占位值必须换成真实固定来源；不是已完成的数据源清单）：

```json
{"ChartQA":{"revision":"FIXED_DATA_VERSION","local_path":"/data/chartqa-test.json","sha256":"REAL_SHA256","records_path":"","image_root":"/data/chartqa/test/png","fields":{"original_id":"id","question":"question","choices":"choices","gold_answer":"answer","question_type":"question_type","task_type":"task_type","image_paths":"image","source_id":"id","subset":"subset"}}}
```

完整归档manifest含gold，生成仅读取input白名单，评分读取隔离labels。所有题目必须在推理前过滤；若图像/Processor输入审计发现不可执行题，应在正式冻结前统一处理，不能看正式生成表现再换题。
