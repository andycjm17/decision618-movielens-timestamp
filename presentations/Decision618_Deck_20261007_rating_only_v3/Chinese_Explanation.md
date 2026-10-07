# 10 月 7 日演示稿：中文讲解

英文 deck 共 7 页，计划 5 分钟。以下是理解和备讲说明。

## 1. Movie rating prediction (0:00-0:25)

行业是 Media & Entertainment。10 月 7 日版本先复现老师的矩阵分解，再加入 metadata 和 timestamp。当前预测目标是 1 至 5 星评分，不再把 NextItNet 的下一物品 HR@10 当作这一版的结果。评分准确度、Top-N 推荐质量和商业收益是三个需要分别验证的问题。

出处：GroupLens MovieLens 1M README; course_rebuild_20261007/Methods_and_findings_EN.md

## 2. MovieLens and rating activity (0:25-1:00)

MovieLens 1M 有 1,000,209 条评分、6,040 名用户、3,706 部实际被评分的电影，电影目录共有 3,883 部。未评分不等于不喜欢。994,169 对用户内相邻评分中，53.21% 同秒，88.72% 间隔不超过 60 秒，说明大量批量录入；timestamp 是评分录入时间，不是观看时间。95.74% 稀疏度以完整 3,883 部目录为分母。

出处：course_rebuild_20261007/results/preparation.json; results/timestamp_eda.json；GroupLens MovieLens 1M README

## 3. Course baseline and fresh replication (1:00-1:50)

复现的是 ColFil_dist.ipynb 的 softImpute：NumPy seed=144、990,206 条训练、10,003 条测试、20-fold CV，rank=1 至 20。老师缓存重新评分选 rank 8。400 次全新 R 拟合选 rank 9，CV RMSE=0.918274；rank 8 为 0.918956。为忠实课堂对比，主表沿用 rank 8，fresh-selected refit 另存。老师没有记录 R seed/版本；本次 R 4.0.2、softImpute 1.4-1，因此是流程复现，不能声称数值完全一致。表内 OSR² 以训练均值作为基准预测。

出处：Decision 618 Lecture 7, PDF pp.45, 47, 52, 55-59; ColFil_dist.ipynb cells 5-15 and 17-23 (zero-based)；course_rebuild_20261007/results/cache_audit.json; fresh_cv_metrics.csv; teacher_rank8_fit.csv

## 4. Random completion and future prediction (1:50-2:35)

Lecture 2 区分无时间依赖的随机分割与预测未来的按时间分割。保留老师的 99/1 随机评分补全；另做全局 80/10/10 时间分割，行数为 800,164 / 100,024 / 100,021，同秒评分不跨边界。参数只在训练/CV/验证选择，最终以 900,188 条 train+validation 重拟合，测试标签不参与。时间 ensemble 用 expanding-time out-of-fold CF 预测，不能用训练内拟合值替代；history 只取冻结的 fit history，query 之后及同秒事件排除。两列目标相同但人群、时间窗、参数选择不同。

出处：Decision 618 Lecture 2, PDF pp.65, 87-89; Lecture 7, PDF p.45；course_rebuild_20261007/results/temporal_split.json; timestamp_experiment.json; analysis_audit.json

## 5. Metadata and time improve rating accuracy (2:35-3:40)

Ridge ensemble 将 CF 预测与电影类型、上映年份和用户人口统计类别组合；timestamp 版再加入时间特征，正则化参数 alpha 只在验证中选择。结果来自 model_metrics.csv 的 scope=all，包含冷启动行。Random RMSE：训练均值 1.1237、CF 0.9198、CF+metadata 0.8990、CF+metadata+timestamp 0.8940。Future RMSE：1.0893、1.2246、1.0303、1.0001。timestamp 的增量须与同一列静态 ensemble 相减，得到 0.005022 与 0.030223，不是百分点，也不是点击率。Future CF 1.2246 比均值 1.0893 更差，已独立核对因子点积、行号及 fallback 后保留。不能用两列差异断言全部是时间带来的损失。

出处：course_rebuild_20261007/results/model_metrics.csv (scope=all); timestamp_experiment.json；course_rebuild_20261007/results/independent_r_audit.csv; analysis_audit.json

## 6. Timestamp features and evaluation limits (3:40-4:25)

老师 Lecture 7 的 ensemble 已涉及日期和时间；本次贡献是可复现的组合模型、timestamp 增量比较与全局时间评估。Calendar 采用 UTC，用户本地时区未知。时间特征包括 UTC 日历、elapsed days、严格早于 query 的历史计数和间隔；同秒排除，漂移变量限制在 meta-fit 范围内。2,000 次按用户配对 bootstrap，统计量是 event-weighted RMSE：Random 增量 95% CI=[0.0031,0.0069]，Future=[0.0111,0.0504]。它们只反映固定分割、既定选模过程下的不确定性，不能当作因果效应。Warm 定义为用户及电影都在当时的 fit 数据中出现：validation warm=23.84%，test warm=95.70%，且 test fit 已加入 validation。这个人群变化限制参数迁移，需重复时间窗验证。

出处：course_rebuild_20261007/results/timestamp_experiment.json; temporal_population.csv；Decision 618 Lecture 7, PDF pp.55-59; course_rebuild_20261007/code/evaluate_timestamp.py

## 7. Further checks before a business pilot (4:25-5:00)

下一步先增加多个全局时间窗、Top-N 推荐指标和冷启动评估，再考虑有限随机对照试验。商业指标可设为每名曝光用户由推荐带来的开播次数，配套完播、目录集中度、延迟及隐私保护。这些是待执行方案，不是已取得的商业效果。MovieLens 没有曝光、收入、运营成本，不能计算或编造 ROI，也不能从较低 RMSE 推出更高留存。

出处：course_rebuild_20261007/Methods_and_findings_EN.md; GroupLens MovieLens 1M README；Business pilot and guardrails are proposed, not measured outcomes

## 关键口径与计算

RMSE = sqrt(mean((真实评分 - 预测评分)^2))。越小越好，单位是评分点。

Random timestamp 增量：0.8989972700 - 0.8939750154 = 0.0050222547，显示为 0.0050。

Future timestamp 增量：1.0303196270 - 1.0000969635 = 0.0302226634，显示为 0.0302。

相邻评分对数：1,000,209 - 6,040 = 994,169；同秒比例：529,046 / 994,169 = 53.214896%，显示为 53.21%。

Fresh CV 是 20 个 rank × 20 个 fold = 400 次拟合。选 rank 9 不改变课堂 rank 8 对照的定位。

时间分割合计：800,164 + 100,024 + 100,021 = 1,000,209。最终 fit：800,164 + 100,024 = 900,188。

Bootstrap CI 是按用户有放回配对抽样、重新计算 event-weighted RMSE；不能把评分当作彼此独立用户。

全部实验结果为已完成本地实验；业务 pilot、Top-N 评估和额外时间窗验证是后续计划。
