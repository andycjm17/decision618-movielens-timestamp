# 10 月 7 日演示稿：评分预测与 NextItNet 对比

英文 deck 共 9 页，计划 5 分钟。以下是理解和备讲说明。

## 1. Movie recommendations (0:00-0:25)

行业是 Media & Entertainment。10 月 7 日版合并两项已完成工作：老师的评分预测方法及 timestamp 扩展，以及此前适配版 NextItNet 的下一部被评分电影排序实验。评分任务内用 RMSE 对比，排序任务内用 HR@10/NDCG@10 对比。它们支持不同用途，不能直接据不同指标宣布一个总冠军。

出处：GroupLens MovieLens 1M README; course_rebuild_20261007/Methods_and_findings_EN.md

## 2. MovieLens and rating activity (0:25-0:50)

MovieLens 1M 有 1,000,209 条评分、6,040 名用户、3,706 部实际被评分的电影，电影目录共有 3,883 部。未评分不等于不喜欢。994,169 对用户内相邻评分中，53.21% 同秒，88.72% 间隔不超过 60 秒，说明大量批量录入；timestamp 是评分录入时间，不是观看时间。95.74% 稀疏度以完整 3,883 部目录为分母。

出处：course_rebuild_20261007/results/preparation.json; results/timestamp_eda.json；GroupLens MovieLens 1M README

## 3. Course baseline and fresh replication (0:50-1:25)

复现的是 ColFil_dist.ipynb 的 softImpute：NumPy seed=144、990,206 条训练、10,003 条测试、20-fold CV，rank=1 至 20。老师缓存重新评分选 rank 8。400 次全新 R 拟合选 rank 9，CV RMSE=0.918274；rank 8 为 0.918956。为忠实课堂对比，主表沿用 rank 8，fresh-selected refit 另存。老师没有记录 R seed/版本；本次 R 4.0.2、softImpute 1.4-1，因此是流程复现，不能声称数值完全一致。表内 OSR² 以训练均值作为基准预测。

出处：Decision 618 Lecture 7, PDF pp.45, 47, 52, 55-59; ColFil_dist.ipynb cells 5-15 and 17-23 (zero-based)；course_rebuild_20261007/results/cache_audit.json; fresh_cv_metrics.csv; teacher_rank8_fit.csv

## 4. Random completion and future prediction (1:25-2:00)

Lecture 2 区分无时间依赖的随机分割与预测未来的按时间分割。保留老师的 99/1 随机评分补全；另做全局 80/10/10 时间分割，行数为 800,164 / 100,024 / 100,021，同秒评分不跨边界。参数只在训练/CV/验证选择，最终以 900,188 条 train+validation 重拟合，测试标签不参与。时间 ensemble 用 expanding-time out-of-fold CF 预测，不能用训练内拟合值替代；history 只取冻结的 fit history，query 之后及同秒事件排除。两列目标相同但人群、时间窗、参数选择不同。

出处：Decision 618 Lecture 2, PDF pp.65, 87-89; Lecture 7, PDF p.45；course_rebuild_20261007/results/temporal_split.json; timestamp_experiment.json; analysis_audit.json

## 5. Rating prediction results (2:00-2:45)

Ridge ensemble 将 CF 预测与电影类型、上映年份和用户人口统计类别组合；timestamp 版再加入时间特征，正则化参数 alpha 只在验证中选择。结果来自 model_metrics.csv 的 scope=all，包含冷启动行。Random RMSE：训练均值 1.1237、CF 0.9198、CF+metadata 0.8990、CF+metadata+timestamp 0.8940。Future RMSE：1.0893、1.2246、1.0303、1.0001。timestamp 的增量须与同一列静态 ensemble 相减，得到 0.005022 与 0.030223，不是百分点，也不是点击率。Future CF 1.2246 比均值 1.0893 更差，已独立核对因子点积、行号及 fallback 后保留。不能用两列差异断言全部是时间带来的损失。

出处：course_rebuild_20261007/results/model_metrics.csv (scope=all); timestamp_experiment.json；course_rebuild_20261007/results/independent_r_audit.csv; analysis_audit.json

## 6. Timestamp features and evaluation limits (2:45-3:20)

老师 Lecture 7 的 ensemble 已涉及日期和时间；本次贡献是可复现的组合模型、timestamp 增量比较与全局时间评估。Calendar 采用 UTC，用户本地时区未知。时间特征包括 UTC 日历、elapsed days、严格早于 query 的历史计数和间隔；同秒排除，漂移变量限制在 meta-fit 范围内。2,000 次按用户配对 bootstrap，统计量是 event-weighted RMSE：Random 增量 95% CI=[0.0031,0.0069]，Future=[0.0111,0.0504]。它们只反映固定分割、既定选模过程下的不确定性，不能当作因果效应。Warm 定义为用户及电影都在当时的 fit 数据中出现：validation warm=23.84%，test warm=95.70%，且 test fit 已加入 validation。这个人群变化限制参数迁移，需重复时间窗验证。

出处：course_rebuild_20261007/results/timestamp_experiment.json; temporal_population.csv；Decision 618 Lecture 7, PDF pp.55-59; course_rebuild_20261007/code/evaluate_timestamp.py

## 7. NextItNet ranking results (3:20-4:05)

本页重用已经完成的适配版 NextItNet 实验，原运行记录为 2026-09-12 03:38:08 UTC（本地 9 月 11 日），不是 10 月 7 日重新训练。主样本为所有方法相同的 6,038 名可评估用户，最终目录为 3,704 部电影，移除用户此前评分过的电影，目标是下一部被评分电影，评分高低都包括。Popularity / item-kNN / SVD / NextItNet 的 HR@10 分别为 4.09% / 7.40% / 1.21% / 19.94%，NDCG@10 为 0.0206 / 0.0389 / 0.0055 / 0.1049。NextItNet 比 item-kNN 高 12.54 个百分点。测试目标时间严格晚于前一条评分的子样本有 3,496 人，剔除 2 个目录外目标后共同评估 3,494 人，HR@10 为 3.81% / 5.95% / 1.35% / 14.31%，NDCG 为 0.0191 / 0.0311 / 0.0066 / 0.0758。这个子样本避免测试边界同秒排序，但仍不是全局未来期测试。NextItNet 相对 item-kNN 的配对 NDCG 差值为 0.065963，95% bootstrap CI=[0.059809,0.072244]，该区间针对 NDCG，不是 HR。

出处：final_project/nextitnet/results/test_metrics.csv; strict_timestamp_metrics.csv; test_per_user.csv；final_project/nextitnet/results/summary.json; selection_before_test.json; protocol_before_training.json

## 8. Rating prediction and next-item ranking (4:05-4:40)

评分预测问用户会给一部电影几星，目标为 1 至 5 星，以 RMSE 越低越好衡量；排序问哪部电影会成为下一条评分，以 HR@10 和 NDCG@10 越高越好衡量。softImpute 与 SVD 都属矩阵分解，但它们是不同实现和实验，不能混为一个基线。NextItNet 按每名用户的时间序列留最后一条为 test、倒数第二条为 validation，相同 timestamp 用 seed=42 的随机键排序。训练 988,129 条评分减去每人第一条无前序位置 6,040，得到 982,089 个训练转移。它沿用官方 Recommenders 的因果残差卷积 encoder，适配为仅电影的 32 维 embedding 与 full-softmax，替代原 item/category candidate-MLP 包装。非重叠 teacher-forced chunk 的边界重置状态。validation 从窗口 10/20/50 中选 10，epoch=8，item-kNN K=40，SVD factors=100。最终各模型加入 validation 重拟合、排除 test 标签，共有 2 个目录外 test 目标，所有模型统一剔除。SVD 优化星级预测，这次低 HR 不能推广到所有矩阵分解或排名优化的 MF。每用户时间留出仍允许其他用户较晚评分进入 fit；需要统一全局时间窗后才能与 timestamp ensemble 作同任务排序比较。

出处：final_project/nextitnet/results/summary.json; protocol_before_training.json; selection_before_test.json；final_project/nextitnet/code/experiment.py; course_rebuild_20261007/Methods_and_findings_EN.md

## 9. Further checks before a business pilot (4:40-5:00)

已有评分 RMSE 与每用户留出排序结果。下一步在多个共享全局时间窗下比较排名模型，并测试冷启动 fallback，再考虑有限随机试验。可用每名曝光用户由推荐带来的开播次数衡量商业结果，配套完播、目录集中度和延迟指标。MovieLens 没有曝光、收入和运营成本，这些是待执行方案，不能把离线准确度当作实际参与度或 ROI。

出处：course_rebuild_20261007/Methods_and_findings_EN.md; GroupLens MovieLens 1M README；Business pilot and guardrails are proposed, not measured outcomes

## 关键口径与计算

RMSE = sqrt(mean((真实评分 - 预测评分)^2))。越小越好，单位是评分点。

Random timestamp 增量：0.8989972700 - 0.8939750154 = 0.0050222547，显示为 0.0050。

Future timestamp 增量：1.0303196270 - 1.0000969635 = 0.0302226634，显示为 0.0302。

每名用户有一个留出目标。HR@10 = 测试目标排在前 10 的用户数 / 可评估用户数。NDCG@10：目标排名 r ≤ 10 时为 1/log2(r+1)，否则为 0，然后按用户平均。HR 更高表示命中更多用户，NDCG 更高也奖励更靠前的命中。

NextItNet HR@10：1,204 / 6,038 = 0.1994037761 = 19.94%。item-kNN：447 / 6,038 = 0.0740311361 = 7.40%。差：100 × (0.1994037761 - 0.0740311361) = 12.537264 个百分点，显示为 12.54。

严格测试边界子样本：3,496 - 2 个目录外目标 = 3,494。NextItNet 命中 500 / 3,494 = 14.31%，item-kNN 命中 208 / 3,494 = 5.95%。这仍是每用户留出，不是全局未来期。

训练转移：988,129 条 training 评分 - 6,040 个序列首位置 = 982,089。train + validation 重拟合时共有 994,169 条评分，测试 6,040 条评分不进入拟合。

相邻评分对数：1,000,209 - 6,040 = 994,169；同秒比例：529,046 / 994,169 = 53.214896%，显示为 53.21%。

Fresh CV：20 个 rank × 20 个 fold = 400 次拟合。选 rank 9 不改变课堂 rank 8 对照的定位。

全局时间分割合计：800,164 + 100,024 + 100,021 = 1,000,209。最终 fit：800,164 + 100,024 = 900,188。

评分增量 CI 按用户配对 bootstrap，重新计算 event-weighted RMSE。排序 CI 比较用户级 NDCG 差值，两套区间的含义不同。

全部数值来自已完成的本地实验。9 月 NextItNet 与 10 月评分实验使用同一 MovieLens 1M 评分表，但预测任务和留出协议不同。共享全局时间排名评估及商业 pilot 是后续计划。
