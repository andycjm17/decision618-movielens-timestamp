# Decision 618：老师的切分方法与 MovieLens 重建

本次检查覆盖 Canvas 目前发布的 9 份技术讲义、1 份嘉宾讲义、课程大纲、项目说明和 23 个 notebook。技术讲义共 1,154 个 PDF 页面。Class 11 和 Class 12 的模块目前没有新文件。材料清单、Canvas 文件链接和 SHA-256 保存在 `sources/course_manifest.json`。以下页码均为从 1 开始计数的 **PDF 页面位置**；部分讲义页脚号码不同。Notebook cell 编号从 0 开始。

## 老师教的原则

Lecture 2 的 Train/Test Split 页面明确建议：有时间依赖时，把最近观察到的数据留作测试；没有时间依赖时，可以随机切分。需要保持类别比例时使用分层抽样。测试集要保留到模型确定后才用于评价。[Lec2_fall2026.pdf，PDF p.65](sources/Lec2_fall2026.pdf)

因此，70/30、80/20 和 MovieLens 的 99/1 都是具体案例的设置，不能作为所有项目的统一规定。训练集用于估计参数；训练内部的交叉验证或单独的 validation set 用于选模型；最终 test set 用于评价选定方案。时间预测中的每一轮训练都应早于它的验证数据。

| 材料 / 案例 | 老师实际使用的方法 | 本项目应学到什么 |
|---|---|---|
| Lecture 1，网络案例 | 随机约 70/30；PDF p.121 | 理解样本外评价；该例没有时间预测任务 |
| Lecture 2，线性回归 | 一般原则在 p.65；SolarPower 图例约 80/20，notebook 读取已切分 CSV | 切分方式由使用场景决定；不能把读取已有数据误写成重新随机切分 |
| Lecture 3，客户流失 | 随机 70/30、分层；PDF p.29，LogisticReg_dist cell 13，seed=2025 | 分类时关注 train/test 的类别比例 |
| Lecture 4，医疗费用 | 随机 70/30；PDF p.16，MedicalClaim_dist cell 12，seed=110 | 回归基准；调参不能用最终测试集 |
| Lecture 4，Parole | 分层 70/30；PDF p.98，Parole_dist seed=140 | 分类任务的分层与概率 / 分类误差 |
| Lecture 4，树模型调参 | 只在 training 中交叉验证；PDF pp.67-70 | 选好复杂度后，重训完整 training，再评价 test |
| Lecture 5，随机森林 | 70/30；PDF p.39，RandomForest_dist seed=144；训练内 10-fold CV | 外层切分与内层调参是两件事 |
| Lecture 5，Boosting notebook | 代码的快速 TEST RUN 为 2-fold，注释与讲义的演示配置另有差异 | 说明实际运行配置，避免把示范注释当成实验记录 |
| Lecture 6，FeatEng_Reg | 练习故意使用 25% train / 75% test | 这是演示过拟合的练习设置，不是推荐的项目默认值 |
| Lecture 6，Regularization / NYC Taxi | Regularization 使用给定 70/30 切分和训练内 10-fold CV；Taxi 读取既有 train/test | scaler、编码、正则化参数都应在训练阶段确定 |
| Lecture 7，MovieLens CF | 随机 99/1，seed=144；训练内部 20-fold CV 选 latent rank；PDF pp.47、52 | 忠实复现此基准，再增加独立的时间检验 |
| Lecture 7，描述性 clustering | PDF p.74 说明这种无监督分析无需监督学习式 train/test | 描述性分析可以用全样本，但不能用全样本拟合预测器后宣称测试准确率 |
| Lecture 8，MNIST / CNN | MNIST 使用官方 60,000 / 10,000；CNN 演示 cell 62 将 test_generator 用作 validation_data | 项目中保留独立的最终 test，不沿用这个演示快捷做法 |
| Lecture 10 / 嘉宾材料 | 优化与业务应用，没有新的预测切分规则 | 将误差、数据可用性和业务使用方式连接起来 |

Lecture 4 p.135 有一处选择 “maximizes average loss” 的笔误；p.136 和代码的正确逻辑是最小化验证损失。

## MovieLens 中忠实复现的部分

数据包含 1,000,209 条 1-5 星评分、6,040 名用户、3,706 部被评分电影；电影目录有 3,883 行。用户和电影主键唯一，评分没有缺失值或重复的 user-movie 对。这里的 observation 是一条用户对电影的评分，预测目标仍为评分；它与 NextItNet 的下一条 item 排序任务不同。

原始依据是 [ColFil_dist.ipynb](sources/ColFil_dist.ipynb)，以及 [Lecture 7](sources/lec7_clusteringCF.pdf)：

1. 按原始行号，使用旧版 NumPy `RandomState(144)` 抽取 99% 作为训练集，得到 **990,206 / 10,003** 条训练 / 测试记录。使用 `default_rng(144)` 会得到另一套切分。最前面的测试行号为 99、238、283、363、417（零基）。[cells 5-7；PDF p.47]
2. 在训练集上重新设 NumPy seed=144，随机分配 1-20 的 CV fold 编号。latent rank 试 1-20，`lambda=0`、`maxit=1000`，R `softImpute` 的默认 ALS 方法；预测截断在 1-5。[cells 10-14；PDF pp.45、52]
3. 从老师提供的每条折外预测重新计算全部 20 个 rank 的 RMSE、MAE 和 OSR²，全部与缓存表一致，最优 rank 为 8。另做了一整轮 **400 次新的 R 拟合**：最优 rank 为 9，CV RMSE=0.918274；rank=8 为 0.918956，差 0.000682。新结果单列在 `results/fresh_cv_metrics.csv`，不混同缓存审计。新 CV 每个拟合的 R seed 为 `144 + 100 * fold + rank`，原缓存的 R 初始化未知。课堂对照保持原缓存选出的 rank=8，另存 fresh-selected 的 refit，不按测试分数重新选 rank。
4. 用老师的全局 MovieID 编码重训 rank=8 的最终模型。原始 notebook 中一个训练未见过的电影被预测为 0 后截断成 1；课堂复现保留这个行为。时间实验使用明确的冷启动 fallback，不删除未知 ID 的测试记录。
5. 重建电影类别的 latent-factor 热力图，沿用老师的类别均值、行内最大绝对值缩放和符号规则。[cells 17-23] 因子方向和顺序不唯一，热力图中的值不是直接预测的星级，也不能把两个重训的 Factor 1 当成同一个客群。

原 notebook 的最终输出为 RMSE=0.919、MAE=0.709、OSR²=0.332；本次 rank=8 重训为 **0.919767 / 0.710589 / 0.330010**。原 notebook 没有设置 R 的随机种子，也没有记录 R 包版本。本次固定 R seed=144，使用 R 4.0.2 / softImpute 1.4-1，因此这是流程复现和独立新拟合，不宣称浮点输出逐位相同。Lecture 7 另一次 slide run 的结果为 0.916，不能与 notebook 的 0.919 混为同一次实验。

OSR² 的计算沿用 Lecture 2 pp.87-89：

`OSR² = 1 - Σ(y_test - prediction)² / Σ(y_test - mean(y_train))²`

分母使用训练均值。CV 中每条评分使用所属 fold 的其余 19 折均值作基准，按 notebook 计算 pooled OSR²；讲义的 AvgR² 是概念性说明。固定评价数据和分母时，最大 OSR² 与最小 RMSE 的 rank 选择一致。

## 重建的集成模型

Lecture 7 PDF pp.55-59 已经提出将 CF 输出、电影信息、用户信息以及 date/time 加入回归 / 树 / 随机森林。老师给出的 notebook 只实现 CF 和类别解释，没有提供这一 ensemble 的完整运行代码。本次沿用讲义思路，以 Lecture 6 的 Ridge 正则化重建一个可复跑的线性集成；没有声称还原了老师未提供的参数和全部 51 个变量。

静态版本包含 CF 预测、18 个类别、电影发行年份、用户性别 / 年龄类别 / 职业类别；时间版本在此基础上增加 UTC 小时 / 星期、月份的正余弦、经过天数、严格早于当前评分时刻的训练期评分次数，以及距离最后一次训练期评分的天数。不使用 ZIP。Calendar 特征是 UTC，用户本地时区不可用；原 notebook 将日期转为 Asia/Singapore，也不代表用户当地时间。

Ridge alpha 候选固定为 1、100、10,000。随机方案用老师的折外 CF 预测训练集成，并在训练内部按用户分组留出 20% 来选择 alpha。该内部选择没有再次嵌套重训 CF，属于近似 stacking 调参；独立的 1% 最终测试集完全不参与选择。

时间方案的 CF 只用过去数据生成后面区间的预测，不能用训练集上的 CF 拟合值来训练第二层。先用前 50% 预测 50%-65%，再用前 65% 预测 65%-80%；用 80%-90% validation 选择超参数。最终 CF 在前 90% 重训；第二层用 50%-90% 的 forward 折外预测重训，最后评价未来 10%。严格时间顺序的保证针对每条折外预测的拟合样本；超参数由共同 validation 决定，此处没有模拟历史上的逐期自动调参。

## Timestamp 扩展与结论

随机测试集中 **98.85%** 的记录，其用户在训练集中仍有更晚的评分。随机切分适合课堂的缺失评分补全；若业务问题是用现在的数据预测未来，它会利用在当时尚未发生的信息。这个比例不能直接等同于“98.85% 数据错误”。

新时间切分只看 timestamp 的 80% / 90% 分位点，不看 rating，且同一秒的全部记录留在同一侧：

| 阶段 | 行数 | UTC 时间区间 |
|---|---:|---|
| Train | 800,164 | 2000-04-25 23:05:32 至 2000-12-02 14:51:59 |
| Validation | 100,024 | 2000-12-02 14:52:18 至 2000-12-29 23:42:47 |
| Test | 100,021 | 2000-12-29 23:43:34 至 2003-02-28 17:49:50 |

最终测试中，前两段数据可以重训为 900,188 条。未知用户 / 电影使用训练均值和 shrinkage=20 的训练期边际偏差回退。history 特征只读取已拟合时期内、严格早于当前时刻的评分；同秒记录排除，最终测试期内不滚动更新历史。

| 模型 | 随机测试 RMSE | 时间测试 RMSE |
|---|---:|---:|
| Training mean | 1.1237 | 1.0893 |
| CF | 0.9198 | 1.2246 |
| CF + metadata | 0.8990 | 1.0303 |
| CF + metadata + timestamp | **0.8940** | **1.0001** |

时间特征的独立增益应该与同一测试集上的静态集成相比：随机测试 RMSE 下降 **0.0050**（0.56%），时间测试下降 **0.0302**（2.93%）。按用户做 2,000 次配对 bootstrap，95% 区间分别为 [0.0031, 0.0069] 和 [0.0111, 0.0504] 个评分点。这些区间描述固定实验的数据不确定性，不包含调参及切分方式的不确定性，也不证明业务收益或因果作用。

时间方案的 validation 只有 23.84% 是已知用户且已知电影，最终 test 则有 95.70%。时间区间也长短不一。这说明数据构成变化很大；不能将随机与时间 RMSE 的全部差异归因于未来信息。时间 CF 的低 rank / lambda=0 是 validation 选出的，在更多 warm 用户的 test 上表现不佳；独立 UDV 点积、冷启动预测和行号核对均通过，因此保留这一结果，没有根据 test 重新选择模型。

994,169 个用户内相邻评分对中，529,046 对同秒（53.21%），88.72% 在一分钟内。30 分钟间隔定义的评分录入批次共 25,163 个，中位数 6 条、90% 分位 115 条。这一阈值是描述性选择；不能称这些批次为观影 session。若继续 NextItNet，应将同秒顺序当作数据限制，并使用独立的时间评价及 ranking 指标，不能与本次星级 RMSE 直接比较。

## 对当前项目的使用方式

这套实验已覆盖老师要求的两个相关任务：数据可视化，以及预测模型建立与评价。课程对应点是 Lecture 2 的样本外评价、Lecture 4 的训练内调参、Lecture 6 的特征工程 / Ridge、Lecture 7 的 collaborative filtering / ensemble。

业务表述可写成：帮助 Media & Entertainment 平台在给定用户、电影及当前时间时估计用户评分，用于支持电影候选排序。本次证据支持较低的离线星级误差；Top-N 质量、点击、观看时长、留存和收入仍需另外验证。D3 应解释为何同时保留随机补全基准和未来测试，而不是将二者平均成一个分数。

本文件及 `MovieLens_course_rebuild.ipynb` 是新的研究与方法附件。原有 D3 报告 / 演示保持为旧实验的记录，不能把本次指标直接贴入其 NextItNet 表格。团队实际贡献须按真实完成的工作确认。最终提交要求仍为：最多 10 页、12pt、双倍行距、自包含的 PDF；另交数据、注释清楚的代码、slides，以及贡献附录。[TermProjectInstructions.pdf，PDF pp.1-3](sources/TermProjectInstructions.pdf)

材料仅保存在本地团队工作目录，没有对外发布或向 Canvas 提交。
