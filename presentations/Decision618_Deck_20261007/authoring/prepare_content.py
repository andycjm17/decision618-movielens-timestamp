from pathlib import Path
import json, csv, hashlib, re, os

ROOT = Path(os.environ.get('WORKSPACE_ROOT', str(Path.cwd()))).resolve()
LAB = ROOT / 'course_rebuild_20261007'
OUT = Path(os.environ.get('OUTPUT_DIR', str(ROOT / 'output/Decision618_Deck_20261007'))).resolve()
OUT.mkdir(parents=True, exist_ok=True)
read = lambda name: json.loads((LAB / 'results' / name).read_text())
metrics = list(csv.DictReader((LAB / 'results/model_metrics.csv').open()))
all_rows = [r for r in metrics if r['scope'] == 'all']
models = ['Training mean', 'softImpute', 'CF + metadata', 'CF + metadata + timestamp']
matrix = [[float(next(r['RMSE'] for r in all_rows if r['model'] == m and r['protocol'] == protocol))
           for protocol in ['random_99_1', 'global_time_80_10_10']] for m in models]
exp = read('timestamp_experiment.json')
eda = read('timestamp_eda.json')
split = read('temporal_split.json')
audit = read('analysis_audit.json')
assert audit['passed']
assert [x['rows'] for x in split['partitions']] == [800164, 100024, 100021]
assert sum(x['rows'] for x in split['partitions']) == 1000209
gains = []
for j, protocol in enumerate(['random', 'temporal']):
    gain = exp['intervals'][protocol]['CF + metadata']
    assert abs(matrix[2][j] - matrix[3][j] - gain['improvement_RMSE']) < 1e-12
    gains.append(gain)

slides = [
    dict(title='Movie rating prediction', label='Business problem', seconds=25,
         script='Our project belongs to Media and Entertainment. We ask whether a movie platform can predict a returning user\'s preferences more accurately. This October seventh version starts with the method taught in class and adds timestamp features. The immediate task is predicting a one-to-five star rating. Recommendation quality and business impact still need separate tests.',
         chinese='行业是 Media & Entertainment。10 月 7 日版本先复现老师的矩阵分解，再加入 metadata 和 timestamp。当前预测目标是 1 至 5 星评分，不再把 NextItNet 的下一物品 HR@10 当作这一版的结果。评分准确度、Top-N 推荐质量和商业收益是三个需要分别验证的问题。',
         sources=['GroupLens MovieLens 1M README; course_rebuild_20261007/Methods_and_findings_EN.md']),
    dict(title='MovieLens and rating activity', label='Data understanding', seconds=35,
         script='MovieLens contains about one million ratings from six thousand users. We checked the source tables, IDs and rating values, and kept missing user-movie pairs unobserved. Timing is useful but easy to misread. Fifty-three percent of adjacent ratings share the same second, and almost eighty-nine percent arrive within a minute. These timestamps describe when someone entered ratings. They cannot establish the order in which that person watched the movies.',
         chinese='MovieLens 1M 有 1,000,209 条评分、6,040 名用户、3,706 部实际被评分的电影，电影目录共有 3,883 部。未评分不等于不喜欢。994,169 对用户内相邻评分中，53.21% 同秒，88.72% 间隔不超过 60 秒，说明大量批量录入；timestamp 是评分录入时间，不是观看时间。95.74% 稀疏度以完整 3,883 部目录为分母。',
         sources=['course_rebuild_20261007/results/preparation.json; results/timestamp_eda.json', 'GroupLens MovieLens 1M README']),
    dict(title='Course baseline and fresh replication', label='Course replication', seconds=50,
         script='The instructor uses softImpute matrix factorization to predict missing ratings. We reproduced the exact random split and all twenty cross-validation folds, using the classroom NumPy seed of one forty-four. Rescoring the instructor\'s cache selects rank eight. Four hundred fresh R fits select rank nine, with rank eight close behind. We retain rank eight for the classroom comparison and report the fresh choice separately. Its test RMSE is point nine one nine eight, close to the instructor\'s point nine one nine. R initialization and package differences prevent a claim of identical numerical results.',
         chinese='复现的是 ColFil_dist.ipynb 的 softImpute：NumPy seed=144、990,206 条训练、10,003 条测试、20-fold CV，rank=1 至 20。老师缓存重新评分选 rank 8。400 次全新 R 拟合选 rank 9，CV RMSE=0.918274；rank 8 为 0.918956。为忠实课堂对比，主表沿用 rank 8，fresh-selected refit 另存。老师没有记录 R seed/版本；本次 R 4.0.2、softImpute 1.4-1，因此是流程复现，不能声称数值完全一致。表内 OSR² 以训练均值作为基准预测。',
         sources=['Decision 618 Lecture 7, PDF pp.45, 47, 52, 55-59; ColFil_dist.ipynb cells 5-15 and 17-23 (zero-based)', 'course_rebuild_20261007/results/cache_audit.json; fresh_cv_metrics.csv; teacher_rank8_fit.csv']),
    dict(title='Random completion and future prediction', label='Evaluation protocol', seconds=45,
         script='The split follows the question. The course\'s ninety-nine-to-one random split tests whether we can fill missing ratings. To test future ratings, we added a global chronological split of roughly eighty, ten and ten percent, keeping equal timestamps together. Validation chooses parameters, and the final fit uses training plus validation before the untouched test period. The ensemble learns from out-of-fold predictions. History features use only earlier fitted entries. Every comparison within a column uses the same holdout. The two columns cover different people and periods.',
         chinese='Lecture 2 区分无时间依赖的随机分割与预测未来的按时间分割。保留老师的 99/1 随机评分补全；另做全局 80/10/10 时间分割，行数为 800,164 / 100,024 / 100,021，同秒评分不跨边界。参数只在训练/CV/验证选择，最终以 900,188 条 train+validation 重拟合，测试标签不参与。时间 ensemble 用 expanding-time out-of-fold CF 预测，不能用训练内拟合值替代；history 只取冻结的 fit history，query 之后及同秒事件排除。两列目标相同但人群、时间窗、参数选择不同。',
         sources=['Decision 618 Lecture 2, PDF pp.65, 87-89; Lecture 7, PDF p.45', 'course_rebuild_20261007/results/temporal_split.json; timestamp_experiment.json; analysis_audit.json']),
    dict(title='Metadata and time improve rating accuracy', label='Results', seconds=65,
         script='Ridge combines CF predictions with movie and user metadata. We compare it with and without time features. Lower RMSE means smaller star-rating errors. On the random test, collaborative filtering scores point nine one nine eight. Metadata lowers that to point eight nine nine zero, and timestamp features lower it again to point eight nine four zero. On the future test, the timestamp ensemble scores one point zero zero zero one, compared with one point zero three zero three for the static ensemble. The controlled timestamp increments are about point zero zero five and point zero three zero two rating points. The future collaborative-filtering model performs worse than the training mean. We kept that result because it matters. These comparisons support the timestamp ensemble on these holdouts. They do not prove better viewing engagement.',
         chinese='Ridge ensemble 将 CF 预测与电影类型、上映年份和用户人口统计类别组合；timestamp 版再加入时间特征，正则化参数 alpha 只在验证中选择。结果来自 model_metrics.csv 的 scope=all，包含冷启动行。Random RMSE：训练均值 1.1237、CF 0.9198、CF+metadata 0.8990、CF+metadata+timestamp 0.8940。Future RMSE：1.0893、1.2246、1.0303、1.0001。timestamp 的增量须与同一列静态 ensemble 相减，得到 0.005022 与 0.030223，不是百分点，也不是点击率。Future CF 1.2246 比均值 1.0893 更差，已独立核对因子点积、行号及 fallback 后保留。不能用两列差异断言全部是时间带来的损失。',
         sources=['course_rebuild_20261007/results/model_metrics.csv (scope=all); timestamp_experiment.json', 'course_rebuild_20261007/results/independent_r_audit.csv; analysis_audit.json']),
    dict(title='Timestamp features and evaluation limits', label='Robustness', seconds=45,
         script='The course already discusses date and time. We test their incremental value alongside counts and gaps from strictly earlier ratings. Equal-second entries contribute no prior history. User-level paired bootstrap intervals remain above zero for both timestamp increments. Those intervals are conditional on this split and selection procedure. A limitation appears in the future evaluation: only twenty-four percent of validation ratings are warm, compared with ninety-six percent in the final test. The validation choice therefore transfers between very different cohorts. We need repeated time windows before treating this as a stable deployment result.',
         chinese='老师 Lecture 7 的 ensemble 已涉及日期和时间；本次贡献是可复现的组合模型、timestamp 增量比较与全局时间评估。Calendar 采用 UTC，用户本地时区未知。时间特征包括 UTC 日历、elapsed days、严格早于 query 的历史计数和间隔；同秒排除，漂移变量限制在 meta-fit 范围内。2,000 次按用户配对 bootstrap，统计量是 event-weighted RMSE：Random 增量 95% CI=[0.0031,0.0069]，Future=[0.0111,0.0504]。它们只反映固定分割、既定选模过程下的不确定性，不能当作因果效应。Warm 定义为用户及电影都在当时的 fit 数据中出现：validation warm=23.84%，test warm=95.70%，且 test fit 已加入 validation。这个人群变化限制参数迁移，需重复时间窗验证。',
         sources=['course_rebuild_20261007/results/timestamp_experiment.json; temporal_population.csv', 'Decision 618 Lecture 7, PDF pp.55-59; course_rebuild_20261007/code/evaluate_timestamp.py']),
    dict(title='Further checks before a business pilot', label='Business application', seconds=35,
         script='The next step is to evaluate Top-N ranking and cold users across additional time windows. If those checks hold, a limited randomized pilot could test recommendation-led movie starts per exposed user, with completion, catalog concentration and latency as guardrails. That is a proposed business test. MovieLens has no exposure logs, revenues or operating costs, so we cannot calculate ROI from these ratings. The current evidence supports further offline testing of the timestamp ensemble.',
         chinese='下一步先增加多个全局时间窗、Top-N 推荐指标和冷启动评估，再考虑有限随机对照试验。商业指标可设为每名曝光用户由推荐带来的开播次数，配套完播、目录集中度、延迟及隐私保护。这些是待执行方案，不是已取得的商业效果。MovieLens 没有曝光、收入、运营成本，不能计算或编造 ROI，也不能从较低 RMSE 推出更高留存。',
         sources=['course_rebuild_20261007/Methods_and_findings_EN.md; GroupLens MovieLens 1M README', 'Business pilot and guardrails are proposed, not measured outcomes'])
]
assert sum(s['seconds'] for s in slides) == 300
for s in slides:
    s['words'] = len(s['script'].split())
content = dict(date='2026-10-07', slides=slides, models=models, rmse=matrix,
               gains=gains, eda=eda, split=split,
               sources={'methods': 'course_rebuild_20261007/Methods_and_findings_EN.md',
                        'manifest': 'course_rebuild_20261007/sources/course_manifest.json'})
(OUT/'authoring/content.json').write_text(json.dumps(content, ensure_ascii=False, indent=2)+'\n')
elapsed=0
en=['# Five-minute presentation script, October 7, 2026', '',
    f"Seven slides, {sum(s['words'] for s in slides)} words. Planned slide budgets total 300 seconds. Rehearse aloud to adjust pauses.", '']
cn=['# 10 月 7 日演示稿：中文讲解', '', '英文 deck 共 7 页，计划 5 分钟。以下是理解和备讲说明。', '']
fmt=lambda t:f'{t//60}:{t%60:02d}'
for i,s in enumerate(slides,1):
    heading=f"## {i}. {s['title']} ({fmt(elapsed)}-{fmt(elapsed+s['seconds'])})"
    en += [heading,'',s['script'],'']
    cn += [heading,'',s['chinese'],'']
    cn += ['出处：'+'；'.join(s['sources']),'']
    elapsed+=s['seconds']
cn += ['## 关键口径与计算', '',
       'RMSE = sqrt(mean((真实评分 - 预测评分)^2))。越小越好，单位是评分点。', '',
       'Random timestamp 增量：0.8989972700 - 0.8939750154 = 0.0050222547，显示为 0.0050。', '',
       'Future timestamp 增量：1.0303196270 - 1.0000969635 = 0.0302226634，显示为 0.0302。', '',
       '相邻评分对数：1,000,209 - 6,040 = 994,169；同秒比例：529,046 / 994,169 = 53.214896%，显示为 53.21%。', '',
       'Fresh CV 是 20 个 rank × 20 个 fold = 400 次拟合。选 rank 9 不改变课堂 rank 8 对照的定位。', '',
       '时间分割合计：800,164 + 100,024 + 100,021 = 1,000,209。最终 fit：800,164 + 100,024 = 900,188。', '',
       'Bootstrap CI 是按用户有放回配对抽样、重新计算 event-weighted RMSE；不能把评分当作彼此独立用户。', '',
       '全部实验结果为已完成本地实验；业务 pilot、Top-N 评估和额外时间窗验证是后续计划。']
(OUT/'Speaker_Script_EN.md').write_text('\n'.join(en)+'\n')
(OUT/'Chinese_Explanation.md').write_text('\n'.join(cn)+'\n')
hashes={f:hashlib.sha256((LAB/f).read_bytes()).hexdigest() for f in [
    'results/model_metrics.csv','results/timestamp_experiment.json','results/timestamp_eda.json',
    'results/temporal_split.json','results/temporal_population.csv','results/fresh_cv_metrics.csv',
    'results/cache_audit.json','Methods_and_findings_EN.md']}
(OUT/'authoring/source_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
print('Prepared source-checked content:',sum(s['words'] for s in slides),'words; 300 seconds')
