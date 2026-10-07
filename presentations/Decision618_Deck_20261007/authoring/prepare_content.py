from pathlib import Path
import json, csv, hashlib, re, os

ROOT = Path(os.environ.get('WORKSPACE_ROOT', str(Path.cwd()))).resolve()
LAB = ROOT / 'course_rebuild_20261007'
OUT = Path(os.environ.get('OUTPUT_DIR', str(ROOT / 'output/Decision618_Deck_20261007_NextItNet'))).resolve()
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'authoring').mkdir(exist_ok=True)
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

# Re-score the recorded NextItNet experiment. This does not train a new model.
import numpy as np
import pandas as pd
NEXT = ROOT / 'final_project/nextitnet/results'
next_summary = json.loads((NEXT/'summary.json').read_text())
assert json.loads((NEXT/'audit.json').read_text())['all_passed']
per_user = pd.read_csv(NEXT/'test_per_user.csv')
sequence = pd.read_csv(NEXT/'sequence_split.csv.gz')
recorded = pd.read_csv(NEXT/'test_metrics.csv').set_index('model')
strict_recorded = pd.read_csv(NEXT/'strict_timestamp_metrics.csv').set_index('model')
assert len(sequence) == 1000209
assert not sequence.duplicated(['user', 'item']).any()
assert sequence.groupby('user').tail(1)['split'].eq('test').all()
previous_time = sequence.groupby('user')['time'].shift()
strict_ids = set(sequence.loc[sequence['split'].eq('test') & sequence['time'].gt(previous_time), 'user'])
assert len(strict_ids) == 3496
fit_ids = set(sequence.loc[sequence['split'].ne('test'), 'item'])
test = sequence.loc[sequence['split'].eq('test')]
eligible_ids = set(test.loc[test['item'].isin(fit_ids), 'user'])
assert len(fit_ids) == 3704 and len(eligible_ids) == 6038
strict_eligible = strict_ids & eligible_ids
assert len(strict_eligible) == 3494
ranking_models = ['Popularity', 'item-kNN', 'SVD', 'NextItNet']
ranking = []
for name in ranking_models:
    rows = per_user.loc[per_user['model'].eq(name)]
    assert set(rows['user_id']) == eligible_ids and len(rows) == 6038
    ranks = rows['rank'].to_numpy()
    hr = (ranks <= 10).astype(float)
    ndcg = np.where(ranks <= 10, 1 / np.log2(ranks + 1), 0)
    assert np.allclose(hr, rows['HR@10'], atol=1e-12)
    assert np.allclose(ndcg, rows['NDCG@10'], atol=1e-12)
    subgroup = rows.loc[rows['user_id'].isin(strict_eligible)]
    values = {}
    for metric in ['HR@10', 'NDCG@10']:
        all_value = float(rows[metric].mean())
        strict_value = float(subgroup[metric].mean())
        assert abs(all_value - float(recorded.loc[name, metric])) < 1e-12
        assert abs(strict_value - float(strict_recorded.loc[name, metric])) < 1e-12
        assert abs(all_value - next(x[metric] for x in next_summary['metrics'] if x['model'] == name)) < 1e-12
        values[metric] = all_value
        values['strict_' + metric] = strict_value
    ranking.append(dict(model=name, users=len(rows), strict_users=len(subgroup), **values))
next_gain = ranking[-1]['HR@10'] - ranking[1]['HR@10']
assert abs(next_gain - 0.1253726399470023) < 1e-12
next_evidence = dict(original_run_utc=next_summary['run_utc'], protocol=next_summary['protocol'],
                     adaptation=next_summary['adaptation'], selection=next_summary['selection'],
                     final_catalog=len(fit_ids), users=len(eligible_ids), strict_users=len(strict_eligible),
                     ranking=ranking, hr_gain_vs_itemknn=next_gain,
                     paired_ndcg_intervals=next_summary['paired_ndcg_intervals'],
                     verification='Recomputed HR@10 and NDCG@10 from saved per-user ranks. Both cohorts match saved aggregate metrics. No retraining.')
(OUT/'evidence').mkdir(exist_ok=True)
(OUT/'evidence/nextitnet_comparison.json').write_text(json.dumps(next_evidence, indent=2)+'\n')
for file in ['test_metrics.csv', 'strict_timestamp_metrics.csv']:
    (OUT/'evidence'/file).write_bytes((NEXT/file).read_bytes())

scripts = [
    "Our project asks how a movie platform can recommend more relevant films to returning users. This October seventh deck brings together the course's rating prediction method and our completed NextItNet experiment. We compare models within each task. Rating errors and recommendation ranks answer different questions.",
    "MovieLens has about one million ratings from six thousand users. We kept missing ratings unobserved. Timing needs care: fifty-three percent of adjacent ratings share the same second, and almost eighty-nine percent arrive within a minute. These are rating-entry timestamps. They cannot establish movie viewing order.",
    "The instructor uses softImpute matrix factorization. We reproduced the exact random split and twenty cross-validation folds with seed one forty-four. The instructor's cache selects rank eight. Four hundred fresh R fits select rank nine, with rank eight close behind. We retain rank eight for the classroom comparison. Its test RMSE is point nine one nine eight. The instructor did not record the R seed or package version, so this is a procedural replication.",
    "The course's ninety-nine-to-one random split tests completion of missing ratings. We also use a global chronological split to predict future ratings, keeping equal timestamps together. Validation chooses parameters. The final fit adds validation before the untouched test period. Ensemble training uses out-of-fold predictions, and history features use only earlier fitted entries. Models within each column share a holdout. The columns cover different people and periods.",
    "Lower RMSE means smaller rating errors. On the random test, CF scores point nine one nine eight. Metadata improves it to point eight nine nine zero, then timestamps to point eight nine four zero. On the future test, timestamps improve the static ensemble from one point zero three zero three to one point zero zero zero one. Future CF performs worse than the training mean. We keep that result because it matters.",
    "The course already discusses date and time. We measure their increment with calendar features and counts and gaps from earlier ratings. Both paired bootstrap intervals stay above zero. These intervals depend on this split and model selection. Another limit is the future cohort: warm ratings account for twenty-four percent of validation but ninety-six percent of final test. We need repeated time windows to check stability.",
    "For next-item ranking, adapted NextItNet puts the held-out movie in its top ten for nineteen point nine four percent of users. Item-kNN achieves seven point four zero percent, a gap of twelve point five four percentage points. NextItNet also leads on NDCG. On the subset whose test rating comes strictly later, its hit rate is fourteen point three one percent versus five point nine five for item-kNN. This reuses our completed experiment. We have not retrained it for October seventh.",
    "The ranking test holds out each user's last rating for test and the previous one for validation. We select a ten-item context using validation, refit on training plus validation, and remove previously rated movies from candidates. Equal timestamps use seeded tie-breaking. This split can still include other users' later ratings. SVD here optimizes star ratings, so its weak ranking score does not establish that every matrix-factorization method is weak.",
    "Both approaches warrant further offline testing. Compare ranking models on shared global time windows and test cold-user fallbacks. A later randomized pilot could measure movie starts per exposed user. MovieLens has no exposure, revenue or cost data to establish engagement gains or ROI."
]
budgets = [25,25,35,35,45,35,45,35,20]
slides[0].update(title='Movie recommendations',
    chinese='行业是 Media & Entertainment。10 月 7 日版合并两项已完成工作：老师的评分预测方法及 timestamp 扩展，以及此前适配版 NextItNet 的下一部被评分电影排序实验。评分任务内用 RMSE 对比，排序任务内用 HR@10/NDCG@10 对比。它们支持不同用途，不能直接据不同指标宣布一个总冠军。')
slides[4]['title'] = 'Rating prediction results'
slides[6:6] = [
    dict(title='NextItNet ranking results', label='Next-item ranking',
         chinese='本页重用已经完成的适配版 NextItNet 实验，原运行记录为 2026-09-12 03:38:08 UTC（本地 9 月 11 日），不是 10 月 7 日重新训练。主样本为所有方法相同的 6,038 名可评估用户，最终目录为 3,704 部电影，移除用户此前评分过的电影，目标是下一部被评分电影，评分高低都包括。Popularity / item-kNN / SVD / NextItNet 的 HR@10 分别为 4.09% / 7.40% / 1.21% / 19.94%，NDCG@10 为 0.0206 / 0.0389 / 0.0055 / 0.1049。NextItNet 比 item-kNN 高 12.54 个百分点。测试目标时间严格晚于前一条评分的子样本有 3,496 人，剔除 2 个目录外目标后共同评估 3,494 人，HR@10 为 3.81% / 5.95% / 1.35% / 14.31%，NDCG 为 0.0191 / 0.0311 / 0.0066 / 0.0758。这个子样本避免测试边界同秒排序，但仍不是全局未来期测试。NextItNet 相对 item-kNN 的配对 NDCG 差值为 0.065963，95% bootstrap CI=[0.059809,0.072244]，该区间针对 NDCG，不是 HR。',
         sources=['final_project/nextitnet/results/test_metrics.csv; strict_timestamp_metrics.csv; test_per_user.csv',
                  'final_project/nextitnet/results/summary.json; selection_before_test.json; protocol_before_training.json']),
    dict(title='Rating prediction and next-item ranking', label='Comparing the tasks',
         chinese='评分预测问用户会给一部电影几星，目标为 1 至 5 星，以 RMSE 越低越好衡量；排序问哪部电影会成为下一条评分，以 HR@10 和 NDCG@10 越高越好衡量。softImpute 与 SVD 都属矩阵分解，但它们是不同实现和实验，不能混为一个基线。NextItNet 按每名用户的时间序列留最后一条为 test、倒数第二条为 validation，相同 timestamp 用 seed=42 的随机键排序。训练 988,129 条评分减去每人第一条无前序位置 6,040，得到 982,089 个训练转移。它沿用官方 Recommenders 的因果残差卷积 encoder，适配为仅电影的 32 维 embedding 与 full-softmax，替代原 item/category candidate-MLP 包装。非重叠 teacher-forced chunk 的边界重置状态。validation 从窗口 10/20/50 中选 10，epoch=8，item-kNN K=40，SVD factors=100。最终各模型加入 validation 重拟合、排除 test 标签，共有 2 个目录外 test 目标，所有模型统一剔除。SVD 优化星级预测，这次低 HR 不能推广到所有矩阵分解或排名优化的 MF。每用户时间留出仍允许其他用户较晚评分进入 fit；需要统一全局时间窗后才能与 timestamp ensemble 作同任务排序比较。',
         sources=['final_project/nextitnet/results/summary.json; protocol_before_training.json; selection_before_test.json',
                  'final_project/nextitnet/code/experiment.py; course_rebuild_20261007/Methods_and_findings_EN.md'])
]
slides[-1]['chinese'] = '已有评分 RMSE 与每用户留出排序结果。下一步在多个共享全局时间窗下比较排名模型，并测试冷启动 fallback，再考虑有限随机试验。可用每名曝光用户由推荐带来的开播次数衡量商业结果，配套完播、目录集中度和延迟指标。MovieLens 没有曝光、收入和运营成本，这些是待执行方案，不能把离线准确度当作实际参与度或 ROI。'
for s, script, seconds in zip(slides, scripts, budgets):
    s.update(script=script, seconds=seconds, words=len(script.split()))
assert len(slides)==9 and sum(s['seconds'] for s in slides)==300
content = dict(date='2026-10-07', slides=slides, models=models, rmse=matrix,
               gains=gains, eda=eda, split=split, nextitnet=next_evidence,
               sources={'methods':'course_rebuild_20261007/Methods_and_findings_EN.md',
                        'manifest':'course_rebuild_20261007/sources/course_manifest.json'})
(OUT/'authoring/content.json').write_text(json.dumps(content,ensure_ascii=False,indent=2)+'\n')
elapsed=0
en=['# Five-minute presentation script, October 7, 2026','',
    f"Nine slides, {sum(s['words'] for s in slides)} words. Planned slide budgets total 300 seconds. Rehearse aloud to adjust pauses.",'']
cn=['# 10 月 7 日演示稿：评分预测与 NextItNet 对比','',
    '英文 deck 共 9 页，计划 5 分钟。以下是理解和备讲说明。','']
fmt=lambda t:f'{t//60}:{t%60:02d}'
for i,s in enumerate(slides,1):
    heading=f"## {i}. {s['title']} ({fmt(elapsed)}-{fmt(elapsed+s['seconds'])})"
    en += [heading,'',s['script'],'']
    cn += [heading,'',s['chinese'],'','出处：'+'；'.join(s['sources']),'']
    elapsed+=s['seconds']
cn += ['## 关键口径与计算','',
    'RMSE = sqrt(mean((真实评分 - 预测评分)^2))。越小越好，单位是评分点。','',
    'Random timestamp 增量：0.8989972700 - 0.8939750154 = 0.0050222547，显示为 0.0050。','',
    'Future timestamp 增量：1.0303196270 - 1.0000969635 = 0.0302226634，显示为 0.0302。','',
    '每名用户有一个留出目标。HR@10 = 测试目标排在前 10 的用户数 / 可评估用户数。NDCG@10：目标排名 r ≤ 10 时为 1/log2(r+1)，否则为 0，然后按用户平均。HR 更高表示命中更多用户，NDCG 更高也奖励更靠前的命中。','',
    'NextItNet HR@10：1,204 / 6,038 = 0.1994037761 = 19.94%。item-kNN：447 / 6,038 = 0.0740311361 = 7.40%。差：100 × (0.1994037761 - 0.0740311361) = 12.537264 个百分点，显示为 12.54。','',
    '严格测试边界子样本：3,496 - 2 个目录外目标 = 3,494。NextItNet 命中 500 / 3,494 = 14.31%，item-kNN 命中 208 / 3,494 = 5.95%。这仍是每用户留出，不是全局未来期。','',
    '训练转移：988,129 条 training 评分 - 6,040 个序列首位置 = 982,089。train + validation 重拟合时共有 994,169 条评分，测试 6,040 条评分不进入拟合。','',
    '相邻评分对数：1,000,209 - 6,040 = 994,169；同秒比例：529,046 / 994,169 = 53.214896%，显示为 53.21%。','',
    'Fresh CV：20 个 rank × 20 个 fold = 400 次拟合。选 rank 9 不改变课堂 rank 8 对照的定位。','',
    '全局时间分割合计：800,164 + 100,024 + 100,021 = 1,000,209。最终 fit：800,164 + 100,024 = 900,188。','',
    '评分增量 CI 按用户配对 bootstrap，重新计算 event-weighted RMSE。排序 CI 比较用户级 NDCG 差值，两套区间的含义不同。','',
    '全部数值来自已完成的本地实验。9 月 NextItNet 与 10 月评分实验使用同一 MovieLens 1M 评分表，但预测任务和留出协议不同。共享全局时间排名评估及商业 pilot 是后续计划。']
(OUT/'Speaker_Script_EN.md').write_text('\n'.join(en)+'\n')
(OUT/'Chinese_Explanation.md').write_text('\n'.join(cn)+'\n')
files = [LAB/f for f in ['results/model_metrics.csv','results/timestamp_experiment.json','results/timestamp_eda.json',
    'results/temporal_split.json','results/temporal_population.csv','results/fresh_cv_metrics.csv',
    'results/cache_audit.json','Methods_and_findings_EN.md']]
files += [NEXT/f for f in ['test_metrics.csv','strict_timestamp_metrics.csv','test_per_user.csv',
    'sequence_split.csv.gz','summary.json','audit.json','selection_before_test.json','protocol_before_training.json']]
hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
(OUT/'authoring/source_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
print('Prepared source-checked comparisons:',sum(s['words'] for s in slides),'words; 9 slides; 300 seconds')
