# Five-minute presentation script, October 7, 2026

Nine slides, 557 words. Planned slide budgets total 300 seconds. Rehearse aloud to adjust pauses.

## 1. Movie recommendations (0:00-0:25)

Our project asks how a movie platform can recommend more relevant films to returning users. This October seventh deck brings together the course's rating prediction method and our completed NextItNet experiment. We compare models within each task. Rating errors and recommendation ranks answer different questions.

## 2. MovieLens and rating activity (0:25-0:50)

MovieLens has about one million ratings from six thousand users. We kept missing ratings unobserved. Timing needs care: fifty-three percent of adjacent ratings share the same second, and almost eighty-nine percent arrive within a minute. These are rating-entry timestamps. They cannot establish movie viewing order.

## 3. Course baseline and fresh replication (0:50-1:25)

The instructor uses softImpute matrix factorization. We reproduced the exact random split and twenty cross-validation folds with seed one forty-four. The instructor's cache selects rank eight. Four hundred fresh R fits select rank nine, with rank eight close behind. We retain rank eight for the classroom comparison. Its test RMSE is point nine one nine eight. The instructor did not record the R seed or package version, so this is a procedural replication.

## 4. Random completion and future prediction (1:25-2:00)

The course's ninety-nine-to-one random split tests completion of missing ratings. We also use a global chronological split to predict future ratings, keeping equal timestamps together. Validation chooses parameters. The final fit adds validation before the untouched test period. Ensemble training uses out-of-fold predictions, and history features use only earlier fitted entries. Models within each column share a holdout. The columns cover different people and periods.

## 5. Rating prediction results (2:00-2:45)

Lower RMSE means smaller rating errors. On the random test, CF scores point nine one nine eight. Metadata improves it to point eight nine nine zero, then timestamps to point eight nine four zero. On the future test, timestamps improve the static ensemble from one point zero three zero three to one point zero zero zero one. Future CF performs worse than the training mean. We keep that result because it matters.

## 6. Timestamp features and evaluation limits (2:45-3:20)

The course already discusses date and time. We measure their increment with calendar features and counts and gaps from earlier ratings. Both paired bootstrap intervals stay above zero. These intervals depend on this split and model selection. Another limit is the future cohort: warm ratings account for twenty-four percent of validation but ninety-six percent of final test. We need repeated time windows to check stability.

## 7. NextItNet ranking results (3:20-4:05)

For next-item ranking, adapted NextItNet puts the held-out movie in its top ten for nineteen point nine four percent of users. Item-kNN achieves seven point four zero percent, a gap of twelve point five four percentage points. NextItNet also leads on NDCG. On the subset whose test rating comes strictly later, its hit rate is fourteen point three one percent versus five point nine five for item-kNN. This reuses our completed experiment. We have not retrained it for October seventh.

## 8. Rating prediction and next-item ranking (4:05-4:40)

The ranking test holds out each user's last rating for test and the previous one for validation. We select a ten-item context using validation, refit on training plus validation, and remove previously rated movies from candidates. Equal timestamps use seeded tie-breaking. This split can still include other users' later ratings. SVD here optimizes star ratings, so its weak ranking score does not establish that every matrix-factorization method is weak.

## 9. Further checks before a business pilot (4:40-5:00)

Both approaches warrant further offline testing. Compare ranking models on shared global time windows and test cold-user fallbacks. A later randomized pilot could measure movie starts per exposed user. MovieLens has no exposure, revenue or cost data to establish engagement gains or ROI.

