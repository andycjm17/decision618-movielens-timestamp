# Five-minute presentation script, October 7, 2026

Seven slides, 596 words. Planned slide budgets total 300 seconds. Rehearse aloud to adjust pauses.

## 1. Movie rating prediction (0:00-0:25)

Our project belongs to Media and Entertainment. We ask whether a movie platform can predict a returning user's preferences more accurately. This October seventh version starts with the method taught in class and adds timestamp features. The immediate task is predicting a one-to-five star rating. Recommendation quality and business impact still need separate tests.

## 2. MovieLens and rating activity (0:25-1:00)

MovieLens contains about one million ratings from six thousand users. We checked the source tables, IDs and rating values, and kept missing user-movie pairs unobserved. Timing is useful but easy to misread. Fifty-three percent of adjacent ratings share the same second, and almost eighty-nine percent arrive within a minute. These timestamps describe when someone entered ratings. They cannot establish the order in which that person watched the movies.

## 3. Course baseline and fresh replication (1:00-1:50)

The instructor uses softImpute matrix factorization to predict missing ratings. We reproduced the exact random split and all twenty cross-validation folds, using the classroom NumPy seed of one forty-four. Rescoring the instructor's cache selects rank eight. Four hundred fresh R fits select rank nine, with rank eight close behind. We retain rank eight for the classroom comparison and report the fresh choice separately. Its test RMSE is point nine one nine eight, close to the instructor's point nine one nine. R initialization and package differences prevent a claim of identical numerical results.

## 4. Random completion and future prediction (1:50-2:35)

The split follows the question. The course's ninety-nine-to-one random split tests whether we can fill missing ratings. To test future ratings, we added a global chronological split of roughly eighty, ten and ten percent, keeping equal timestamps together. Validation chooses parameters, and the final fit uses training plus validation before the untouched test period. The ensemble learns from out-of-fold predictions. History features use only earlier fitted entries. Every comparison within a column uses the same holdout. The two columns cover different people and periods.

## 5. Metadata and time improve rating accuracy (2:35-3:40)

Ridge combines CF predictions with movie and user metadata. We compare it with and without time features. Lower RMSE means smaller star-rating errors. On the random test, collaborative filtering scores point nine one nine eight. Metadata lowers that to point eight nine nine zero, and timestamp features lower it again to point eight nine four zero. On the future test, the timestamp ensemble scores one point zero zero zero one, compared with one point zero three zero three for the static ensemble. The controlled timestamp increments are about point zero zero five and point zero three zero two rating points. The future collaborative-filtering model performs worse than the training mean. We kept that result because it matters. These comparisons support the timestamp ensemble on these holdouts. They do not prove better viewing engagement.

## 6. Timestamp features and evaluation limits (3:40-4:25)

The course already discusses date and time. We test their incremental value alongside counts and gaps from strictly earlier ratings. Equal-second entries contribute no prior history. User-level paired bootstrap intervals remain above zero for both timestamp increments. Those intervals are conditional on this split and selection procedure. A limitation appears in the future evaluation: only twenty-four percent of validation ratings are warm, compared with ninety-six percent in the final test. The validation choice therefore transfers between very different cohorts. We need repeated time windows before treating this as a stable deployment result.

## 7. Further checks before a business pilot (4:25-5:00)

The next step is to evaluate Top-N ranking and cold users across additional time windows. If those checks hold, a limited randomized pilot could test recommendation-led movie starts per exposed user, with completion, catalog concentration and latency as guardrails. That is a proposed business test. MovieLens has no exposure logs, revenues or operating costs, so we cannot calculate ROI from these ratings. The current evidence supports further offline testing of the timestamp ensemble.

