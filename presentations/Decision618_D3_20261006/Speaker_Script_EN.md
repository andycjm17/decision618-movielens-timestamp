# Five-minute presentation script
Planned slide budgets total 300 seconds. Rehearse aloud with transitions and pauses. Actual timing depends on the presenter.

## 1. The next movie recommendation (0:00-0:25)
Our project asks what a movie platform should recommend to a returning user. We use recent rating history to build a shortlist of ten movies. The business goal is to help people find something they want to start. MovieLens lets us test recommendation relevance offline, although it cannot tell us whether the system increases revenue.

## 2. MovieLens and rating activity (0:25-1:05)
The dataset has about one million ratings from six thousand users. The team notebook has cleaned the source tables and joined the movie and user information. Most user-movie pairs have no rating. We leave those cells unobserved rather than treating them as dislikes. There is also a timing problem: over half of adjacent ratings share the same timestamp. So our sequence represents rating activity, and we cannot describe it as a confirmed viewing sequence.

## 3. Course method and project extension (1:05-1:50)
The class used matrix factorization to predict missing star ratings. It learns a compact representation of each user and movie. Our extension uses NextItNet to predict the next movie a person rates. It reads the recent movie IDs through a causal convolution network, so a prediction cannot see later positions. We adapted the Recommenders encoder to an item-only model. Validation selected ten recent items and eight training epochs. This changes both the model and the prediction target, which matters when we compare results.

## 4. Evaluation setup (1:50-2:30)
For each user, we reserve the last rating for testing and the previous one for validation. All model choices use validation results. We then refit with the validation history included, while keeping the final target out. Each model ranks the same fitted catalog and removes previously rated movies. Two users have test movies outside that catalog, so all methods evaluate the same six thousand and thirty-eight users. Our target includes every rating, even a low star rating.

## 5. NextItNet leads the tested ranking baselines (2:30-3:30)
The key result is hit rate at ten. Popularity finds the held-out movie for about four percent of users. Item-kNN reaches seven point four percent. Adapted NextItNet reaches nineteen point nine four percent, a gain of twelve point five four percentage points over item-kNN. NDCG also improves, which means the correct movie tends to appear closer to the top. The paired bootstrap interval for that improvement stays above zero. We also ran SVD, but it optimizes star ratings, so its weaker ranking result does not establish that sequence models are generally better than matrix factorization.

## 6. A stricter timestamp check (3:30-4:15)
We checked the users whose final rating happened at a strictly later timestamp. This removes an ambiguous ordering at the test boundary. For this smaller group, NextItNet reaches a fourteen point three one percent hit rate, compared with five point nine five percent for item-kNN. Its advantage remains. But the split is still per user, and another user's later ratings can enter training. A global time split and a model retrained on shuffled histories would give us a stronger test of the contribution of sequence order.

## 7. A limited pilot is the next step (4:15-5:00)
We would move next to a limited pilot, after those additional checks. Returning users would receive the personalized row. The demo already gives new users a popularity fallback. New movies would need a metadata route, which connects to the hybrid method discussed in class. We would compare recommendation-led starts in a randomized test and monitor completion, title concentration and latency. MovieLens does not contain the business outcomes or costs needed for ROI. Our conclusion is that this version merits further testing, with privacy controls and a way for users to opt out.
