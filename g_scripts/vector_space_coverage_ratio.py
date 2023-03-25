import numpy as np
from itertools import combinations

vals = list(range(5))


def cover_ratio(bag, degree=3):
    if len(bag) <= 1:
        return 0

    vectors = np.array(list(bag))[:, None]
    dists = np.abs(vectors - vectors.T)
    dists = dists[dists != 0].reshape(len(bag), -1)

    # argmin_dists = np.argmin(dists, axis=1)
    # pairs = np.array(list({tuple(sorted([i, v], reverse=True)) for i, v in enumerate(argmin_dists)}))
    #
    # min_dists = dists[pairs[:, 0], pairs[:, 1]]
    min_dists = np.min(dists, axis=1)

    wrap = lambda x: 1 + -1 / (x + 1)

    sum_min_dists = np.sum(wrap(min_dists))

    return sum_min_dists


bag_rate_s = []

for c in range(len(vals) + 1):
    comb = combinations(vals, c)
    for bag in comb:
        bag_rate = (bag, cover_ratio(bag) / 5)
        bag_rate_s.append(bag_rate)

bag_rate_s.sort(key=lambda bag_rate: bag_rate[1])

for line in bag_rate_s:
    print(line)

# bag = np.array((0, 1, 2, 3, 4))
# vectors = np.array(list(bag))[:, None]
# dists = np.abs(vectors - vectors.T)
# dists = dists[dists != 0].reshape(len(bag), -1)
#
# argmin_dists = np.argmin(dists, axis=1)
# pairs = np.array(list({tuple(sorted([i, v], reverse=True)) for i, v in enumerate(argmin_dists)}))
#
# min_dists = dists[pairs[:, 0], pairs[:, 1]]
#
# sum_min_dists = np.sum(np.sqrt(min_dists))
