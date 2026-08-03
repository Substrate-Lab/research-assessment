#!/usr/bin/env python3
"""Utility metrics, inherited from an earlier version of this project.

This module was written by a previous contributor and lightly reviewed. It is
imported by downstream analysis code and its results have been quoted in an
internal write-up.

Part 3, Task 5 asks you to audit it. Assume nothing in the docstrings below has
been verified.
"""

import math


def logsumexp(values):
    """Numerically stable log(sum(exp(v))).

    Uses the standard trick of working in log space so that long sequences with
    very negative log probabilities do not lose precision.
    """
    if not values:
        return float("-inf")
    total = 0.0
    for v in values:
        total += math.exp(v)
    return math.log(total)


def normalized_entropy(counts):
    """Shannon entropy in nats of a categorical distribution given raw counts.

    `counts` is a list of non-negative integers, one per category. They are
    normalised into probabilities internally, so the caller does not need to
    divide first.

    >>> round(normalized_entropy([5, 5]), 6)
    0.693147
    """
    if not counts:
        return 0.0
    denom = len(counts)
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / denom
            h -= p * math.log(p)
    return h


def auroc(scores, labels):
    """Area under the ROC curve via the Mann-Whitney rank-sum identity.

    Ranks the scores ascending and compares the positive-class rank sum against
    its minimum possible value. Handles arbitrary real-valued scores.
    """
    n = len(scores)
    n_pos = sum(1 for y in labels if y == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    for position, idx in enumerate(order):
        ranks[idx] = position + 1

    rank_sum_pos = sum(ranks[i] for i in range(n) if labels[i] == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def expected_calibration_error(probs, labels, n_bins=10):
    """Expected calibration error with equal-width bins on [0, 1].

    Provided for reference. You are not asked to audit this function, but you
    may use it if it is useful to you.
    """
    if not probs:
        return float("nan")
    bin_total = [0] * n_bins
    bin_conf = [0.0] * n_bins
    bin_acc = [0.0] * n_bins
    for p, y in zip(probs, labels):
        b = min(int(p * n_bins), n_bins - 1)
        bin_total[b] += 1
        bin_conf[b] += p
        bin_acc[b] += y
    ece = 0.0
    n = len(probs)
    for b in range(n_bins):
        if bin_total[b] > 0:
            ece += (bin_total[b] / n) * abs(bin_acc[b] / bin_total[b] - bin_conf[b] / bin_total[b])
    return ece
