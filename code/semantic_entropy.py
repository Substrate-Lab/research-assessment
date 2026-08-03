#!/usr/bin/env python3
"""Substrate Labs research assessment -- Part 3, Tasks 1-3.

Fill in the functions marked TODO. Standard library only: no numpy, no scipy,
no sklearn, no network access. Python 3.9+.

Run `python3 semantic_entropy.py` to execute the driver at the bottom, and
`python3 -m unittest discover tests` to run the public tests.

Read the specs carefully. Where the paper leaves a choice open, we have pinned
it down here so that everyone's output is comparable -- Task 4 asks you what
you think of one of those choices.
"""

import json
import math
import os


# --------------------------------------------------------------------------
# Task 1: semantic clustering
# --------------------------------------------------------------------------

def cluster_by_entailment(entailment):
    """Partition generations into semantic clusters by bidirectional entailment.

    `entailment` is an N x N matrix of 0/1 ints where entailment[i][j] == 1
    means "generation i entails generation j". It is NOT symmetric, and the
    relation it encodes is NOT transitive -- it comes from a noisy model.

    Two generations are in the same semantic cluster when they entail each
    other in both directions.

    Because bidirectional entailment is not an equivalence relation here, the
    result depends on the order you process things in. Use exactly this
    convention so results are comparable across candidates:

      - process generations in index order 0, 1, ..., N-1
      - each existing cluster's representative is its LOWEST-index member
      - assign generation i to the FIRST cluster, in cluster-creation order,
        whose representative bidirectionally entails i
      - if no cluster matches, open a new cluster containing only i

    Returns:
        list[list[int]]: clusters, each an ascending list of generation
        indices. Clusters appear in creation order.

    Example:
        >>> cluster_by_entailment([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
        [[0, 1], [2]]
    """
    # TODO: implement
    raise NotImplementedError


# --------------------------------------------------------------------------
# Task 2: entropy estimators
# --------------------------------------------------------------------------

def discrete_semantic_entropy(clusters, n_generations):
    """Plug-in entropy of the cluster proportions, in NATS.

    Estimate p(C_k) as the fraction of generations landing in cluster k, then
    return the Shannon entropy of that categorical distribution:

        H = - sum_k p_k * log(p_k),   p_k = |C_k| / n_generations

    This is the "discrete" variant used by the paper, and it needs no token
    probabilities, so it works on black-box models.

    Returns:
        float: entropy in nats. Exactly 0.0 when every generation is in one
        cluster; log(n_generations) when all are distinct.
    """
    # TODO: implement
    raise NotImplementedError


def logsumexp(values):
    """Numerically stable log(sum(exp(v) for v in values)).

    Must not overflow or underflow for realistic sequence log probabilities,
    which for a 100-token generation run to roughly -200 and below. A naive
    implementation silently returns -inf or raises here; yours must not.

    Args:
        values: list[float], possibly empty.

    Returns:
        float: the log-sum-exp, or float('-inf') for an empty list.
    """
    # TODO: implement
    raise NotImplementedError


def cluster_log_probabilities(clusters, gen_logprobs):
    """[STRETCH, optional] Aggregate member log probs into cluster log probs.

    Skip this and weighted_semantic_entropy if you are short on time. The
    driver and the tests both handle them being left unimplemented.

    p(C_k) = sum over generations s in C_k of p(s), then renormalised across
    the observed clusters so the result is a proper distribution.

    Args:
        clusters: output of cluster_by_entailment.
        gen_logprobs: list[float], one log probability per generation, indexed
            the same way as the entailment matrix.

    Returns:
        list[float]: log p(C_k) for each cluster, in the same order as
        `clusters`. Exponentiating and summing them should give 1.0.
    """
    # TODO: implement
    raise NotImplementedError


def weighted_semantic_entropy(clusters, gen_logprobs):
    """[STRETCH, optional] Entropy of the likelihood-weighted cluster distribution.

        H = - sum_k p(C_k) * log p(C_k)

    with p(C_k) from cluster_log_probabilities. Unlike the discrete variant
    this uses the model's own probabilities rather than raw sample counts.

    Returns:
        float: entropy in nats.
    """
    # TODO: implement
    raise NotImplementedError


def sequence_logprob(token_logprobs):
    """Joint log probability of a generation: the sum of its token log probs."""
    # TODO: implement
    raise NotImplementedError


def length_normalized_logprob(token_logprobs):
    """Mean per-token log probability.

    Raises:
        ValueError: on an empty generation.
    """
    # TODO: implement
    raise NotImplementedError


# --------------------------------------------------------------------------
# Task 3: evaluation
# --------------------------------------------------------------------------

def auroc(scores, labels):
    """Area under the ROC curve. Do not import sklearn -- write it yourself.

    `scores` are real-valued (higher = more likely positive) and `labels` are
    0/1 ints. Equivalent to the probability that a randomly chosen positive
    outscores a randomly chosen negative, with TIED pairs counting one half.

    Tie handling is not optional here: the discrete semantic entropy of ten
    samples takes few distinct values, so ties are common in this data and a
    tie-blind implementation gets a visibly wrong number.

    You do not have to derive anything. The standard recipe:

      1. Sort the indices by score, ascending.
      2. Assign ranks 1..n. For any block of EQUAL scores, give every member
         of that block the average of the ranks the block spans. This is the
         only fiddly step and it is what handles ties.
      3. Let R be the sum of ranks belonging to positive examples, and let
         n_pos and n_neg be the class counts. Then

             AUROC = (R - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    A brute-force double loop over positive/negative pairs, counting 1 for a
    win and 0.5 for a tie, is also completely acceptable and easier to get
    right. n is 300, so speed does not matter.

    Returns:
        float: AUROC in [0, 1], or float('nan') if either class is absent,
        since AUROC is undefined then.
    """
    # TODO: implement
    raise NotImplementedError


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load(name):
    with open(os.path.join(DATA_DIR, name)) as f:
        return json.load(f)


def score_example(example):
    """Compute the uncertainty scores for a single example.

    The two likelihood-weighted scores are stretch work; if you have not
    implemented them, they are simply omitted from the results.
    """
    clusters = cluster_by_entailment(example["entailment"])
    n = len(example["generations"])
    out = {
        "num_clusters": float(len(clusters)),
        "discrete_se": discrete_semantic_entropy(clusters, n),
    }
    try:
        joint = [sequence_logprob(g["token_logprobs"]) for g in example["generations"]]
        normed = [length_normalized_logprob(g["token_logprobs"]) for g in example["generations"]]
        out["weighted_se_joint"] = weighted_semantic_entropy(clusters, joint)
        out["weighted_se_normed"] = weighted_semantic_entropy(clusters, normed)
    except NotImplementedError:
        pass
    return out


def main():
    data = load("generations.json")
    examples = data["examples"]
    labels = [e["is_hallucination"] for e in examples]
    scored = [score_example(e) for e in examples]

    print("eval set: %d examples, %d positive (%.1f%%)"
          % (len(examples), sum(labels), 100.0 * sum(labels) / len(labels)))
    for key in ("num_clusters", "discrete_se", "weighted_se_joint", "weighted_se_normed"):
        if key not in scored[0]:
            print("  AUROC(%-18s) = skipped (stretch, not implemented)" % key)
            continue
        print("  AUROC(%-18s) = %.4f" % (key, auroc([s[key] for s in scored], labels)))


if __name__ == "__main__":
    main()
