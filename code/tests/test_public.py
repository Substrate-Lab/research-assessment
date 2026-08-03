"""Public tests for Part 3. Run: python3 -m unittest discover tests

These are a self-check, not the full grading set. Passing every test here does
not mean your implementation is correct -- it means it is not obviously wrong.
We run additional tests when reviewing. You are encouraged to add your own;
tests you write are part of what we assess.
"""

import json
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import semantic_entropy as se  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def load(name):
    with open(os.path.join(DATA_DIR, name)) as f:
        return json.load(f)


def parts(clusters):
    """Normalise the partition so container choice does not matter.

    Returning tuples instead of lists is fine. We grade the partition, not the
    container type.
    """
    return [list(c) for c in clusters]


def _stretch_implemented():
    """True when the optional likelihood-weighted functions are filled in."""
    try:
        se.cluster_log_probabilities([[0]], [-1.0])
        se.weighted_semantic_entropy([[0]], [-1.0])
    except NotImplementedError:
        return False
    except Exception:
        return True   # implemented but misbehaving: let the real test report it
    return True


STRETCH = _stretch_implemented()


class TestClustering(unittest.TestCase):
    def test_docstring_example(self):
        clusters = parts(se.cluster_by_entailment([[1, 1, 0], [1, 1, 0], [0, 0, 1]]))
        self.assertEqual(clusters, [[0, 1], [2]])

    def test_singleton(self):
        self.assertEqual(parts(se.cluster_by_entailment([[1]])), [[0]])

    def test_all_distinct(self):
        n = 4
        m = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        self.assertEqual(parts(se.cluster_by_entailment(m)), [[0], [1], [2], [3]])

    def test_one_way_entailment_does_not_merge(self):
        # 0 entails 1 but 1 does not entail 0: they must stay apart.
        m = [[1, 1], [0, 1]]
        self.assertEqual(parts(se.cluster_by_entailment(m)), [[0], [1]])

    def test_non_transitive_chain_is_not_closed(self):
        # 0<->1 and 1<->2, but 0 and 2 do not mutually entail. The relation is
        # not transitive, so the greedy pass in the spec must NOT take the
        # transitive closure: generation 2 is compared against cluster [0, 1]'s
        # representative, which is 0, and 0 does not entail it.
        # If you reached for union-find, this is where it shows up.
        m = [[1, 1, 0],
             [1, 1, 1],
             [0, 1, 1]]
        self.assertEqual(parts(se.cluster_by_entailment(m)), [[0, 1], [2]])

    def test_every_generation_assigned_exactly_once(self):
        for case in load("toy_cases.json")["cases"]:
            clusters = parts(se.cluster_by_entailment(case["entailment"]))
            flat = sorted(i for c in clusters for i in c)
            self.assertEqual(flat, list(range(len(case["generations"]))), case["id"])


class TestEntropy(unittest.TestCase):
    def test_toy_cases(self):
        for case in load("toy_cases.json")["cases"]:
            clusters = parts(se.cluster_by_entailment(case["entailment"]))
            self.assertEqual(len(clusters), case["expected_num_clusters"], case["id"])
            h = se.discrete_semantic_entropy(clusters, len(case["generations"]))
            self.assertAlmostEqual(h, case["expected_discrete_se_nats"], places=9, msg=case["id"])

    def test_unanimous_is_exactly_zero(self):
        self.assertEqual(se.discrete_semantic_entropy([[0, 1, 2]], 3), 0.0)

    def test_uniform_is_log_n(self):
        clusters = [[i] for i in range(8)]
        self.assertAlmostEqual(se.discrete_semantic_entropy(clusters, 8), math.log(8), places=12)


class TestLogSumExp(unittest.TestCase):
    def test_matches_naive_on_safe_values(self):
        vals = [-1.0, -2.0, -3.0]
        self.assertAlmostEqual(se.logsumexp(vals), math.log(sum(math.exp(v) for v in vals)), places=12)

    def test_empty(self):
        self.assertEqual(se.logsumexp([]), float("-inf"))

    def test_single(self):
        self.assertAlmostEqual(se.logsumexp([-42.0]), -42.0, places=12)

    def test_survives_long_form_magnitudes(self):
        # A ~400-token generation at about -2 nats/token. A naive implementation
        # underflows to zero here and then raises on log(0).
        vals = [-800.0, -801.0, -802.5]
        got = se.logsumexp(vals)
        self.assertTrue(math.isfinite(got))
        self.assertAlmostEqual(got, -799.6284609681, places=6)

    def test_shift_invariance(self):
        vals = [-3.0, -5.0, -1.5]
        shift = 500.0
        a = se.logsumexp(vals)
        b = se.logsumexp([v - shift for v in vals]) + shift
        self.assertAlmostEqual(a, b, places=9)


@unittest.skipUnless(STRETCH, "optional stretch functions not implemented")
class TestClusterProbabilities(unittest.TestCase):
    def test_normalises_to_one(self):
        clusters = [[0, 1], [2]]
        logps = [-1.0, -2.0, -0.5]
        out = se.cluster_log_probabilities(clusters, logps)
        self.assertAlmostEqual(sum(math.exp(x) for x in out), 1.0, places=12)

    def test_single_cluster_is_certain(self):
        out = se.cluster_log_probabilities([[0, 1, 2]], [-1.0, -2.0, -3.0])
        self.assertAlmostEqual(math.exp(out[0]), 1.0, places=12)

    def test_weighted_entropy_zero_for_single_cluster(self):
        self.assertAlmostEqual(se.weighted_semantic_entropy([[0, 1]], [-1.0, -2.0]), 0.0, places=12)

    def test_weighted_entropy_within_bounds(self):
        clusters = [[0], [1], [2]]
        logps = [-1.0, -1.3, -0.8]
        h = se.weighted_semantic_entropy(clusters, logps)
        self.assertGreaterEqual(h, 0.0)
        self.assertLessEqual(h, math.log(3) + 1e-12)


class TestLogProbHelpers(unittest.TestCase):
    def test_sequence_logprob(self):
        self.assertAlmostEqual(se.sequence_logprob([-1.0, -2.0, -0.5]), -3.5, places=12)

    def test_length_normalized(self):
        self.assertAlmostEqual(se.length_normalized_logprob([-1.0, -2.0, -3.0]), -2.0, places=12)

    def test_length_normalized_rejects_empty(self):
        with self.assertRaises(ValueError):
            se.length_normalized_logprob([])


class TestAUROC(unittest.TestCase):
    def test_perfect_separation(self):
        self.assertAlmostEqual(se.auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]), 1.0, places=12)

    def test_perfectly_inverted(self):
        self.assertAlmostEqual(se.auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]), 0.0, places=12)

    def test_all_scores_tied_is_one_half(self):
        self.assertAlmostEqual(se.auroc([1.0] * 6, [1, 0, 1, 0, 1, 0]), 0.5, places=12)

    def test_partial_ties(self):
        # Worked by hand: 2 positives, 2 negatives, one tied pair.
        self.assertAlmostEqual(se.auroc([1.0, 1.0, 2.0, 0.0], [1, 0, 1, 0]), 0.875, places=12)

    def test_single_class_is_nan(self):
        self.assertTrue(math.isnan(se.auroc([0.1, 0.2], [1, 1])))
        self.assertTrue(math.isnan(se.auroc([0.1, 0.2], [0, 0])))

    def test_invariant_to_monotone_rescaling(self):
        scores = [0.3, 0.1, 0.9, 0.44, 0.7]
        labels = [0, 0, 1, 1, 0]
        a = se.auroc(scores, labels)
        b = se.auroc([5.0 * s + 2.0 for s in scores], labels)
        self.assertAlmostEqual(a, b, places=12)


if __name__ == "__main__":
    unittest.main()
