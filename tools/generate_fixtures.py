#!/usr/bin/env python3
"""Generate the offline fixtures for the Substrate Labs programming task.

The candidate-facing coding task must run with no GPU, no API key, and no model
downloads, so we ship pre-computed "model outputs" instead of generating them
from a real LLM. Everything here is SYNTHETIC and is labelled as such in the
data files. The point of the exercise is the estimator and the clustering, not
the language model.

Generative process, per example:

  1. Draw the true number of semantic clusters K ~ p(K), and cluster weights
     w ~ Dirichlet-ish over K components.
  2. Sample N generation assignments c_1..c_N ~ Categorical(w).
  3. Give each cluster a distinct answer, and render each generation as one of
     that answer's paraphrases. Paraphrases are lexically different but
     semantically identical -- that is the whole premise of semantic entropy,
     and it means naive string-matching gives a different answer to entailment
     clustering.
  4. Build the directed entailment matrix E, where E[i][j] == 1 means
     "generation i entails generation j". Ground truth is bidirectional
     entailment within a cluster and none across. Each off-diagonal directed
     entry is then flipped independently with probability EPS, which is what
     makes the observed relation noisy, non-transitive, and order-dependent.
  5. Emit per-token log probabilities loosely consistent with the cluster
     weights, so that length normalisation is a live question.
  6. Draw the hallucination label from a Bernoulli whose rate increases with
     the TRUE latent entropy. The label is therefore predictable from a good
     semantic-entropy estimate but never perfectly so.

Run:  python3 tools/generate_fixtures.py
"""

import json
import math
import os
import random

SEED = 20260802
N_GENERATIONS = 10
EPS = 0.03  # per-entry directed entailment flip probability
N_EVAL = 300
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code", "data")

# (question, [(canonical answer, [paraphrase surface forms])])
QUESTION_BANK = [
    (
        "In which city is the Bundesbank headquartered?",
        [
            ("Frankfurt", ["Frankfurt", "It is in Frankfurt.", "Frankfurt am Main"]),
            ("Berlin", ["Berlin", "The Bundesbank is in Berlin.", "Berlin, Germany"]),
            ("Bonn", ["Bonn", "It's headquartered in Bonn.", "Bonn."]),
            ("Hamburg", ["Hamburg", "In Hamburg.", "Hamburg, Germany"]),
        ],
    ),
    (
        "Who developed the first successful polio vaccine?",
        [
            ("Jonas Salk", ["Jonas Salk", "Salk", "It was Jonas Salk."]),
            ("Albert Sabin", ["Albert Sabin", "Sabin", "Albert Sabin did."]),
            ("Maurice Hilleman", ["Maurice Hilleman", "Hilleman", "Maurice Hilleman."]),
        ],
    ),
    (
        "What year was the transistor invented?",
        [
            ("1947", ["1947", "In 1947.", "It was 1947."]),
            ("1948", ["1948", "In 1948.", "1948."]),
            ("1951", ["1951", "Around 1951.", "1951."]),
            ("1956", ["1956", "In 1956.", "1956."]),
        ],
    ),
    (
        "Which enzyme unwinds the DNA double helix during replication?",
        [
            ("Helicase", ["Helicase", "DNA helicase", "It is helicase."]),
            ("Topoisomerase", ["Topoisomerase", "DNA topoisomerase", "Topoisomerase."]),
            ("Primase", ["Primase", "DNA primase", "Primase does."]),
        ],
    ),
    (
        "What is the capital of Australia?",
        [
            ("Canberra", ["Canberra", "It's Canberra.", "The capital is Canberra."]),
            ("Sydney", ["Sydney", "Sydney.", "It is Sydney."]),
            ("Melbourne", ["Melbourne", "Melbourne.", "It's Melbourne."]),
        ],
    ),
    (
        "Which element has atomic number 74?",
        [
            ("Tungsten", ["Tungsten", "Tungsten (W)", "It is tungsten."]),
            ("Tantalum", ["Tantalum", "Tantalum (Ta)", "Tantalum."]),
            ("Rhenium", ["Rhenium", "Rhenium (Re)", "It's rhenium."]),
        ],
    ),
    (
        "Who wrote the novel 'The Leopard'?",
        [
            ("Giuseppe Tomasi di Lampedusa", ["Giuseppe Tomasi di Lampedusa", "Lampedusa", "Tomasi di Lampedusa"]),
            ("Italo Calvino", ["Italo Calvino", "Calvino", "It was Italo Calvino."]),
            ("Alberto Moravia", ["Alberto Moravia", "Moravia", "Alberto Moravia."]),
        ],
    ),
    (
        "What is the SI unit of magnetic flux?",
        [
            ("Weber", ["Weber", "The weber (Wb)", "It is the weber."]),
            ("Tesla", ["Tesla", "The tesla (T)", "Tesla."]),
            ("Henry", ["Henry", "The henry (H)", "It's the henry."]),
        ],
    ),
    (
        "In which year did the Bretton Woods system collapse?",
        [
            ("1971", ["1971", "In 1971.", "1971."]),
            ("1973", ["1973", "In 1973.", "1973."]),
            ("1944", ["1944", "1944.", "In 1944."]),
        ],
    ),
    (
        "Which protein is the primary target of statins?",
        [
            ("HMG-CoA reductase", ["HMG-CoA reductase", "HMGCR", "It targets HMG-CoA reductase."]),
            ("PCSK9", ["PCSK9", "PCSK9 protein", "PCSK9."]),
            ("ACAT", ["ACAT", "Acyl-CoA cholesterol acyltransferase", "ACAT."]),
        ],
    ),
    (
        "Who proved the incompleteness theorems?",
        [
            ("Kurt Godel", ["Kurt Godel", "Godel", "It was Kurt Godel."]),
            ("Alan Turing", ["Alan Turing", "Turing", "Alan Turing."]),
            ("Alonzo Church", ["Alonzo Church", "Church", "Alonzo Church did."]),
        ],
    ),
    (
        "What is the largest moon of Saturn?",
        [
            ("Titan", ["Titan", "It is Titan.", "Titan."]),
            ("Rhea", ["Rhea", "Rhea.", "It's Rhea."]),
            ("Enceladus", ["Enceladus", "Enceladus.", "It is Enceladus."]),
        ],
    ),
]

DATASET_NAMES = ["trivia_synth", "bio_synth", "nq_synth", "squad_synth"]


def entropy_nats(weights):
    """Shannon entropy in nats of a probability vector."""
    return -sum(w * math.log(w) for w in weights if w > 0.0)


def draw_weights(rng, k, concentration=0.9):
    """Dirichlet(concentration) draw via normalised Gamma variates."""
    raw = [rng.gammavariate(concentration, 1.0) for _ in range(k)]
    total = sum(raw)
    return [x / total for x in raw]


def build_entailment(rng, assignments, eps):
    """Directed entailment matrix with independent per-entry label noise.

    E[i][j] == 1 means generation i entails generation j. The diagonal is
    always 1: a generation always entails itself, and that is the one entry no
    NLI model gets wrong.
    """
    n = len(assignments)
    e = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                e[i][j] = 1
                continue
            truth = 1 if assignments[i] == assignments[j] else 0
            e[i][j] = 1 - truth if rng.random() < eps else truth
    return e


def make_generations(rng, question_idx, assignments):
    """Render each assignment as a paraphrase, with per-token log probs."""
    _, answers = QUESTION_BANK[question_idx]
    gens = []
    for a in assignments:
        canonical, paraphrases = answers[a]
        text = rng.choice(paraphrases)
        # Token count proxy: whitespace tokens, floored at 1.
        n_tok = max(1, len(text.split()))
        # More frequent clusters get slightly higher per-token log probs. The
        # spread is deliberate: it makes length normalisation matter.
        base = -0.25 - 0.55 * a + rng.gauss(0.0, 0.12)
        logprobs = [min(-0.01, base + rng.gauss(0.0, 0.18)) for _ in range(n_tok)]
        gens.append({"text": text, "canonical": canonical, "token_logprobs": [round(x, 6) for x in logprobs]})
    return gens


def make_example(rng, idx):
    question_idx = rng.randrange(len(QUESTION_BANK))
    question, answers = QUESTION_BANK[question_idx]
    max_k = len(answers)
    k = rng.choices(range(1, max_k + 1), weights=[3, 4, 3, 2][:max_k])[0]

    weights = draw_weights(rng, k)
    assignments = rng.choices(range(k), weights=weights, k=N_GENERATIONS)

    h_true = entropy_nats(weights)
    # Hallucination rate rises with true latent entropy. Tuned so the eval set
    # is neither trivially separable nor pure noise.
    p_hall = 1.0 / (1.0 + math.exp(-(4.4 * h_true - 2.6)))
    label = 1 if rng.random() < p_hall else 0

    return {
        "id": "ex_%04d" % idx,
        "dataset": DATASET_NAMES[idx % len(DATASET_NAMES)],
        "question": question,
        "generations": make_generations(rng, question_idx, assignments),
        "entailment": build_entailment(rng, assignments, EPS),
        "is_hallucination": label,
        # Latent state. Present so reviewers can audit the fixtures; candidates
        # are told not to use these fields, and the hidden tests do not need them.
        "_latent": {
            "true_num_clusters": k,
            "true_assignments": assignments,
            "true_entropy_nats": round(h_true, 6),
        },
    }


def build_toy_cases():
    """Small hand-checkable cases so candidates can self-verify.

    These are constructed exactly, with no entailment noise, so the correct
    discrete semantic entropy is known in closed form.
    """

    def clean_matrix(assignments):
        n = len(assignments)
        return [[1 if assignments[i] == assignments[j] else 0 for j in range(n)] for i in range(n)]

    def gen_stub(texts):
        return [{"text": t, "canonical": t, "token_logprobs": [-0.5, -0.5]} for t in texts]

    cases = [
        {
            "id": "toy_unanimous",
            "note": "All 10 generations mean the same thing. One cluster.",
            "assignments": [0] * 10,
            "texts": ["Paris", "It's Paris.", "The capital is Paris."] * 3 + ["Paris"],
            "expected_num_clusters": 1,
            "expected_discrete_se_nats": 0.0,
        },
        {
            "id": "toy_all_distinct",
            "note": "10 mutually contradictory answers. Ten clusters, maximum entropy.",
            "assignments": list(range(10)),
            "texts": ["ans_%d" % i for i in range(10)],
            "expected_num_clusters": 10,
            "expected_discrete_se_nats": math.log(10),
        },
        {
            "id": "toy_even_split",
            "note": "Two equally sized semantic clusters.",
            "assignments": [0] * 5 + [1] * 5,
            "texts": ["Paris", "It's Paris.", "Paris.", "The capital is Paris.", "Paris"]
            + ["Lyon", "It's Lyon.", "Lyon.", "The capital is Lyon.", "Lyon"],
            "expected_num_clusters": 2,
            "expected_discrete_se_nats": math.log(2),
        },
        {
            "id": "toy_skewed",
            "note": "8/1/1 split. Low but non-zero entropy.",
            "assignments": [0] * 8 + [1, 2],
            "texts": ["Titan"] * 8 + ["Rhea", "Enceladus"],
            "expected_num_clusters": 3,
            "expected_discrete_se_nats": -(0.8 * math.log(0.8) + 0.1 * math.log(0.1) * 2),
        },
    ]

    out = []
    for c in cases:
        out.append(
            {
                "id": c["id"],
                "note": c["note"],
                "generations": gen_stub(c["texts"]),
                "entailment": clean_matrix(c["assignments"]),
                "expected_num_clusters": c["expected_num_clusters"],
                "expected_discrete_se_nats": round(c["expected_discrete_se_nats"], 10),
            }
        )
    return out


def main():
    rng = random.Random(SEED)
    examples = [make_example(rng, i) for i in range(N_EVAL)]

    os.makedirs(OUT_DIR, exist_ok=True)

    header = {
        "_README": (
            "SYNTHETIC data generated by tools/generate_fixtures.py. These are NOT real "
            "language model outputs. The generative process is documented in that file. "
            "Fields beginning with an underscore are latent state for auditing; do not "
            "use them in your solution."
        ),
        "seed": SEED,
        "n_generations_per_example": N_GENERATIONS,
        "entailment_flip_probability": EPS,
    }

    eval_path = os.path.join(OUT_DIR, "generations.json")
    with open(eval_path, "w") as f:
        json.dump({**header, "examples": examples}, f, indent=1)

    toy_path = os.path.join(OUT_DIR, "toy_cases.json")
    with open(toy_path, "w") as f:
        json.dump({**header, "cases": build_toy_cases()}, f, indent=1)

    pos = sum(e["is_hallucination"] for e in examples)
    print("wrote %s  (%d examples, %d positive = %.1f%%)" % (eval_path, len(examples), pos, 100.0 * pos / len(examples)))
    print("wrote %s  (%d cases)" % (toy_path, len(build_toy_cases())))


if __name__ == "__main__":
    main()
