# Part 2: Machine learning problems (35%)

Five problems. Problem 1 is general machine learning knowledge. Problems 2 to 5
are about the paper in Part 1.

> **If you have not written Part 1 yet, do that first.** Problems 2, 3 and 5
> point at specific parts of the paper, and knowing which parts we picked will
> steer what you notice when you read it. See the note in [README.md](README.md).

**Show your reasoning.** Most of these have no single right answer, and we are
reading for how you think rather than what you recall. Where a question does
have a right answer, a stated assumption that turns out to be wrong is worth
more than a bare number we cannot follow.

**Where a standard result is needed, we give it to you.** Anything marked "take
this as given" is yours to use without proof, and the derivations that would
otherwise gate these problems are optional stretch parts. What we read for is
what you do with the result.

None of the core parts should need more than about fifteen minutes. If one is
eating your afternoon, you have read more into it than we put there. Write down
your interpretation and move on.

Parts marked **[stretch]** are optional and only ever add to your score. Nobody
is expected to do all of them.

Handwritten and photographed is fine. Order-of-magnitude answers are fine where
you say so. A short table of standard normal quantiles is at the bottom, so you
do not need a stats package for anything here.

---

## Problem 1: Machine learning fundamentals

Eight short questions. **Two to four sentences each**, and no more. We are
checking breadth and whether you have actually trained and evaluated models,
not whether you can recite definitions.

If you do not know one, say so and give your best reasoning. "I have not used
this, but I would expect X because Y" is a genuinely good answer and scores far
better than bluffing.

**(a)** Transformers use LayerNorm rather than BatchNorm. Give the main reason,
and name one practical problem BatchNorm would cause in a language model.

**(b)** You fine-tune a model and validation accuracy comes out *higher* than
training accuracy, consistently, across the whole run. Give two ordinary
explanations that do not involve a bug.

**(c)** What is the KV cache, and how does its memory cost scale as the context
gets longer? Why does that shape the economics of long-context serving?

**(d)** A colleague reports that a new base model scores 15 points higher than
the old one on your benchmark. Name the three things you would check before
believing the model is better.

**(e)** When does LoRA do noticeably worse than full fine-tuning? Give a
concrete situation and say what about it causes the gap.

**(f)** The log probability a model assigns to a piece of text depends on its
tokenizer. Explain why, and give one situation where this silently invalidates a
comparison people commonly make.

**(g)** You have 10,000 labelled examples, one 7B open-weight model, and one
GPU. You could train a linear probe on frozen hidden states, train a LoRA
adapter, or fully fine-tune. Pick one, and defend it against the other two.

**(h)** Your model performs well in training and badly in deployment. How would
you tell whether the inputs changed, or the relationship between inputs and
labels changed? Why does the answer change what you do about it?

---

## Problem 2: How much can you learn from ten samples?

Semantic entropy is estimated by sampling `N` answers, clustering them by
meaning, and taking the entropy of the cluster proportions. The paper uses
`N = 10` (Appendix B.2).

Let `p = (p_1, ..., p_K)` be the true distribution over `K` semantic clusters,
and let `H_hat` be the entropy of the observed cluster proportions.

It is a standard result that

```
E[H_hat] = H - (K - 1) / (2N) + O(1/N^2)
```

so the estimate is biased **low**. **Take this as given.**

**(a)** At `N = 10`, compute the bias for `K = 2` and for `K = 10`. Compare both
against the full range of values `H_hat` can take at `N = 10`.

**(b)** How many samples would you need for the bias at `K = 10` to fall below
`0.05` nats? Compare that to the `N = 10` the paper uses.

**(c)** The paper thresholds semantic entropy into "high" and "low" and trains a
classifier on the resulting binary label (Section 4, Eq. 5). AUROC is invariant
to any strictly increasing transformation of the score.

So does this bias actually matter for the reported results? Argue both
directions and commit to an answer.

This part is the one we care about. The tempting answer is wrong, and so is the
tempting rebuttal to it.

**(d) [stretch]** Derive the formula above. Taylor-expand
`f(u) = -sum_k u_k log u_k` around `p` to second order and use
`E[(n_k/N - p_k)^2] = p_k (1 - p_k) / N`.

---

## Problem 3: What is a linear probe actually measuring?

The paper trains `L2`-regularised logistic regressions on hidden states, using
scikit-learn's **default** regularisation strength, on five concatenated layers.
That is 20480 features for the 7B models and 40960 for the 70B, against 1000 to
2000 training examples, and the features are not standardised. All of that is
stated in the paper; finding where is part of Part 1, not this problem.

**(a)** Probing is a standard interpretability tool with a standard failure
mode: a probe can succeed because the information is there, or because the probe
itself is powerful enough to manufacture it.

Name the control that distinguishes these, and say what result would tell you
the representation genuinely encodes the target.

**(b)** Here is a fact about `L2`-regularised linear models. Multiplying every
feature by a constant `c > 0` and keeping the penalty `lambda` fixed gives
exactly the same fit as leaving the features alone and using a penalty of
`lambda / c^2`. **Take this as given.** So with `lambda` fixed, larger features
mean weaker effective regularisation.

Now: in a pre-norm transformer, each block adds its output back into the
residual stream, so the norm of the hidden state grows substantially with depth,
and nothing rescales it before the probe sees it.

The paper's headline internal result is that probe AUROC rises with layer depth
and peaks in the mid-to-late layers (Figures 2, 3, A.1 to A.4), read as evidence
that later layers encode semantic uncertainty.

Give an alternative explanation of that curve that has nothing to do with what
the layers encode.

**(c)** Design the control that separates the two explanations. Say what you
would change, what you would plot, and what the plot looks like under each
hypothesis. It should be cheap: the hidden states are already on disk.

**(d)** The clustering step that produces the probe's training labels groups two
answers together when an NLI model says they entail each other in both
directions, assigning greedily: process answers in order, put each into the
first existing cluster whose representative matches, otherwise start a new one.

A noisy NLI model makes this relation non-transitive: it can say A matches B and
B matches C while denying A matches C.

Give a concrete three-answer example where processing the answers in a different
order produces a different number of clusters, and therefore a different
entropy. Say what that implies about semantic entropy as a quantity.

**(e) [stretch]** Model the NLI oracle as independently wrong on each ordered
pair with probability `eps`. If all `N` answers really do mean the same thing,
show the probability of correctly returning one cluster is `q^(N-1)` where
`q = (1 - eps)^2`, and evaluate at `N = 10`, `eps = 0.05`. Comment on the number.

---

## Problem 4: AUROC is not the thing you care about

Every headline number in the paper is an AUROC.

**(a) Base rates.** From Table 3, Llama-3-70B answers 88.5% of TriviaQA
questions correctly in the long-form setting, so the hallucination rate is
`pi = 0.115`. Suppose a detector achieves AUROC `= 0.75`.

Assume the binormal equal-variance model: detector scores are `N(0, 1)` on
non-hallucinations and `N(mu, 1)` on hallucinations. Under that model
AUROC `= Phi(mu / sqrt(2))`. **Take this as given.**

  1. Find `mu`.
  2. At a threshold giving 80% recall, compute the false positive rate, then the
     **precision**. The quantile table at the bottom has what you need.
  3. State the result in one sentence a non-specialist would understand, then
     say what it implies about shipping this detector as a filter that withholds
     answers.

**(b) Selection.** This part is about methodology in general, not about any
particular paper.

  1. A hyperparameter (a layer, a learning rate, a threshold) is chosen by
     trying `M` candidates and keeping the best. Each measured AUROC is the true
     value plus independent `N(0, sigma^2)` noise. The largest of `M` such draws
     sits about `sigma * sqrt(2 log M)` above the truth. **Take this as given.**

     Evaluate for `M = 28` and `sigma = 0.02`. Compare with the effect sizes the
     paper reports in Table 2 (2.2 to 10.5 AUROC points), and say in one
     sentence what the comparison means.

  2. Now suppose that hyperparameter is tuned **separately for the proposed
     method and for the baseline**, using in-distribution performance, and the
     headline result is then an out-of-distribution comparison.

     Does that make the procedure sound? Give the argument that it does, then
     the argument that it does not, then say which you believe. The strongest
     answer identifies an **asymmetry** in how the two arms are treated.

     Finish with one sentence: what selection rule would have made the
     comparison sound, at the same compute budget?

**(c) [stretch]** Also general. A paper introduces a training-data filtering
step, reports that it gave a small improvement for the proposed method, and notes
that it did not help the baseline, so the baseline numbers are reported without
it.

What is wrong with this, and what would you need in order to bound its effect on
the headline comparison?

---

## Problem 5: Decoding, inference, and debugging

**(a) Decoding.** Semantic entropy is computed from samples drawn at temperature
`T`, with `top-p = 0.9` and `top-K = 50` (Appendix B.2).

  1. Holding the model and the question fixed, what happens to the estimated
     semantic entropy as `T -> 0`, and as `T` becomes large?
  2. Finish this sentence precisely: "semantic entropy is a property of ___".
     It is one sentence and it changes how the paper's claims should be read.
  3. The paper defines semantic entropy over the model's full distribution
     (Eqs. 1 to 3) but samples from a truncated one. Does truncation push the
     reported entropy up or down, and does it affect confident and uncertain
     questions equally?

**(b) Cost.** The paper's selling point is that probes avoid the 10x cost of
sampling ten generations per query.

  1. Where does that 10x actually go at inference time? Be specific about which
     part of the computation repeats.
  2. Building the probe's training set still requires semantic entropy labels.
     So what has actually been saved, and what has not? Is the paper's cost
     claim fair?

**(c) Debugging.** A colleague reports: a semantic entropy probe gets 0.79 AUROC
in distribution, but 0.50, chance, on a new dataset. The same pipeline gets 0.74
on that dataset when the probe is trained on it directly. No errors, no NaNs.

Give your four most likely hypotheses, ordered by how cheap they are to rule
out. For each, state the single observation that would discriminate it. Be
specific about what you would print or plot.

**(d) [stretch]** Cluster probabilities are formed by summing member
probabilities in log space. Doing that naively as
`log(sum(exp(logp)))` breaks on long generations, because `exp` underflows to
zero below about `-745` in float64. At roughly `-2` nats per token, at what
generation length does this start failing? Is it an actual bug at the lengths
the paper uses, or a latent one? Write the stable version.

---

## Standard normal quantiles

| `p` | `Phi^{-1}(p)` |
|---|---|
| 0.55 | 0.1257 |
| 0.60 | 0.2533 |
| 0.65 | 0.3853 |
| 0.70 | 0.5244 |
| 0.75 | 0.6745 |
| 0.80 | 0.8416 |
| 0.84 | 0.9945 |
| 0.90 | 1.2816 |
| 0.95 | 1.6449 |

And `Phi(0.1123) ≈ 0.5447`, `Phi(0.25) ≈ 0.5987`, `Phi(0.5) ≈ 0.6915`.
