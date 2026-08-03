# Part 1: Paper critique (40%)

## The paper

**Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs**
Kossen, Han, Razzak, Schut, Malik, Gal (OATML, Oxford), 2024.
arXiv:2406.15927 &mdash; https://arxiv.org/abs/2406.15927

Free, open access, 22 pages including appendices. **Read all of it, appendices
included.** The experimental details are not in the main body.

### Required background

The method it builds on:

**Detecting hallucinations in large language models using semantic entropy**
Farquhar, Kossen, Kuhn, Gal. *Nature* 630, 625-630 (2024). Open access:
https://www.nature.com/articles/s41586-024-07421-0

You do not need to read the Nature paper end to end, but you do need to
understand what semantic entropy is and how it is computed, because the paper
under review inherits those choices. Section 3 of the SEP paper summarises it.
Knowing which weaknesses are inherited and which are introduced is part of the
exercise.

### One-paragraph orientation

Semantic entropy (SE) detects hallucinations by sampling several answers to a
question, clustering them by meaning, and measuring the entropy of the cluster
distribution. A model that keeps saying the same thing in different words is
confident; one that scatters across meanings is probably confabulating. It works
well but costs 5 to 10 extra generations per query. Semantic entropy probes
(SEPs) try to get the benefit without the cost: train a linear probe on the
model's hidden states to predict SE directly, from a single forward pass. The
paper's headline claim is that SEPs beat probes trained to predict answer
correctness, particularly when generalising to tasks the probe was not trained
on.

## What to write

Around **1100 to 1400 words**. Structure it as below. Word counts are guidance,
not limits.

Keep it tight. We would rather read 1100 words that all do work than 2000 with
padding, and we are not scoring length.

### 1. Steelman (~150 words)

State the strongest version of the paper's contribution. Not a summary: the best
case a smart advocate would make, including why the design choices are
reasonable given the constraints the authors faced.

We ask for this first on purpose. A critique that has not first understood why
the authors did what they did is usually wrong, and always less useful. If you
find nothing worth defending here, you have misread the paper.

### 2. The load-bearing claim (~100 words)

Identify the single claim that the contribution rests on: the one that, if
false, means the paper does not have a result. State what evidence is offered
for it, and where.

Papers usually make many claims. Ranking them is the skill.

### 3. Three weaknesses (~600 words, ~200 each)

Your three strongest criticisms. For each:

- **Where.** Section, figure, table, equation, or appendix. Be specific enough
  that we can turn to it.
- **What.** What is wrong, or unsupported, or not what it appears to be.
- **So what.** How much does this change the conclusion? Does it weaken the
  claim, rescope it, or kill it? Be honest when the answer is "weakens it
  slightly".
- **Confidence.** How sure are you, and what would settle it?

Order them by how much they matter, and say why you ordered them that way.

Quality bar: a weakness we can verify by looking where you point, that we had
not already assumed from the abstract. Three real ones beat ten gestures.

**Anti-patterns.** These score zero, because they apply to almost any paper and
show only that you skimmed:

- "Only evaluated on English"
- "Only tested on small / open models"
- "No theoretical guarantees"
- "Needs more datasets" (unless you say which and why *those*)
- "Limited discussion of ethical implications"
- Restating a limitation the authors already state, without adding to it

If one of these is genuinely among the three most important problems with the
paper, you may make it, but then you have to do the work: say concretely what
breaks, and why it matters more than the alternatives you passed over.

### 4. The experiment you would run (~250 words)

Design one experiment that would falsify or substantially strengthen the
load-bearing claim. Specify:

- The hypothesis, stated so it could come out false
- What you would run: data, models, conditions, the control
- What result would change your mind, quantitatively, decided in advance
- Rough compute cost, and whether it fits in a week on 2 A100s

The bar is that we could hand it to someone and they could start on Monday.
"Test on more models" is not an experiment. The control is the part most people
skip, and it is the part we read most closely.

### 5. [optional] What you would build on (~100 words)

Only if you have time. Assume the result is basically right: what is the most
interesting thing it opens up, and why that?

Skipping this costs you nothing.

## Practical notes

- Cite by location, not by vibe. "Section 5, Table 3" beats "somewhere they
  mention".
- Quote sparingly and exactly. If you quote, keep it short and make sure it says
  what you claim it says.
- If you find something you are unsure about, say so and explain the ambiguity.
  Finding a genuine ambiguity in a paper is a real result, and "this is either a
  typo or a serious problem, here is how to tell" is a strong sentence.
- You are allowed to conclude the paper is broadly correct and useful. A
  well-argued positive review with three real, bounded caveats will score
  perfectly well. We are testing your reading, not your hostility.
