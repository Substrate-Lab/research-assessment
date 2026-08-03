# Substrate Labs Research Assessment

Thanks for your interest in Substrate Labs. This is our technical assessment for
research roles.

We are a small, independent, non-profit AI research lab. Our through-line is
*understanding the structure of intelligence*, and two threads recur across our
work: getting models to know what they do not know, and reading structure out of
systems. This assessment is built on a paper that sits squarely in the first
thread, so it doubles as a look at the kind of problem you would actually work
on here.

This repository is public and read-only. Fork or download it, work locally, and
send us your submission as described in [00-logistics.md](00-logistics.md).
Everyone gets the same questions.

## What this is

Three parts. All three are about the same paper and the same ideas, approached
from different angles.

| Part | What | Weight | Time |
|---|---|---|---|
| [1. Paper critique](01-paper-critique.md) | Read a paper, argue with it in writing | 40% | ~1.75h |
| [2. ML problems](02-written-problems.md) | Machine learning fundamentals, plus four problems on the paper | 35% | ~1.75h |
| [3. Programming](03-programming.md) | Implement the paper's method, audit inherited code | 25% | ~1.25h |

Budget around **4.5 hours total**. You have **one week** from receipt. Work at
whatever pace suits you.

> **Do Part 1 before you read Parts 2 and 3.**
>
> Parts 2 and 3 quote specific passages from the paper, including two from the
> appendix. Reading them first will steer what you notice, and Part 1 is meant
> to be your own reading of the paper.
>
> This is on your honour and it is genuinely in your interest. We know which
> passages Part 2 discloses, and a critique built on exactly those, with nothing
> added, reads very differently from one that found them independently. Doing
> Part 1 first is how you get credit for what you actually saw.

Every part has a **core** and, in places, clearly labelled **stretch** items.
Stretch items only ever help you. A submission that does the core well and skips
every stretch item is a strong submission. We would much rather read three
sections done carefully than five done in a hurry.

## What we are actually looking for

We are not checking whether you can recall a definition. We are trying to find
out how you think when a problem is underspecified, which is the normal
condition of research.

Concretely, we score:

- **Correctness.** Does the maths work, does the code run, are the claims true.
- **Depth over breadth.** One weakness traced to its consequences beats six
  listed in a sentence each.
- **Specificity.** "The evaluation is weak" tells us nothing. "Table 1 reports
  a standard error over four tasks, and for Llama-2-70B long-form the interval
  is wider than the effect" tells us a lot.
- **Calibration.** Say how confident you are, and say what would change your
  mind. Confident wrongness scores far worse than flagged uncertainty.
- **Judgement.** Given limited compute and time, what would you actually do
  next, and why that rather than the obvious thing.

Things that do not help: length, jargon, hedging everything, or listing generic
limitations that could be pasted into a review of any paper ("only English",
"small models", "no theoretical guarantees"). We have read those. They tell us
you skimmed.

## Ground rules

Read [00-logistics.md](00-logistics.md) before you start. It covers the AI tool
policy, which is permissive but has one hard requirement, and how to submit.

## Questions

If something is ambiguous, make a reasonable assumption, **write the assumption
down**, and keep going. Noticing an ambiguity and resolving it explicitly is
worth more to us than a clarifying email. If something is genuinely blocking,
email us and we will answer within a day.

Good luck. We hope some of this is fun.
