# Logistics and ground rules

## Time

Budget roughly 4.5 hours. You have one week from the day you receive this.

That is the core. The stretch items scattered through each part are genuinely
optional and a submission that skips every one of them can still score in the
top band.

If life happens and you need longer, email us and ask. An extension costs you
nothing. We have never held one against a candidate. What we cannot assess is a
submission that was clearly written in the last two hours before the deadline.

If you run out of time, submit what you have and add a short note saying what
you would have done with more. That note is worth real credit. Knowing where you
stopped and why is a research skill.

## Using AI tools

**You may use any AI tool you want**, including for the writing, the maths, and
the code. We use them daily. Pretending otherwise in 2026 would be silly, and an
assessment that depends on you not using them is measuring the wrong thing.

There is one hard requirement:

> **Disclose what you used and where.** Add a short `AI-USE.md` to your
> submission. A few lines is fine: which tools, on which parts, and roughly how
> much of the final text or code came from them.

And one standard you are held to:

> **You own every claim you submit.** If your critique asserts that Figure 4
> shows something, and it does not, that is your error regardless of what
> produced the sentence. We check specific claims against the paper.

This is not a trap, and disclosure does not lower your score. We have hired
people who used models heavily throughout. What we are screening out is
submissions the candidate cannot defend, because the next stage is a
conversation where we ask you to go deeper on the thing you wrote.

### On not making things up

One rule we hold ourselves to internally, and which matters more here than in
most labs, given what we work on:

**Do not report a number you did not compute.** If you need a quantity you could
not get, write `[TODO: not computed]` and say what you would have needed. An
honest gap is fine. A plausible-looking fabricated figure is the single fastest
way to fail this assessment, and it is the one thing we check hardest.

The same goes for citations. If you cite a paper, you should have opened it.

## What to submit

A single archive, or a link to a private repo, containing:

```
your-name/
├── AI-USE.md               # required, see above
├── 01-critique.md          # or .pdf
├── 02-problems.md          # or .pdf; scans of handwritten work are fine
├── 03-code/                # your copy of code/, with your changes
│   ├── semantic_entropy.py
│   ├── RESULTS.md          # Part 3, Tasks 4 and 6
│   ├── AUDIT.md            # Part 3, Task 5
│   └── tests/
└── NOTES.md                # optional: assumptions, what you would do next
```

Markdown or PDF both fine. LaTeX is welcome but not expected. Handwritten and
photographed maths is completely acceptable as long as it is legible. Do not
spend time on formatting; spend it on content.

Email the archive or link to **research@substrate-labs.org** with the subject
line `Assessment: <your name>`.

## How we review

Two reviewers score each submission independently against a fixed rubric, then
reconcile. We read Part 1 with the paper open and check specific claims.

We know this is a real time investment, so: every candidate who submits gets
written feedback, whatever the outcome. Usually a paragraph per part, saying
what landed and what did not.

## Accessibility

Part 3 runs on any laptop. Standard library Python 3.9+, no GPU, no API keys, no
network access, no downloads, nothing to install. If any part of this assessment
is difficult for you for reasons of access or accommodation, email us. We will
adapt it. That has no bearing on how we score you.
