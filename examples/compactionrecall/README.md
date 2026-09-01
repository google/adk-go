# Compaction recall

Measures whether a compaction config still lets the agent **answer**, rather
than how small it makes the prompt. Plants checkable facts, buries them under
filler turns until compaction has summarized them away, then asks about each one
and scores whether the value came back.

- **Concept:** Judge a `compaction.Config` on recall, not on prompt size, by
  probing for facts that are known to sit behind a compaction boundary.
- **Needs LLM?** Yes (Gemini)

Related: [`../compaction`](../compaction) — how to *enable* compaction on an
agent. This one is about what enabling it costs you.

## Goal

Every other measurement of compaction reports prompt size, which is easy to
measure and says nothing about whether the conversation survived. Tail retention
seeds each new summary with the previous one, so a value stated early is
recompressed on every pass. Whether it survives is a property of the summarizer
prompt, and the only way to know is to ask the agent afterwards.

Point it at your own settings to find out what they cost you:

```bash
GOOGLE_API_KEY=... go run ./examples/compactionrecall -threshold=32000 -retention=10
```

Two arms run the identical conversation, differing in one instruction and
nothing else, so the comparison isolates that instruction rather than measuring
a third prompt:

| arm | summarizer prompt |
| --- | --- |
| `default` | the shipped prompt, which requires concrete values to be carried forward verbatim |
| `prior` | the same prompt as it was before that instruction was added |

If you are writing your own `PromptTemplate`, add it as an arm and run it here
before trusting it. Asking only for a "concise" summary is enough to reintroduce
the loss.

## Authentication

The model client reads an API key from the environment:

```bash
export GOOGLE_API_KEY=...
```

Vertex AI is not wired up here; pass `-model` to change the model.

## Reading the output

**Read the per-run scores, not the average.** Recall is close to
all-or-nothing: each pass either copies a value forward or generalizes it to its
label — "the deployment region" in place of `europe-west4` — and a value replaced
by its label cannot come back, because the next pass sees only the previous
summary and the retained tail. A config tends to score 8/8 or 0/8, so two configs
that both average 50% can mean "always half-remembers" and "works perfectly half
the time". That is why `-runs` defaults to 3 and each run is printed.

`buried` counts the facts actually behind a compaction boundary at probe time. A
fact still sitting in the raw tail was never summarized, so recalling it measures
nothing; only the buried ones say anything about what compaction cost.

## Flags

| Flag | Default | Purpose |
| --- | --- | --- |
| `-model` | `gemini-flash-latest` | Model for both the conversation and the summarizer. |
| `-threshold` | `700` | `compaction.Config.TokenThreshold`. |
| `-retention` | `3` | `compaction.Config.EventRetentionSize`. |
| `-interval` | `0` | `compaction.Config.CompactionInterval`; 0 leaves the sliding window off. |
| `-fillerx` | `3` | Repeat the filler block this many times, to force more passes. |
| `-runs` | `3` | Repeat each arm this many times. |
| `-arms` | `default,prior` | Which arms to run. |
| `-style` | `incidental` | How facts are stated. See below. |
| `-v` | off | Print every summary and every probe answer. |

### `-style`

`incidental` states each value inside a message whose actual point is something
else, with no cue that it matters, and spreads them through the conversation so
no single compaction window sees them all:

> Quick bit of context before I forget: we finally got the new environment
> provisioned in europe-west4, so the Frankfurt team should see decent latency.
> Anyway, explain in two sentences why connection pooling matters.

`flagged` states each value on its own and follows it with "Note it", and plants
them all up front. That is an easier test, and a fairer one only if your users
really do announce what matters.

It defaults to `incidental` because the easier shape flatters any prompt that
names the categories it should keep, which is exactly what the shipped prompt
does. A fix measured only against flagged facts would be measuring its own cue.

## Running the sample

```bash
# The default: both arms, three runs each.
GOOGLE_API_KEY=... go run ./examples/compactionrecall

# One arm, one run, and show the summaries and answers.
GOOGLE_API_KEY=... go run ./examples/compactionrecall -arms=default -runs=1 -v
```

This calls a real model. One run of one arm is roughly 70 model calls, so the
default of two arms over three runs is a few hundred. A failed run is reported
and skipped rather than discarding the batch, because a busy model returns 503
often enough to matter at that call count.

## Example session

Four runs per arm on `gemini-3.5-flash`, `-style=incidental -fillerx=1`, about 36
compaction passes per run:

```text
  default  run 1: recalled 8/8  buried=8  lostWhileBuried=0  compactions=38
  default  run 2: recalled 8/8  buried=8  lostWhileBuried=0  compactions=38
  default  run 3: recalled 8/8  buried=8  lostWhileBuried=0  compactions=37
  default  run 4: recalled 8/8  buried=8  lostWhileBuried=0  compactions=35
  default  TOTAL 32/32 probes, of which buried 32/32, per-run 8/8 8/8 8/8 8/8, runs losing everything: 0/4

  prior    run 1: recalled 0/8  buried=8  lostWhileBuried=8  compactions=35
           lost: europe-west4, INC-77312, PostGIS, 14 March, Priya, 0.3, tarn-staging-91, 7
  prior    run 2: recalled 0/8  buried=8  lostWhileBuried=8  compactions=35
           lost: europe-west4, INC-77312, PostGIS, 14 March, Priya, 0.3, tarn-staging-91, 7
  prior    run 3: recalled 0/8  buried=8  lostWhileBuried=8  compactions=35
           lost: europe-west4, INC-77312, PostGIS, 14 March, Priya, 0.3, tarn-staging-91, 7
  prior    run 4: recalled 5/8  buried=8  lostWhileBuried=3  compactions=36
           lost: europe-west4, INC-77312, PostGIS
  prior    TOTAL 5/32 probes, of which buried 5/32, per-run 0/8 0/8 0/8 5/8, runs losing everything: 3/4
```

The exact numbers come from a model and vary between runs. Two things in that
output are worth noticing beyond the totals.

The `prior` arm's scores are `0, 0, 0, 5` rather than something near 5/8 each
time — the all-or-nothing shape, and the reason an average would have described
neither outcome.

In the run that partly survived, the three lost values are the three stated
*earliest*. They had been through the most passes, which is the ratchet doing
exactly what the mechanism predicts.

A note on this particular configuration: 35–38 compactions across 32 turns is
more than one per turn, which means `-threshold=700` is small enough that the
summary alone exceeds it and compaction re-triggers constantly. It is fine for
comparing two prompts, since both arms run identically, but do not read a cost
figure off it. The package documentation covers choosing a threshold.
