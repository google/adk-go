---
name: adk-sample-creator
description: Author or rework a runnable example under `examples/` in the ADK Go repository — the directory, `main.go`, the README with its Mermaid diagram and real transcript, and the index row. Use when adding a sample for a feature (workflow graph, tool, registry client, model backend, server), when asked to "add an example" for something, or when bringing an existing example up to the repo's standard.
---

# ADK Go sample creator

An example earns its place by teaching **one** thing faster than the package
docs can. It must run. Two families set the standard, and which one you are
writing decides several of the choices below:

- **Engine samples** — `examples/workflow/*`. They exercise ADK itself, need
  nothing beyond the binary and sometimes a model, and their subject is the
  shape of a graph.
- **Integration samples** — `examples/agentregistry/*`. They talk to a live
  external service, so they carry cloud prerequisites, credentials, failures
  worth unwrapping, and output that depends on the reader's own project.

Copy the shape of the nearer family rather than inventing one.

`examples/README.md` documents an older launcher API (`ParseAndRun`,
`FormatSyntax`). Follow the templates here, not that file.

## Layout

```
examples/<area>/<name>/
├── main.go
└── README.md
```

`lower_snake` directory names (`hitl_simple`), nested by theme once a family
grows (`routing/llm`, `dynamic/hitl`). When the area keeps an index table —
`examples/workflow/README.md` and `examples/agentregistry/README.md` do — add a
row with the same columns as its neighbours, including `LLM?`. Most areas keep
none, and none is needed.

## main.go

Copy [`templates/main.go.tmpl`](templates/main.go.tmpl): it carries the Apache
2.0 header the `goheader` linter demands, the package doc line, and the launcher
wiring. `context.Background()` is correct there — `main` is an entrypoint, and
it stays banned in library code.

`full.NewLauncher()` serves `console`, which is the default, plus `web` and its
`api`, `a2a`, `webui`, `pubsub` and `eventarc` sublaunchers. When anything is
deferred, move the body into `run(ctx) error` and let `main` do the `log.Fatal`.

Each of these has cost a review round:

- **Pick a versioned model, not a `-latest` alias**, in any sample whose README
  documents the Vertex AI path: Vertex does not serve the aliases, so the
  documented instructions would fail at model construction. `gemini-3.5-flash`
  answers on both backends.
- **A required variable is read plainly and checked for empty**, failing with a
  message that names it; `cmp.Or(os.Getenv("X"), "default")` is for the optional
  ones, and hides a missing value if you reach for it first. Watch for a
  variable a neighbouring library also reads: `GOOGLE_CLOUD_LOCATION` steers
  genai *and* the Agent Registry client, so a region meant for the model
  silently moves the catalog lookup.
- **`log.SetFlags(0)` when the output is a transcript.** Otherwise every line
  carries a timestamp, and a README that omits it is showing output the code
  cannot produce.
- **`log.Fatal` only from `main`.** Inside a body that defers a close, it skips
  the defer.
- **Render API failures as something actionable** — unwrap the typed error and
  keep the part that identifies the fix (see `explain` in the `agentregistry`
  samples), rather than printing the service's whole JSON envelope.
- Comments say WHY. Never narrate the next line.

## README

Copy [`templates/README.md.tmpl`](templates/README.md.tmpl) and fill the
placeholders in; it spells out what each section is for. The order is fixed —
title and the `Concept` / `Needs LLM?` block, `Goal`, `Requirements`, `How it
works`, `Running the sample`, `Example session`, `What it shows`, `Notes` — but
`Requirements`, `What it shows` and `Notes` come out when they have nothing to
say.

Older examples call the third section `Authentication` and the fourth
`Workflow`; leave those alone. Use the template's names for anything new, with
one exception: an engine sample really is built on a workflow graph, so
`Workflow` stays the truer heading there, over a node-graph diagram rather than
a sequence. An integration sample is the case the template is cut for —
`Requirements` earns its place because an enabled API and an IAM role are not
credentials, and the diagram is a sequence because the interesting part is what
happens once, at startup, versus what happens on every turn.

### Mermaid

Mermaid is the repo default; no ASCII art. Reach for `sequenceDiagram` first: a
sample is a sequence of calls, and the ordering is usually the point — a flow
chart that mixes startup with the first user turn hides that resolution happened
once, before the agent existed. Use `graph LR`/`graph TD` when the subject
really is a topology, such as an engine sample's node graph.

### The transcript is evidence

- Paste output you actually got. Never invent it, never tidy it up.
- Say where it came from and which part generalizes: *"Real output from a
  project with the Cloud Logging API enabled, abridged in the middle. Your logs
  will differ."*
- Abridge with `...`, and say that you did.
- When the sample is deterministic, say so: *"No model runs here, so the output
  is exactly this."* That is the answer to "is this reproducible?" — a question
  reviewers do ask.
- Anonymize project IDs to `my-project`; leave generated resource IDs
  structurally intact.

### Prose

Write so a reviewer never has to read a sentence twice. Do not build an
antithesis whose second half points back at a noun two clauses ago — state the
rule, then the exception, in separate sentences. And state each fact once: the
same explanation in a doc comment, the How it works section and the Notes is two
copies too many.

## Before you call it done

From the repo root, with a workspace (`test -f go.work || go work init && go
work use -r .`):

```bash
go build -mod=readonly work
go vet ./examples/...
golangci-lint run
go mod tidy -diff
```

Then **run the sample** and paste what it printed into the README. Examples are
part of the root module, so one that does not compile breaks CI for everyone.

Final check:

- [ ] row added to the area's index table, if it keeps one
- [ ] every environment variable the code reads is in the table
- [ ] the transcript is real output, and says what about it generalizes
- [ ] license header present, `golangci-lint run` clean
- [ ] it runs
