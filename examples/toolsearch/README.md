# Tool search

An agent with a catalog of six tools that declares only two of them to the
model up front: `get_current_time` and `search_tools`. The other five are
declared from the model step after `search_tools` returns them.

- **Concept:** wrap a toolset with `toolsearch.New` so large catalogs stay out
  of the initial request.
- **Needs LLM?** Yes. Set `GOOGLE_API_KEY`.

## Why

Every declared tool is sent in every model request. With a few MCP servers
attached that can be tens of thousands of input tokens before the model reads
the user's message, and the model picks the right tool less reliably as the list
grows. Tool search keeps those tools available while declaring only the ones the
conversation has needed so far.

## How it works

1. `toolsearch.New(catalog, cfg)` returns a toolset that exposes
   `search_tools` plus `cfg.CoreToolNames`.
2. The model calls `search_tools` with a task description, an exact
   `select:name1,name2` list, or a regex. It gets back names and short
   descriptions, not schemas.
3. The matching names are recorded in session state for this agent, and the
   matching tools are declared from the next model step on. The model calls
   them without a new user message.
4. Discovered tools stay declared for the rest of the session.

`GatedToolNames` is optional. It lists the gated names in the `search_tools`
description, so the model can select a tool by exact name. Leave it out when the
list is too long or changes often.

## Running the sample

```bash
go run ./examples/toolsearch console
```

## Example session

From a run against `gemini-flash-lite-latest`. The console prints only the
agent's text. The tool calls are added from the same run and abridged.

```text
User -> Multiply 6 by 7.
  call search_tools {query: multiply numbers}
  resp search_tools {matches: [multiply_numbers, add_numbers]}
  call multiply_numbers {a: 6, b: 7}
  resp multiply_numbers {value: 42}
Agent -> 6 multiplied by 7 is 42.
User -> What time is it?
  call get_current_time {}
Agent -> The current time is 2026-10-05T12:05:30Z.
User -> How many words are in 'the quick brown fox'?
  call search_tools {query: count words}
  call count_words {text: the quick brown fox}
Agent -> There are 4 words in 'the quick brown fox'.
```

`get_current_time` is a core tool, so it is called without a search.

## Limitations

- In live (bidirectional streaming) mode the agent resolves its tools once per
  session, so tools discovered during it are declared only in the next session.
- An agent can use one tool search toolset. Two would each expose a tool named
  `search_tools`. Put every searchable tool in one base toolset instead.
- Discovery is not authorization. Filter the base catalog for the caller before
  wrapping it.
