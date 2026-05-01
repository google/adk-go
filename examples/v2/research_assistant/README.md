# research_assistant

Real Gemini-backed research assistant that uses **dynamic tool loading**
via `toolregistry`. The agent declares only `list_tools` and `load_tool`
to the LLM upfront; the model discovers what's available and activates
specific tools on demand. Keeps the LLM context lean even when the
catalog is large.

## Run

```
export GOOGLE_API_KEY=...
go run ./examples/v2/research_assistant/             # interactive console
go run ./examples/v2/research_assistant/ web         # adk-web
```

## Try it

> Find a recent overview of the Agent Development Kit and summarize the
> key points.

Watch the model:

1. Call `list_tools(query="search")` → sees `web_search`.
2. Call `load_tool(name="web_search")` → activates it.
3. Call `web_search(query="agent development kit")` → gets results.
4. (Optional) `load_tool(name="fetch_url")` then `load_tool(name="summarize")` for full content + condensation.

Tools `citation_check` and `save_note` stay dormant unless the user's
task surfaces a need for them — they don't bloat every prompt.

## Wiring for production

Replace each tool body in `main.go` with a real implementation (your
search API, fetcher, summarizer LLM call, citation checker, notes
sink). The registry pattern stays identical.
