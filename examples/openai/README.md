# OpenAI model integration

Runs an ordinary ADK `llmagent` — with a tool — on an **OpenAI** model instead
of Gemini, using the `google.golang.org/adk/v2/model/openai` package (the
`openaimodel.NewModel` constructor). It talks to OpenAI's **Responses API**, so
the same `model.LLM` also serves any endpoint that implements that API — recent
**Ollama**, **LM Studio**, and **vLLM** builds — via a base URL. (Endpoints that
only expose the older Chat Completions API won't work.)

- **Concept:** Swap `gemini.NewModel(...)` for `openaimodel.NewModel(...)`; agents, tools, the runner, and the launcher are unchanged.
- **Needs LLM?** Yes (OpenAI, or an OpenAI-compatible endpoint)

## Goal

Show that ADK is model-agnostic: the OpenAI model plugs into the exact same
`llmagent.New` / launcher wiring as the [quickstart](../quickstart). The only
difference is the constructor —

```go
model, err := openaimodel.NewModel(ctx, "gpt-4o-mini", &openaimodel.ClientConfig{
    APIKey:  os.Getenv("OPENAI_API_KEY"),
    BaseURL: os.Getenv("OPENAI_BASE_URL"), // empty for api.openai.com
})
```

— and everything downstream stays identical. The sample also registers a
`get_weather` function tool to demonstrate that OpenAI tool calling flows
through ADK's normal `functiontool` path.

This mirrors adk-python, where the same idea is expressed with the LiteLLM
wrapper (`LlmAgent(model=LiteLlm(model="openai/gpt-4o"))`); adk-go instead ships
a native OpenAI model that implements `model.LLM` directly.

## Configuration

| Variable          | Required                         | Default       | Purpose                                          |
| ----------------- | -------------------------------- | ------------- | ------------------------------------------------ |
| `OPENAI_API_KEY`  | Yes for api.openai.com           | —             | Your OpenAI API key.                             |
| `OPENAI_BASE_URL` | Yes for a compatible endpoint    | api.openai.com | Base URL of a Responses-API-compatible server.  |
| `OPENAI_MODEL`    | No                               | `gpt-4o-mini` | Model name to serve.                             |

## Running the sample

Against OpenAI:

```bash
export OPENAI_API_KEY=sk-...
go run ./examples/openai/ console
```

Against a local endpoint that implements the Responses API (recent Ollama shown;
no key needed):

```bash
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_MODEL=llama3.1
go run ./examples/openai/ console
```

The console streams tokens by default; add `-streaming_mode none` for
block-at-a-time output.

## Example session

The model calls `get_weather` and relays the result (exact wording varies):

```text
User -> what's the weather in Paris?
Agent -> It is currently 22°C and sunny in Paris.
```

## Notes

The OpenAI model targets the [Responses API]. `Temperature`, `TopP`,
`MaxOutputTokens`, structured output (JSON schema), and system instructions all
work as usual. A few Gemini-style `GenerateContentConfig` knobs are not supported
and return a descriptive error if set: `TopK`, `StopSequences`, multiple
candidates, frequency/presence penalties, request labels, and safety settings.

[Responses API]: https://platform.openai.com/docs/api-reference/responses
