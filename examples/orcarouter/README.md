# OrcaRouter model integration

Runs an ordinary ADK `llmagent` — with a tool — on an **OrcaRouter** model
instead of Gemini. OrcaRouter is an OpenAI-compatible AI gateway that routes to
many models behind a single base URL, using the `google.golang.org/adk/v2/model/openai`
package (the `openaimodel.NewModel` constructor). Because OrcaRouter serves the
OpenAI **Responses API**, this sample shares the exact same `model.LLM` wiring as
the [OpenAI example](../openai/README.md); the only difference is the client
config, which points at OrcaRouter's endpoint.

- **Concept:** Swap `gemini.NewModel(...)` for `openaimodel.NewModel(...)` with OrcaRouter's base URL; agents, tools, the runner, and the launcher are unchanged.
- **Needs LLM?** Yes (OrcaRouter API key)

## Goal

Show that ADK is model-agnostic beyond the providers it ships constructors for:
OrcaRouter plugs into the exact same `llmagent.New` / launcher wiring as the
[quickstart](../quickstart) and the [OpenAI example](../openai). The only
difference is the constructor —

```go
model, err := openaimodel.NewModel(ctx, "orcarouter/free", &openaimodel.ClientConfig{
    APIKey:  os.Getenv("ORCAROUTER_API_KEY"),
    BaseURL: "https://api.orcarouter.ai/v1",
})
```

— and everything downstream stays identical. The sample also registers a
`get_weather` function tool to demonstrate that OrcaRouter tool calling flows
through ADK's normal `functiontool` path.

## Configuration

| Variable            | Required            | Default            | Purpose                          |
| ------------------- | ------------------- | ------------------ | -------------------------------- |
| `ORCAROUTER_API_KEY`| Yes                 | —                  | Your OrcaRouter API key (`sk-orca-...`). |
| `ORCAROUTER_MODEL`  | No                  | `orcarouter/free`  | Model name to serve.             |

Any model ID on the gateway works — for example `orcarouter/fusion`,
`openai/gpt-4.1`, `anthropic/claude-opus-4.5`, or `deepseek/deepseek-v4-pro`.

## Running the sample

```bash
export ORCAROUTER_API_KEY=sk-orca-...
go run ./examples/orcarouter/ console
```

To pick a different model:

```bash
export ORCAROUTER_MODEL=orcarouter/fusion
go run ./examples/orcarouter/ console
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

The sample targets the [Responses API]. `Temperature`, `TopP`,
`MaxOutputTokens`, structured output (JSON schema), and system instructions all
work as usual. A few Gemini-style `GenerateContentConfig` knobs are not supported
and return a descriptive error if set: `TopK`, `StopSequences`, multiple
candidates, frequency/presence penalties, request labels, and safety settings.

[Responses API]: https://platform.openai.com/docs/api-reference/responses
