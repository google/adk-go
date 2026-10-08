# OpenAI samples

Two samples, one per OpenAI HTTP API. Both run the same weather agent, so the
diff between them is the endpoint and nothing else.

| Sample | Endpoint | Use it for |
| :---- | :---- | :---- |
| [responses](responses) | `POST /v1/responses` | OpenAI's newer API, and the default in `openaimodel`. Also works with compatible providers that implement it. |
| [completions](completions) | `POST /v1/chat/completions` | OpenAI, and almost any OpenAI-compatible provider. Required for a provider that has no Responses API. |

The endpoint is chosen by `ClientConfig.API`; its zero value is the Responses
API, so existing code keeps the behavior it has. Your provider's documentation
says which of the two it serves.
