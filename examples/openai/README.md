# OpenAI samples

Two samples, one per OpenAI HTTP API. Both run the same weather agent, so the
diff between them is the endpoint and nothing else.

| Sample | Endpoint | Use it for |
| :---- | :---- | :---- |
| [responses](responses) | `POST /v1/responses` | OpenAI's newer API, and the default in `openaimodel`. |
| [completions](completions) | `POST /v1/chat/completions` | OpenAI, and providers that implement only this surface: DeepSeek, Groq, Together, Fireworks, Mistral, OpenRouter, Ollama. |

The endpoint is chosen by `ClientConfig.API`; its zero value is the Responses
API, so existing code keeps the behaviour it has.
