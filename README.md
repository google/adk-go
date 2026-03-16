# Agent Development Kit (ADK) for Go

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Go Doc](https://img.shields.io/badge/Go%20Package-Doc-blue.svg)](https://pkg.go.dev/google.golang.org/adk)
[![Nightly Check](https://github.com/google/adk-go/actions/workflows/nightly.yml/badge.svg)](https://github.com/google/adk-go/actions/workflows/nightly.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![View Code Wiki](https://www.gstatic.com/_/boq-sdlc-agents-ui/_/r/YUi5dj2UWvE.svg)](https://codewiki.google/github.com/google/adk-go)

<html>
    <h2 align="center">
      <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256"/>
    </h2>
    <h3 align="center">
      An open-source, code-first Go toolkit for building, evaluating, and deploying sophisticated AI agents with flexibility and control.
    </h3>
    <h3 align="center">
      Important Links:
      <a href="https://google.github.io/adk-docs/">Docs</a> &
      <a href="https://github.com/google/adk-go/tree/main/examples">Samples</a> &
      <a href="https://github.com/google/adk-python">Python ADK</a> &
      <a href="https://github.com/google/adk-java">Java ADK</a> &
      <a href="https://github.com/google/adk-web">ADK Web</a>.
    </h3>
</html>

Agent Development Kit (ADK) is a flexible and modular framework that applies software development principles to AI agent creation. It is designed to simplify building, deploying, and orchestrating agent workflows, from simple tasks to complex systems. While optimized for Gemini, ADK is model-agnostic, deployment-agnostic, and compatible with other frameworks.

This Go version of ADK is ideal for developers building cloud-native agent applications, leveraging Go's strengths in concurrency and performance.

---

## ✨ Key Features

*   **Idiomatic Go:** Designed to feel natural and leverage the power of Go.
*   **Rich Tool Ecosystem:** Utilize pre-built tools, custom functions, or integrate existing tools to give agents diverse capabilities.
*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Go for ultimate flexibility, testability, and versioning.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents.
*   **Deploy Anywhere:** Easily containerize and deploy agents, with strong support for cloud-native environments like Google Cloud Run.

## 🚀 Installation

To add ADK Go to your project, run:

```bash
go get google.golang.org/adk
```

### Anthropic Claude model package

ADK Go now includes an Anthropic provider that implements the same `model.LLM` interface used by Gemini.

```go
import (
    "context"

    anthropicapi "github.com/anthropics/anthropic-sdk-go"
    adkanthropic "google.golang.org/adk/model/anthropic"
)

func buildModel(ctx context.Context) {
    _, _ = adkanthropic.NewModel(ctx, anthropicapi.Model(anthropic.ModelClaudeSonnet4_6), &adkanthropic.Config{
        APIKey: "${ANTHROPIC_API_KEY}",
    })
}
```

See [`examples/anthropic/main.go`](examples/anthropic/main.go) for a full runnable sample.

## 📄 License

This project is licensed under the Apache 2.0 License - see the
[LICENSE](LICENSE) file for details.

The exception is internal/httprr - see its [LICENSE file](internal/httprr/LICENSE).
