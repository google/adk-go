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


## FAQ

### What is Agent Development Kit (ADK) for Go?

ADK is Google's open-source, code-first Go toolkit for building, evaluating, and deploying sophisticated AI agents. It applies software development principles to AI agent creation, designed for cloud-native applications leveraging Go's concurrency and performance strengths.

### How does ADK compare to other agent frameworks?

| Feature | ADK Go | LangChain Go | CrewAI | AutoGen |
|---------|--------|--------------|--------|---------|
| Language | Go | Go | Python | Python |
| Code-First | ✅ Yes | ✅ Yes | ❌ Config | ❌ Config |
| Multi-Agent | ✅ Modular | ✅ Chains | ✅ Crews | ✅ Teams |
| Model Agnostic | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| Optimized for Gemini | ✅ Yes | ❌ No | ❌ No | ❌ No |
| Cloud-Native | ✅ Cloud Run | ⚠️ Limited | ❌ No | ❌ No |
| Deployment Agnostic | ✅ Yes | ✅ Yes | ❌ No | ❌ No |

### What are the key features?

| Feature | Description |
|---------|-------------|
| **Idiomatic Go** | Designed to feel natural, leveraging Go's strengths |
| **Rich Tool Ecosystem** | Pre-built tools, custom functions, tool integration |
| **Code-First Development** | Logic, tools, orchestration in Go code |
| **Modular Multi-Agent** | Compose multiple specialized agents |
| **Deploy Anywhere** | Containerize and deploy (Cloud Run support) |
| **Model Agnostic** | Optimized for Gemini, works with other models |
| **Testable & Versioned** | Software engineering best practices |

### How do I install ADK Go?

```bash
go get google.golang.org/adk
```

### What LLM providers are supported?

| Provider | Status |
|----------|--------|
| **Google Gemini** | ✅ Optimized, primary |
| **OpenAI** | ✅ Supported |
| **Anthropic** | ✅ Supported |
| **Other** | ✅ Model-agnostic architecture |

### What are the ADK language variants?

| Language | Repository |
|----------|------------|
| **Python** | [google/adk-python](https://github.com/google/adk-python) |
| **Java** | [google/adk-java](https://github.com/google/adk-java) |
| **Go** | [google/adk-go](https://github.com/google/adk-go) |
| **Web** | [google/adk-web](https://github.com/google/adk-web) |

### How do I build a multi-agent system?

ADK supports modular multi-agent systems where you compose specialized agents:

```go
// Define specialized agents
agent1 := adk.NewAgent("researcher", tools...)
agent2 := adk.NewAgent("writer", tools...)

// Compose multi-agent workflow
workflow := adk.NewWorkflow(agent1, agent2)
```

See [examples/](https://github.com/google/adk-go/tree/main/examples) for full samples.

### How do I deploy to Google Cloud Run?

ADK is designed for cloud-native deployment:

1. Build your agent as a Go service
2. Containerize with Docker
3. Deploy to Cloud Run

See deployment guides in the documentation.

### What tools are available?

- Pre-built tools for common tasks
- Custom function tools (define in Go)
- Integration with existing tool libraries

### Where can I find examples?

All example code is in [examples/](https://github.com/google/adk-go/tree/main/examples):
- Basic agent setup
- Multi-agent workflows
- Tool integration
- Deployment examples

### What license does ADK use?

Apache 2.0 License (see [LICENSE](LICENSE)).
Note: `internal/httprr` has a separate license.

### Where can I get help?

| Resource | Link |
|----------|------|
| Documentation | [google.github.io/adk-docs](https://google.github.io/adk-docs/) |
| Samples | [examples/](https://github.com/google/adk-go/tree/main/examples) |
| Reddit | [r/agentdevelopmentkit](https://www.reddit.com/r/agentdevelopmentkit/) |
| Go Doc | [pkg.go.dev](https://pkg.go.dev/google.golang.org/adk) |
| Python ADK | [google/adk-python](https://github.com/google/adk-python) |
| Java ADK | [google/adk-java](https://github.com/google/adk-java) |

## 📄 License

This project is licensed under the Apache 2.0 License - see the
[LICENSE](LICENSE) file for details.

The exception is internal/httprr - see its [LICENSE file](internal/httprr/LICENSE).
