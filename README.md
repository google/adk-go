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

## 🆕 OpenAI Adapter for Local LLMs

**This fork adds OpenAI-compatible adapter support**, enabling you to run ADK agents on:
- 🖥️ **Local LLMs** (LM Studio, Ollama)
- ☁️ **OpenAI API** (GPT-4, GPT-3.5-turbo)
- 🔧 **Any OpenAI-compatible endpoint**

### ✨ Features
- ✅ **Multi-turn tool calling** - Full conversation flow with tool execution
- ✅ **Streaming responses** - Server-Sent Events (SSE) for real-time output
- ✅ **Session management** - Automatic conversation history with TTL
- ✅ **Error handling** - Exponential backoff, rate limiting, retry logic
- ✅ **Comprehensive testing** - 146 tests, 74.8% coverage

### 🚀 Quick Start

**1. Start vLLM with Docker** (recommended — supports MTP speculative decoding)
```bash
docker compose up -d
# First run downloads ~3GB model; subsequent starts use cached weights.
# Wait for health check: curl http://localhost:8000/health
```

**2. Run Example**
```bash
cd examples/openai
go build -o weather_agent main.go
./weather_agent console
```

**3. Try it**
```
> What's the weather in London?
Agent: The weather in London is sunny with a temperature of 22°C...
```

<details>
<summary>Alternative: LM Studio / Ollama</summary>

```bash
# LM Studio (port 1234):
export OPENAI_BASE_URL=http://localhost:1234/v1

# Ollama (port 11434):
export OPENAI_BASE_URL=http://localhost:11434/v1
```
</details>

### 📦 Usage

```go
import "google.golang.org/adk/model/openai"

// Create OpenAI model adapter
model, err := openai.NewModel("Qwen/Qwen3.5-4B", &openai.Config{
    BaseURL: "http://localhost:8000/v1",
})
if err != nil {
    // Handle error
}

// Create agent with tools
agent, err := llmagent.New(llmagent.Config{
    Name:  "my_assistant",
    Model: model,
    Tools: []tool.Tool{/* your tools */},
})
if err != nil {
    // Handle error
}
```

### 🏗️ Architecture

```
model/openai/
├── openai.go          # Main adapter implementation
├── streaming.go       # SSE streaming support
├── converters.go      # ADK ↔ OpenAI format conversion
├── tool_executor.go   # Tool execution engine
├── session.go         # Session management
└── error_handling.go  # Retry & error logic
```

### 🤖 Supported Models

| Model | Provider | Tool Calling | Status | Notes |
|-------|----------|--------------|--------|-------|
| Qwen3.5 (4B, 8B) | Alibaba | ✅ Full | ✅ Recommended | MTP speculative decoding via vLLM |
| Gemma 3 (12B, 4B) | Google | ✅ Full | ✅ Works | |
| GPT-4 | OpenAI | ✅ Full | ✅ Works | |
| Mistral 7B | Mistral | ⚠️ Limited | ✅ Works | |

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

## 📄 License

This project is licensed under the Apache 2.0 License - see the
[LICENSE](LICENSE) file for details.

The exception is internal/httprr - see its [LICENSE file](internal/httprr/LICENSE).
