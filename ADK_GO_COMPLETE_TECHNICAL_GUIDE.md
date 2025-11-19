# Agent Development Kit (ADK) for Go: Complete Technical Guide

**Version**: Based on google.golang.org/adk
**Target Audience**: Advanced Go developers building production AI agent systems
**Prerequisites**: Go 1.24.4+, Understanding of LLMs, REST APIs, and concurrent programming

---

## Table of Contents

1. [Introduction and Core Concepts](#1-introduction-and-core-concepts)
2. [Architecture Overview](#2-architecture-overview)
3. [Installation and Environment Setup](#3-installation-and-environment-setup)
4. [Agent System Deep Dive](#4-agent-system-deep-dive)
5. [Tool Development and Integration](#5-tool-development-and-integration)
6. [Session, State, and Memory Management](#6-session-state-and-memory-management)
7. [Runner and Execution Model](#7-runner-and-execution-model)
8. [Deployment Patterns](#8-deployment-patterns)
9. [Advanced Features](#9-advanced-features)
10. [Testing and Development Workflow](#10-testing-and-development-workflow)
11. [Troubleshooting and Best Practices](#11-troubleshooting-and-best-practices)
12. [Extension and Customization](#12-extension-and-customization)

---

## 1. Introduction and Core Concepts

### 1.1 What is ADK Go?

Agent Development Kit (ADK) for Go is a **code-first, modular framework** for building AI agents with:
- **Software engineering principles**: Testability, versioning, composability
- **Model agnostic**: While optimized for Gemini, supports any LLM via the `model.LLM` interface
- **Deployment agnostic**: Run as CLI, REST API, A2A server, or WebUI
- **Production ready**: Built-in session management, artifact storage, memory services

### 1.2 Core Terminology

| Term | Definition | Code Reference |
|------|------------|----------------|
| **Agent** | Entity that processes user input and generates events | `agent.Agent` interface in `agent/agent.go:39` |
| **LLMAgent** | Agent backed by an LLM model | `llmagent.New()` in `agent/llmagent/llmagent.go:33` |
| **Tool** | Callable functionality exposed to agents | `tool.Tool` interface in `tool/tool.go:28` |
| **Session** | Series of interactions between user and agents | `session.Session` in `session/session.go:30` |
| **Event** | Single interaction record in a conversation | `session.Event` in `session/session.go:90` |
| **Runner** | Execution engine for agent invocations | `runner.Runner` in `runner/runner.go:80` |
| **InvocationContext** | Context passed to agents during execution | `agent.InvocationContext` in `agent/context.go` |
| **Artifact** | Versioned file stored per session | `artifact.Service` in `artifact/service.go:31` |
| **Memory** | Cross-session long-term knowledge store | `memory.Service` in `memory/service.go:30` |

### 1.3 Architectural Principles

**Code-First Development**: Define agent logic, tools, and orchestration directly in Go code, not YAML or JSON configurations.

**Hierarchical Agent Trees**: Agents can have sub-agents, forming execution trees with branch-aware context isolation.

**Event-Driven Streaming**: Agents yield events via Go 1.23+ iterators (`iter.Seq2[*session.Event, error]`).

**Callback Lifecycle**: Before/After callbacks at agent and tool levels for cross-cutting concerns (logging, caching, metrics).

---

## 2. Architecture Overview

### 2.1 System Components

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                          │
│  (Console, REST API, A2A Protocol, WebUI)                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      LAUNCHER LAYER                          │
│  cmd/launcher/{full, prod, console, web, universal}         │
│  • ParseAndRun / Execute                                     │
│  • Unified CLI parsing                                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                       RUNNER LAYER                           │
│  runner/runner.go                                            │
│  • Session retrieval/management                              │
│  • Agent tree traversal                                      │
│  • Event streaming                                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                       AGENT LAYER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  LLMAgent    │  │ CustomAgent  │  │WorkflowAgent │      │
│  │ (llmagent)   │  │ (agent.New)  │  │(seq/par/loop)│      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                        TOOL LAYER                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ FunctionTool│ │ GeminiTool  │ │  MCPToolset │           │
│  │(custom funcs│ │(GoogleSearch│ │(MCP servers)│           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      SERVICE LAYER                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │   Session   │ │  Artifact   │ │   Memory    │           │
│  │   Service   │ │  Service    │ │  Service    │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                       MODEL LAYER                            │
│  model/llm.go (interface)                                    │
│  model/gemini/gemini.go (Gemini implementation)              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

**Single Invocation Flow**:

1. **User Input** → Launcher receives user message (`*genai.Content`)
2. **Session Retrieval** → Runner fetches/creates session from SessionService
3. **Agent Selection** → Runner determines which agent to run (based on last transfer or root)
4. **Context Creation** → InvocationContext constructed with artifacts, memory, session
5. **Event Generation** → Agent.Run() yields events via iterator
6. **Event Persistence** → Events saved to session
7. **Response Streaming** → Events streamed back to client

### 2.3 Agent Execution Tree

```go
// Example agent tree structure
RootAgent (LLMAgent)
├── SearchAgent (LLMAgent with GoogleSearch)
├── PoemAgent (LLMAgent with custom function tool)
└── WorkflowAgent (SequentialAgent)
    ├── SubAgent1 (Custom Agent)
    └── SubAgent2 (LLMAgent)
```

**Branch Isolation**: When `ParallelAgent` runs sub-agents, each gets a unique branch (e.g., `root.parallel.subagent1`). Events with different branches don't pollute each other's conversation history.

---

## 3. Installation and Environment Setup

### 3.1 Installation

```bash
# Add ADK Go to your project
go get google.golang.org/adk

# Verify installation
go list -m google.golang.org/adk
# Output: google.golang.org/adk v0.x.x
```

### 3.2 Required Dependencies

From `go.mod`:

```go
require (
    google.golang.org/genai v1.20.0           // Gemini API client
    google.golang.org/api v0.252.0            // Google APIs
    github.com/gorilla/mux v1.8.1              // HTTP routing (optional)
    github.com/modelcontextprotocol/go-sdk     // MCP support
    github.com/a2aproject/a2a-go               // A2A protocol
)
```

### 3.3 Environment Variables

```bash
# Required for Gemini models
export GOOGLE_API_KEY="your-api-key"

# Optional: For Google Cloud services
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"

# Optional: For GitHub MCP server example
export GITHUB_PAT="your-github-token"
```

### 3.4 Project Structure

Recommended structure for an ADK Go application:

```
myagent/
├── go.mod
├── go.sum
├── main.go                    # Entry point
├── agents/                    # Agent definitions
│   ├── root_agent.go
│   ├── search_agent.go
│   └── custom_agent.go
├── tools/                     # Custom tools
│   ├── calculator.go
│   └── database_query.go
├── services/                  # Service implementations
│   ├── session.go
│   └── artifact.go
└── config/                    # Configuration
    └── config.go
```

---

## 4. Agent System Deep Dive

### 4.1 Agent Interface

**Core Interface** (`agent/agent.go:39`):

```go
type Agent interface {
    Name() string
    Description() string
    Run(InvocationContext) iter.Seq2[*session.Event, error]
    SubAgents() []Agent
    internal() *agent  // Internal method
}
```

**Invocation Context** provides access to:
- `Agent()`: Current agent
- `Session()`: Mutable session (state, events)
- `Artifacts()`: Artifact storage
- `Memory()`: Cross-session memory
- `UserContent()`: Original user message
- `InvocationID()`: Unique ID for this run
- `Branch()`: Current branch in agent tree
- `EndInvocation()`: Signal early termination

### 4.2 LLMAgent Configuration

**Complete Configuration** (`agent/llmagent/llmagent.go:102`):

```go
type Config struct {
    // === Basic Configuration ===
    Name        string   // Unique name in agent tree (cannot be "user")
    Description string   // One-line capability description for LLM
    Model       model.LLM

    // === Instructions ===
    Instruction         string  // Template string with {var} placeholders
    InstructionProvider InstructionProvider  // Dynamic instruction generation
    GlobalInstruction   string  // Applies to entire agent tree (root only)

    // === Tools ===
    Tools    []tool.Tool      // Individual tools
    Toolsets []tool.Toolset   // Tool collections

    // === Sub-Agents ===
    SubAgents []agent.Agent

    // === Agent Transfer Control ===
    DisallowTransferToParent bool
    DisallowTransferToPeers  bool

    // === Schema Validation ===
    InputSchema  *genai.Schema  // Validate user input
    OutputSchema *genai.Schema  // Structured LLM output
    OutputKey    string         // Key to extract from structured output

    // === Content Inclusion ===
    IncludeContents IncludeContents  // Control conversation history

    // === Callbacks ===
    BeforeAgentCallbacks []agent.BeforeAgentCallback
    AfterAgentCallbacks  []agent.AfterAgentCallback
    BeforeModelCallbacks []BeforeModelCallback
    AfterModelCallbacks  []AfterModelCallback
    BeforeToolCallbacks  []BeforeToolCallback
    AfterToolCallbacks   []AfterToolCallback

    // === Model Configuration ===
    GenerateContentConfig *genai.GenerateContentConfig  // Temperature, safety, etc.
}
```

**Instruction Templates**:

```go
// Variables from session state
Instruction: "You are a {role} assistant. User's name is {user_name}."
// At runtime: session.State().Get("role") and Get("user_name")

// Artifact content insertion
Instruction: "Use the guidelines in {artifact.style_guide}."
// Inserts text content of artifact named "style_guide"

// Optional variables (no error if missing)
Instruction: "User preference: {preference?}"
// Returns empty string if "preference" not in state
```

### 4.3 Creating Custom Agents

**Example 1: Simple Custom Agent**:

```go
package main

import (
    "iter"
    "google.golang.org/adk/agent"
    "google.golang.org/adk/model"
    "google.golang.org/adk/session"
    "google.golang.org/genai"
)

func weatherAgentRun(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
    return func(yield func(*session.Event, error) bool) {
        // Access session state
        city, err := ctx.Session().State().Get("city")
        if err != nil {
            yield(nil, err)
            return
        }

        // Create response event
        event := session.NewEvent(ctx.InvocationID())
        event.LLMResponse = model.LLMResponse{
            Content: &genai.Content{
                Parts: []*genai.Part{
                    {Text: "Weather in " + city.(string) + " is sunny!"},
                },
            },
        }

        // Yield event
        yield(event, nil)
    }
}

func main() {
    customAgent, err := agent.New(agent.Config{
        Name:        "weather_agent",
        Description: "Provides weather information",
        Run:         weatherAgentRun,
    })
    if err != nil {
        panic(err)
    }
}
```

**Example 2: Agent with State Modification**:

```go
func counterAgentRun(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
    return func(yield func(*session.Event, error) bool) {
        // Read current count
        count, err := ctx.Session().State().Get("count")
        if err != nil {
            count = 0
        }

        // Increment
        newCount := count.(int) + 1
        ctx.Session().State().Set("count", newCount)

        // Create event with state delta
        event := session.NewEvent(ctx.InvocationID())
        event.Actions.StateDelta["count"] = newCount
        event.LLMResponse = model.LLMResponse{
            Content: &genai.Content{
                Parts: []*genai.Part{
                    {Text: fmt.Sprintf("Count is now %d", newCount)},
                },
            },
        }

        yield(event, nil)
    }
}
```

### 4.4 LLMAgent Example

**Quickstart Agent** (`examples/quickstart/main.go`):

```go
func main() {
    ctx := context.Background()

    // 1. Create Gemini model
    model, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{
        APIKey: os.Getenv("GOOGLE_API_KEY"),
    })
    if err != nil {
        log.Fatalf("Failed to create model: %v", err)
    }

    // 2. Create LLM agent with Google Search tool
    a, err := llmagent.New(llmagent.Config{
        Name:        "weather_time_agent",
        Model:       model,
        Description: "Agent to answer questions about the time and weather in a city.",
        Instruction: "Your SOLE purpose is to answer questions about the current time and weather in a specific city. You MUST refuse to answer any questions unrelated to time or weather.",
        Tools: []tool.Tool{
            geminitool.GoogleSearch{},
        },
    })
    if err != nil {
        log.Fatalf("Failed to create agent: %v", err)
    }

    // 3. Configure launcher
    config := &launcher.Config{
        AgentLoader: agent.NewSingleLoader(a),
    }

    // 4. Run with full launcher (console, REST, A2A, WebUI)
    l := full.NewLauncher()
    if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
        log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
    }
}
```

### 4.5 Workflow Agents

#### 4.5.1 SequentialAgent

**Runs sub-agents in order, each seeing previous outputs**.

```go
import "google.golang.org/adk/agent/workflowagents/sequentialagent"

sequentialAgent, err := sequentialagent.New(sequentialagent.Config{
    AgentConfig: agent.Config{
        Name:        "sequential_workflow",
        Description: "Runs research then summarization",
        SubAgents:   []agent.Agent{researchAgent, summarizerAgent},
    },
})
```

**Use Cases**:
- Multi-step pipelines (research → analysis → summary)
- Progressive refinement
- Chain-of-thought workflows

#### 4.5.2 ParallelAgent

**Runs sub-agents concurrently in isolated branches**.

```go
import "google.golang.org/adk/agent/workflowagents/parallelagent"

parallelAgent, err := parallelagent.New(parallelagent.Config{
    AgentConfig: agent.Config{
        Name:        "parallel_workflow",
        Description: "Generates multiple perspectives",
        SubAgents:   []agent.Agent{agent1, agent2, agent3},
    },
})
```

**Key Points** (`agent/workflowagents/parallelagent/agent.go:66`):
- Uses `errgroup` for concurrency
- Each sub-agent gets unique branch: `parent.ParallelAgent.SubAgentName`
- All events collected and yielded
- If any sub-agent errors, all are cancelled

**Use Cases**:
- Multiple algorithm attempts
- A/B response generation
- Parallel data processing

#### 4.5.3 LoopAgent

**Runs sub-agent repeatedly up to max iterations**.

```go
import "google.golang.org/adk/agent/workflowagents/loopagent"

loopAgent, err := loopagent.New(loopagent.Config{
    MaxIterations: 5,
    AgentConfig: agent.Config{
        Name:        "retry_workflow",
        Description: "Retries sub-agent up to 5 times",
        SubAgents:   []agent.Agent{retriableAgent},
    },
})
```

**Exit Control**:
- Tool `exitlooptool.ExitLoopTool` allows early termination
- Sub-agent calls this tool to break the loop

**Use Cases**:
- Retry logic with backoff
- Iterative refinement
- Monte Carlo simulations

### 4.6 Agent Callbacks

**Before Agent Callbacks** (`agent/agent.go:119`):

```go
type BeforeAgentCallback func(CallbackContext) (*genai.Content, error)

beforeCallback := func(ctx agent.CallbackContext) (*genai.Content, error) {
    // Logging example
    log.Printf("Agent %s starting for user %s", ctx.AgentName(), ctx.UserID())

    // Gating example: block execution based on state
    state := ctx.ReadonlyState()
    blocked, _ := state.Get("blocked")
    if blocked.(bool) {
        return &genai.Content{
            Parts: []*genai.Part{{Text: "Agent is currently blocked"}},
        }, nil
    }

    // Return nil to continue normal execution
    return nil, nil
}

llmagent.New(llmagent.Config{
    // ...
    BeforeAgentCallbacks: []agent.BeforeAgentCallback{beforeCallback},
})
```

**After Agent Callbacks** (`agent/agent.go:125`):

```go
afterCallback := func(ctx agent.CallbackContext) (*genai.Content, error) {
    // Analytics example
    recordAgentCompletion(ctx.AgentName(), ctx.SessionID())

    // State cleanup
    ctx.State().Set("agent_completed", time.Now())

    return nil, nil
}
```

**Model-Level Callbacks** (LLMAgent only):

```go
import "google.golang.org/adk/agent/llmagent"

beforeModel := func(ctx llmagent.BeforeModelContext, req *model.LLMRequest) (*model.LLMResponse, error) {
    // Caching example
    cacheKey := generateCacheKey(req)
    if cached, found := cache.Get(cacheKey); found {
        return cached.(*model.LLMResponse), nil
    }

    // Modify request (e.g., adjust temperature)
    if req.Config == nil {
        req.Config = &genai.GenerateContentConfig{}
    }
    req.Config.Temperature = 0.7

    return nil, nil  // Continue to actual LLM call
}

afterModel := func(ctx llmagent.AfterModelContext, req *model.LLMRequest, resp *model.LLMResponse) (*model.LLMResponse, error) {
    // Token usage tracking
    if resp.UsageMetadata != nil {
        logTokenUsage(resp.UsageMetadata)
    }

    return nil, nil  // Use original response
}

llmagent.New(llmagent.Config{
    // ...
    BeforeModelCallbacks: []llmagent.BeforeModelCallback{beforeModel},
    AfterModelCallbacks:  []llmagent.AfterModelCallback{afterModel},
})
```

### 4.7 Agent Transfer

**Implicit Transfer**: LLM chooses to call sub-agent as a tool (when using `agenttool.New`).

**Explicit Transfer** via `Actions.TransferTo`:

```go
func agentRun(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
    return func(yield func(*session.Event, error) bool) {
        // Decide to transfer to sub-agent
        event := session.NewEvent(ctx.InvocationID())
        event.Actions.TransferTo = "sub_agent_name"
        event.LLMResponse = model.LLMResponse{
            Content: &genai.Content{
                Parts: []*genai.Part{{Text: "Transferring to specialist..."}},
            },
        }
        yield(event, nil)
    }
}
```

**Transfer Control**:
- `DisallowTransferToParent`: Sub-agent cannot transfer back to parent
- `DisallowTransferToPeers`: Sub-agent cannot transfer to siblings

---

## 5. Tool Development and Integration

### 5.1 Tool Interface

**Base Interface** (`tool/tool.go:28`):

```go
type Tool interface {
    Name() string
    Description() string
    IsLongRunning() bool
}
```

**Tool Types**:

1. **FunctionTool**: Custom Go functions
2. **GeminiTool**: Gemini API native tools (GoogleSearch, CodeExecution, etc.)
3. **MCPToolset**: Model Context Protocol servers
4. **AgentTool**: Wrap agent as a tool

### 5.2 Creating Function Tools

**Basic Function Tool**:

```go
import "google.golang.org/adk/tool/functiontool"

type CalculatorInput struct {
    Operation string  `json:"operation" jsonschema:"description=Operation: add, subtract, multiply, divide"`
    A         float64 `json:"a" jsonschema:"description=First number"`
    B         float64 `json:"b" jsonschema:"description=Second number"`
}

type CalculatorOutput struct {
    Result float64 `json:"result"`
    Error  string  `json:"error,omitempty"`
}

func calculatorHandler(ctx tool.Context, input CalculatorInput) (CalculatorOutput, error) {
    var result float64
    switch input.Operation {
    case "add":
        result = input.A + input.B
    case "subtract":
        result = input.A - input.B
    case "multiply":
        result = input.A * input.B
    case "divide":
        if input.B == 0 {
            return CalculatorOutput{Error: "division by zero"}, nil
        }
        result = input.A / input.B
    default:
        return CalculatorOutput{}, fmt.Errorf("unknown operation: %s", input.Operation)
    }

    return CalculatorOutput{Result: result}, nil
}

func main() {
    calcTool, err := functiontool.New(functiontool.Config{
        Name:        "calculator",
        Description: "Performs basic arithmetic operations",
    }, calculatorHandler)
    if err != nil {
        log.Fatal(err)
    }

    // Use in agent
    agent, _ := llmagent.New(llmagent.Config{
        Name:  "math_agent",
        Model: model,
        Tools: []tool.Tool{calcTool},
    })
}
```

**JSON Schema Annotations**:

```go
type WeatherInput struct {
    City    string `json:"city" jsonschema:"description=City name,required"`
    Country string `json:"country" jsonschema:"description=Country code (ISO 3166-1 alpha-2)"`
    Units   string `json:"units" jsonschema:"description=Temperature units,enum=celsius,enum=fahrenheit,default=celsius"`
}
```

### 5.3 Tool Context Access

**Tool Context** (`tool/tool.go:42`):

```go
func advancedToolHandler(ctx tool.Context, input MyInput) (MyOutput, error) {
    // Session state access
    apiKey, err := ctx.ReadonlyState().Get("api_key")
    if err != nil {
        return MyOutput{}, fmt.Errorf("API key not configured")
    }

    // Modify state via Actions
    ctx.Actions().StateDelta["last_tool_call"] = time.Now()

    // Transfer to another agent
    ctx.Actions().TransferTo = "specialist_agent"

    // Memory search
    memories, err := ctx.SearchMemory(ctx, "previous weather queries")

    // Function call ID (for long-running ops)
    callID := ctx.FunctionCallID()

    // User/Session identification
    userID := ctx.UserID()
    sessionID := ctx.SessionID()

    // Execute logic...
    return MyOutput{...}, nil
}
```

### 5.4 Long-Running Tools

**Pattern for Async Operations**:

```go
type AsyncJobInput struct {
    JobType string `json:"job_type"`
}

type AsyncJobOutput struct {
    JobID  string `json:"job_id"`
    Status string `json:"status"`
}

func asyncJobHandler(ctx tool.Context, input AsyncJobInput) (AsyncJobOutput, error) {
    // Start async job
    jobID := startBackgroundJob(input.JobType)

    // Mark this tool call as long-running
    ctx.Actions().LongRunningToolIDs = append(
        ctx.Actions().LongRunningToolIDs,
        ctx.FunctionCallID(),
    )

    return AsyncJobOutput{
        JobID:  jobID,
        Status: "started",
    }, nil
}

// IsLongRunning returns true
func (t *AsyncJobTool) IsLongRunning() bool {
    return true
}
```

### 5.5 Gemini Native Tools

**Google Search** (`tool/geminitool/tool.go`):

```go
import "google.golang.org/adk/tool/geminitool"

agent, _ := llmagent.New(llmagent.Config{
    Name:  "search_agent",
    Model: model,
    Tools: []tool.Tool{
        geminitool.GoogleSearch{},  // Pre-built tool
    },
})
```

**Custom Gemini Tool**:

```go
import (
    "google.golang.org/adk/tool/geminitool"
    "google.golang.org/genai"
)

codeExecutionTool := geminitool.New("code_execution", &genai.Tool{
    CodeExecution: &genai.CodeExecution{},
})

// Or retrieval tool
retrievalTool := geminitool.New("data_retrieval", &genai.Tool{
    Retrieval: &genai.Retrieval{
        ExternalAPI: &genai.ExternalAPI{
            Endpoint: "https://api.example.com/data",
            AuthConfig: &genai.AuthConfig{
                APIKey: os.Getenv("DATA_API_KEY"),
            },
        },
    },
})
```

### 5.6 MCP Toolset Integration

**Model Context Protocol** enables tool sharing across agent systems.

**In-Memory MCP Server** (`examples/mcp/main.go:62`):

```go
import (
    "github.com/modelcontextprotocol/go-sdk/mcp"
    "google.golang.org/adk/tool/mcptoolset"
)

type WeatherInput struct {
    City string `json:"city" jsonschema:"city name"`
}

type WeatherOutput struct {
    Summary string `json:"weather_summary"`
}

func getWeatherHandler(ctx context.Context, req *mcp.CallToolRequest, input WeatherInput) (*mcp.CallToolResult, WeatherOutput, error) {
    return nil, WeatherOutput{
        Summary: fmt.Sprintf("Today in %q is sunny", input.City),
    }, nil
}

func main() {
    // Create in-memory MCP server
    clientTransport, serverTransport := mcp.NewInMemoryTransports()

    server := mcp.NewServer(&mcp.Implementation{
        Name:    "weather_server",
        Version: "v1.0.0",
    }, nil)

    mcp.AddTool(server, &mcp.Tool{
        Name:        "get_weather",
        Description: "returns weather in the given city",
    }, getWeatherHandler)

    server.Connect(ctx, serverTransport, nil)

    // Create MCP toolset for agent
    mcpToolSet, _ := mcptoolset.New(mcptoolset.Config{
        Transport: clientTransport,
    })

    agent, _ := llmagent.New(llmagent.Config{
        Name:     "helper_agent",
        Model:    model,
        Toolsets: []tool.Toolset{mcpToolSet},
    })
}
```

**Remote MCP Server** (GitHub example):

```go
import (
    "golang.org/x/oauth2"
    "github.com/modelcontextprotocol/go-sdk/mcp"
)

func githubMCPTransport(ctx context.Context) mcp.Transport {
    ts := oauth2.StaticTokenSource(
        &oauth2.Token{AccessToken: os.Getenv("GITHUB_PAT")},
    )
    return &mcp.StreamableClientTransport{
        Endpoint:   "https://api.githubcopilot.com/mcp/",
        HTTPClient: oauth2.NewClient(ctx, ts),
    }
}

mcpToolSet, _ := mcptoolset.New(mcptoolset.Config{
    Transport: githubMCPTransport(ctx),
})
```

### 5.7 Agent-as-Tool Pattern

**Use Case**: Combine tools of different types in one agent (workaround for genai API limitations).

```go
import "google.golang.org/adk/tool/agenttool"

// Create specialized agents
searchAgent, _ := llmagent.New(llmagent.Config{
    Name:  "search_agent",
    Model: model,
    Tools: []tool.Tool{geminitool.GoogleSearch{}},
})

poemAgent, _ := llmagent.New(llmagent.Config{
    Name:  "poem_agent",
    Model: model,
    Tools: []tool.Tool{poemTool},  // Custom function tool
})

// Wrap agents as tools
searchTool := agenttool.New(searchAgent, nil)
poemToolWrapped := agenttool.New(poemAgent, nil)

// Root agent uses both
rootAgent, _ := llmagent.New(llmagent.Config{
    Name:  "root_agent",
    Model: model,
    Tools: []tool.Tool{searchTool, poemToolWrapped},
})
```

### 5.8 Toolsets (Dynamic Tool Collections)

```go
type Toolset interface {
    Name() string
    Tools(ctx agent.ReadonlyContext) ([]Tool, error)
}

// Example: Toolset that returns different tools based on user role
type RoleBasedToolset struct{}

func (t *RoleBasedToolset) Name() string {
    return "role_based_tools"
}

func (t *RoleBasedToolset) Tools(ctx agent.ReadonlyContext) ([]tool.Tool, error) {
    role, err := ctx.State().Get("user_role")
    if err != nil {
        return []tool.Tool{}, nil
    }

    switch role.(string) {
    case "admin":
        return []tool.Tool{adminTool1, adminTool2}, nil
    case "user":
        return []tool.Tool{userTool1}, nil
    default:
        return []tool.Tool{}, nil
    }
}
```

### 5.9 Tool Predicate Filtering

```go
import "google.golang.org/adk/tool"

// Only expose specific tools to an agent
allowedTools := []string{"calculator", "weather_lookup"}
predicate := tool.StringPredicate(allowedTools)

// Use in agent config (via toolset wrapper or custom logic)
```

---

## 6. Session, State, and Memory Management

### 6.1 Session Lifecycle

**Session Interface** (`session/session.go:30`):

```go
type Session interface {
    ID() string                 // Unique session identifier
    AppName() string            // Application name
    UserID() string             // User identifier
    State() State               // Key-value state
    Events() Events             // Conversation events
    LastUpdateTime() time.Time
}
```

**Session Service**:

```go
type Service interface {
    Get(ctx context.Context, req *GetRequest) (*GetResponse, error)
    Save(ctx context.Context, req *SaveRequest) error
    List(ctx context.Context, req *ListRequest) (*ListResponse, error)
    Delete(ctx context.Context, req *DeleteRequest) error
}
```

**In-Memory Session Service**:

```go
import "google.golang.org/adk/session"

sessionService := session.InMemoryService()

// Used in runner config
runner.New(runner.Config{
    Agent:          myAgent,
    SessionService: sessionService,
    AppName:        "my_app",
})
```

### 6.2 State Management

**State Interface** (`session/session.go:49`):

```go
type State interface {
    Get(string) (any, error)
    Set(string, any) error
    All() iter.Seq2[string, any]
}
```

**Usage in Agents**:

```go
func agentRun(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
    return func(yield func(*session.Event, error) bool) {
        // Read state
        userName, err := ctx.Session().State().Get("user_name")
        if err != nil {
            userName = "Guest"
        }

        // Modify state
        ctx.Session().State().Set("last_access", time.Now())
        ctx.Session().State().Set("interaction_count", 42)

        // Iterate all state
        for key, value := range ctx.Session().State().All() {
            log.Printf("State: %s = %v", key, value)
        }

        // Create event with state delta
        event := session.NewEvent(ctx.InvocationID())
        event.Actions.StateDelta["user_name"] = userName
        event.Actions.StateDelta["last_access"] = time.Now()

        yield(event, nil)
    }
}
```

**State Delta in Events**:

```go
type EventActions struct {
    StateDelta    map[string]any  // State changes in this event
    ArtifactDelta map[string]int64 // Artifact version updates
    TransferTo    string          // Agent transfer
    // ...
}
```

### 6.3 Events

**Event Structure** (`session/session.go:90`):

```go
type Event struct {
    // Metadata
    ID           string
    Timestamp    time.Time
    InvocationID string
    Branch       string
    Author       string

    // Content
    LLMResponse model.LLMResponse  // Contains Content, UsageMetadata, etc.

    // Actions
    Actions            EventActions
    LongRunningToolIDs []string
}
```

**Creating Events**:

```go
event := session.NewEvent(ctx.InvocationID())
event.Author = ctx.Agent().Name()
event.Branch = ctx.Branch()
event.LLMResponse = model.LLMResponse{
    Content: &genai.Content{
        Role: genai.RoleModel,
        Parts: []*genai.Part{
            {Text: "Response text"},
            {FunctionCall: &genai.FunctionCall{...}},
        },
    },
    UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
        PromptTokenCount:     100,
        CandidatesTokenCount: 50,
    },
}
event.Actions.StateDelta["key"] = "value"

yield(event, nil)
```

**Event Types**:

1. **User Input Event**: `Role: genai.RoleUser`
2. **Model Response**: `Role: genai.RoleModel` with text parts
3. **Function Call**: Parts contain `FunctionCall`
4. **Function Response**: Parts contain `FunctionResponse`
5. **Streaming Partial**: `Partial: true` for incremental text

### 6.4 Artifacts

**Artifact Service** (`artifact/service.go:31`):

```go
type Service interface {
    Save(ctx context.Context, req *SaveRequest) (*SaveResponse, error)
    Load(ctx context.Context, req *LoadRequest) (*LoadResponse, error)
    Delete(ctx context.Context, req *DeleteRequest) error
    List(ctx context.Context, req *ListRequest) (*ListResponse, error)
    Versions(ctx context.Context, req *VersionsRequest) (*VersionsResponse, error)
}
```

**Using Artifacts in Agents**:

```go
func agentRun(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
    return func(yield func(*session.Event, error) bool) {
        // Save artifact
        saveResp, err := ctx.Artifacts().Save(ctx, "report.txt", &genai.Part{
            Text: "Generated report content...",
        })
        if err != nil {
            yield(nil, err)
            return
        }

        // Load artifact
        loadResp, err := ctx.Artifacts().Load(ctx, "report.txt")
        if err != nil {
            yield(nil, err)
            return
        }

        content := loadResp.Part.Text

        // Load specific version
        oldVersion, err := ctx.Artifacts().LoadVersion(ctx, "report.txt", 1)

        // List all artifacts
        listResp, err := ctx.Artifacts().List(ctx)
        for _, name := range listResp.FileNames {
            log.Printf("Artifact: %s", name)
        }

        // Event with artifact delta
        event := session.NewEvent(ctx.InvocationID())
        event.Actions.ArtifactDelta["report.txt"] = saveResp.Version

        yield(event, nil)
    }
}
```

**Artifact in Instructions**:

```go
llmagent.New(llmagent.Config{
    Instruction: "Follow the guidelines in {artifact.style_guide}",
    // At runtime, ADK loads artifact "style_guide" and inserts text content
})
```

**In-Memory Artifact Service**:

```go
import "google.golang.org/adk/artifact"

artifactService := artifact.InMemoryService()

runner.New(runner.Config{
    Agent:           myAgent,
    SessionService:  sessionService,
    ArtifactService: artifactService,  // Optional
})
```

**GCS Artifact Service**:

```go
import "google.golang.org/adk/artifact/gcsartifact"

gcsService, err := gcsartifact.NewService(ctx, gcsartifact.Config{
    BucketName: "my-artifacts-bucket",
})

runner.New(runner.Config{
    Agent:           myAgent,
    ArtifactService: gcsService,
})
```

### 6.5 Memory (Cross-Session)

**Memory Service** (`memory/service.go:30`):

```go
type Service interface {
    AddSession(ctx context.Context, s session.Session) error
    Search(ctx context.Context, req *SearchRequest) (*SearchResponse, error)
}

type SearchRequest struct {
    Query   string
    UserID  string
    AppName string
}

type SearchResponse struct {
    Memories []Entry
}

type Entry struct {
    Content   *genai.Content
    Author    string
    Timestamp time.Time
}
```

**Using Memory in Agents**:

```go
func agentRun(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
    return func(yield func(*session.Event, error) bool) {
        // Search memory for relevant past interactions
        memResp, err := ctx.Memory().Search(ctx, "user preferences")
        if err != nil {
            yield(nil, err)
            return
        }

        // Use memories to enrich context
        var pastPreferences []string
        for _, entry := range memResp.Memories {
            for _, part := range entry.Content.Parts {
                pastPreferences = append(pastPreferences, part.Text)
            }
        }

        // Generate response incorporating memory
        response := fmt.Sprintf("Based on your past preferences: %s",
            strings.Join(pastPreferences, ", "))

        event := session.NewEvent(ctx.InvocationID())
        event.LLMResponse = model.LLMResponse{
            Content: &genai.Content{
                Parts: []*genai.Part{{Text: response}},
            },
        }

        yield(event, nil)
    }
}
```

**In-Memory Memory Service**:

```go
import "google.golang.org/adk/memory"

memoryService := memory.InMemoryService()

runner.New(runner.Config{
    Agent:         myAgent,
    MemoryService: memoryService,  // Optional
})
```

**Adding Sessions to Memory**:

```go
// Typically done after session completion
err := memoryService.AddSession(ctx, session)
```

---

## 7. Runner and Execution Model

### 7.1 Runner Configuration

**Runner Config** (`runner/runner.go:40`):

```go
type Config struct {
    AppName         string
    Agent           agent.Agent         // Root agent
    SessionService  session.Service     // Required
    ArtifactService artifact.Service    // Optional
    MemoryService   memory.Service      // Optional
}

r, err := runner.New(runner.Config{
    AppName:         "my_agent_app",
    Agent:           rootAgent,
    SessionService:  session.InMemoryService(),
    ArtifactService: artifact.InMemoryService(),
    MemoryService:   memory.InMemoryService(),
})
```

### 7.2 Runner Execution

**Run Method** (`runner/runner.go:93`):

```go
func (r *Runner) Run(
    ctx context.Context,
    userID, sessionID string,
    msg *genai.Content,
    cfg agent.RunConfig,
) iter.Seq2[*session.Event, error]
```

**Direct Runner Usage**:

```go
ctx := context.Background()
r, _ := runner.New(runner.Config{...})

userID := "user123"
sessionID := "session456"
userMessage := &genai.Content{
    Role: genai.RoleUser,
    Parts: []*genai.Part{
        {Text: "What's the weather in Tokyo?"},
    },
}

runConfig := agent.RunConfig{
    StreamingMode:            agent.StreamingModeText,
    SaveInputBlobsAsArtifacts: false,
}

// Iterate through events
for event, err := range r.Run(ctx, userID, sessionID, userMessage, runConfig) {
    if err != nil {
        log.Fatalf("Error: %v", err)
    }

    if event.IsFinalResponse() {
        fmt.Printf("Final response: %v\n", event.LLMResponse.Content)
    } else {
        fmt.Printf("Event: %v\n", event)
    }
}
```

### 7.3 Streaming Modes

**StreamingMode** in `agent.RunConfig`:

```go
type StreamingMode int

const (
    StreamingModeNone StreamingMode = iota  // No streaming
    StreamingModeText                        // Stream partial text
    StreamingModeEvents                      // Stream all events
)
```

**Text Streaming Example**:

```go
runConfig := agent.RunConfig{
    StreamingMode: agent.StreamingModeText,
}

for event, err := range r.Run(ctx, userID, sessionID, msg, runConfig) {
    if event.LLMResponse.Partial {
        // Partial text update
        fmt.Print(event.LLMResponse.Content.Parts[0].Text)
    } else if event.IsFinalResponse() {
        fmt.Println("\n[Final]")
    }
}
```

### 7.4 Agent Selection

**How Runner Chooses Agent**:

1. Check last event for `Actions.TransferTo` → use that agent
2. Else use root agent

**Agent Tree Traversal** (`internal/agent/parentmap`):

- Runner builds parent map at initialization
- Enables finding agents by name anywhere in tree
- Validates no duplicate names

---

## 8. Deployment Patterns

### 8.1 Launcher System

**Launcher Variants**:

| Launcher | Import Path | Modes Supported |
|----------|-------------|-----------------|
| **Full** | `cmd/launcher/full` | Console, REST API, A2A, WebUI |
| **Prod** | `cmd/launcher/prod` | REST API, A2A (no console/webui) |
| **Console** | `cmd/launcher/console` | Console only |
| **Universal** | `cmd/launcher/universal` | Base for custom launchers |

### 8.2 Console Mode

**Usage**:

```bash
# Run agent in interactive console
go run ./main.go console

# Or with full launcher
go run ./main.go help  # Show all modes
```

**Example** (`examples/quickstart/main.go:59`):

```go
import "google.golang.org/adk/cmd/launcher/full"

l := full.NewLauncher()
err := l.Execute(ctx, &launcher.Config{
    AgentLoader: agent.NewSingleLoader(myAgent),
}, os.Args[1:])
```

**Console Interaction**:

```
> What's the weather in London?
[Agent: weather_time_agent]
The current weather in London is partly cloudy with a temperature of 15°C.

> What's the time there?
[Agent: weather_time_agent]
The current time in London is 14:30 GMT.
```

### 8.3 REST API Mode

**Start REST Server**:

```bash
go run ./main.go restapi --port 8080
```

**API Endpoints** (`server/adkrest/handler.go`):

- `POST /sessions/{sessionId}/invoke` - Send message to agent
- `POST /sessions/{sessionId}/artifacts` - Upload artifact
- `GET /sessions/{sessionId}/artifacts` - List artifacts
- `GET /sessions/{sessionId}/artifacts/{fileName}` - Download artifact
- `GET /sessions` - List sessions
- `GET /sessions/{sessionId}` - Get session details
- `DELETE /sessions/{sessionId}` - Delete session

**Invoke Request**:

```bash
curl -X POST http://localhost:8080/sessions/session123/invoke \
  -H "Content-Type: application/json" \
  -H "X-User-ID: user456" \
  -d '{
    "message": {
      "role": "user",
      "parts": [{"text": "What'\''s the weather in Tokyo?"}]
    },
    "stream": true
  }'
```

**Streaming Response** (SSE):

```
data: {"event": {...}, "partial": true}

data: {"event": {...}, "partial": true}

data: {"event": {...}, "partial": false}
```

**Standalone REST API** (`examples/rest/main.go`):

```go
import "google.golang.org/adk/server/adkrest"

config := &launcher.Config{
    AgentLoader:    agent.NewSingleLoader(myAgent),
    SessionService: session.InMemoryService(),
}

apiHandler := adkrest.NewHandler(config)

mux := http.NewServeMux()
mux.Handle("/api/", http.StripPrefix("/api", apiHandler))

http.ListenAndServe(":8080", mux)
```

### 8.4 A2A (Agent-to-Agent) Protocol

**Start A2A Server**:

```bash
go run ./main.go a2a --port 8081
```

**A2A Protocol**: Standardized agent communication protocol from the A2A project.

**Server Implementation** (`examples/a2a/main.go:63`):

```go
import (
    "github.com/a2aproject/a2a-go/a2a"
    "github.com/a2aproject/a2a-go/a2asrv"
    "google.golang.org/adk/server/adka2a"
)

// Create agent card
agentCard := &a2a.AgentCard{
    Name:               myAgent.Name(),
    Skills:             adka2a.BuildAgentSkills(myAgent),
    PreferredTransport: a2a.TransportProtocolJSONRPC,
    URL:                baseURL.JoinPath("/invoke").String(),
    Capabilities:       a2a.AgentCapabilities{Streaming: true},
}

// Register handlers
mux := http.NewServeMux()
mux.Handle(a2asrv.WellKnownAgentCardPath, a2asrv.NewStaticAgentCardHandler(agentCard))

executor := adka2a.NewExecutor(adka2a.ExecutorConfig{
    RunnerConfig: runner.Config{
        AppName:        myAgent.Name(),
        Agent:          myAgent,
        SessionService: session.InMemoryService(),
    },
})
requestHandler := a2asrv.NewHandler(executor)
mux.Handle("/invoke", a2asrv.NewJSONRPCHandler(requestHandler))

http.Serve(listener, mux)
```

**Client (Remote Agent)**:

```go
import "google.golang.org/adk/agent/remoteagent"

remoteAgent, err := remoteagent.NewA2A(remoteagent.A2AConfig{
    Name:            "remote_weather_agent",
    AgentCardSource: "http://localhost:8081",  // A2A server URL
})

// Use remote agent as sub-agent
rootAgent, _ := llmagent.New(llmagent.Config{
    Name:      "orchestrator",
    SubAgents: []agent.Agent{remoteAgent},
})
```

### 8.5 WebUI Mode

**Start with WebUI**:

```bash
go run ./main.go webui --port 8080
```

**Features**:
- Interactive chat interface
- Artifact upload/download
- Session management
- Real-time streaming

### 8.6 Cloud Run Deployment

**ADK Go CLI** (`cmd/adkgo`):

```bash
# Build CLI
go install google.golang.org/adk/cmd/adkgo

# Deploy to Cloud Run
adkgo deploy cloudrun \
  --project my-gcp-project \
  --region us-central1 \
  --source ./my-agent
```

**Manual Deployment**:

```dockerfile
# Dockerfile
FROM golang:1.24 AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o /agent ./main.go

FROM alpine:latest
RUN apk --no-cache add ca-certificates
COPY --from=builder /agent /agent
EXPOSE 8080
CMD ["/agent", "restapi", "--port", "8080"]
```

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/my-project/my-agent
gcloud run deploy my-agent \
  --image gcr.io/my-project/my-agent \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_API_KEY=secret
```

---

## 9. Advanced Features

### 9.1 Custom Model Integration

**Implement model.LLM Interface** (`model/llm.go:26`):

```go
package mymodel

import (
    "context"
    "iter"
    "google.golang.org/adk/model"
    "google.golang.org/genai"
)

type CustomModel struct {
    apiKey string
    baseURL string
}

func NewCustomModel(apiKey, baseURL string) model.LLM {
    return &CustomModel{apiKey: apiKey, baseURL: baseURL}
}

func (m *CustomModel) Name() string {
    return "custom-model-v1"
}

func (m *CustomModel) GenerateContent(
    ctx context.Context,
    req *model.LLMRequest,
    stream bool,
) iter.Seq2[*model.LLMResponse, error] {
    return func(yield func(*model.LLMResponse, error) bool) {
        // 1. Convert ADK LLMRequest to your API format
        apiRequest := convertToAPIRequest(req)

        // 2. Call your LLM API
        if stream {
            streamResp, err := m.callAPIStreaming(ctx, apiRequest)
            if err != nil {
                yield(nil, err)
                return
            }

            // 3. Stream responses
            for chunk := range streamResp {
                adkResp := &model.LLMResponse{
                    Content: &genai.Content{
                        Role: genai.RoleModel,
                        Parts: []*genai.Part{{Text: chunk.Text}},
                    },
                    Partial:      !chunk.Done,
                    TurnComplete: chunk.Done,
                }
                if !yield(adkResp, nil) {
                    return
                }
            }
        } else {
            // 4. Non-streaming call
            apiResp, err := m.callAPI(ctx, apiRequest)
            if err != nil {
                yield(nil, err)
                return
            }

            adkResp := &model.LLMResponse{
                Content: &genai.Content{
                    Role: genai.RoleModel,
                    Parts: []*genai.Part{{Text: apiResp.Text}},
                },
                UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
                    PromptTokenCount:     apiResp.PromptTokens,
                    CandidatesTokenCount: apiResp.CompletionTokens,
                },
            }
            yield(adkResp, nil)
        }
    }
}

func (m *CustomModel) callAPI(ctx context.Context, req APIRequest) (*APIResponse, error) {
    // HTTP call to your LLM API
    // ...
}

func (m *CustomModel) callAPIStreaming(ctx context.Context, req APIRequest) (<-chan APIChunk, error) {
    // Streaming HTTP call
    // ...
}
```

**Use Custom Model**:

```go
customModel := mymodel.NewCustomModel(apiKey, baseURL)

agent, _ := llmagent.New(llmagent.Config{
    Name:  "custom_llm_agent",
    Model: customModel,
    // ...
})
```

### 9.2 Multi-Agent Architectures

**Hierarchical Delegation**:

```go
// Specialist agents
codeAgent, _ := llmagent.New(llmagent.Config{
    Name:        "code_specialist",
    Model:       model,
    Description: "Expert in code generation and debugging",
    Instruction: "You are a coding expert...",
    Tools:       []tool.Tool{codeExecutionTool},
})

dataAgent, _ := llmagent.New(llmagent.Config{
    Name:        "data_specialist",
    Model:       model,
    Description: "Expert in data analysis",
    Instruction: "You analyze data...",
    Tools:       []tool.Tool{databaseTool, chartTool},
})

writingAgent, _ := llmagent.New(llmagent.Config{
    Name:        "writing_specialist",
    Model:       model,
    Description: "Expert in content writing",
    Instruction: "You write high-quality content...",
})

// Orchestrator agent
orchestrator, _ := llmagent.New(llmagent.Config{
    Name:        "orchestrator",
    Model:       model,
    Description: "Routes tasks to specialist agents",
    Instruction: "You analyze user requests and delegate to the appropriate specialist.",
    SubAgents:   []agent.Agent{codeAgent, dataAgent, writingAgent},
})
```

**Parallel Processing + Evaluation**:

```go
// Multiple solution generators
solver1, _ := llmagent.New(llmagent.Config{Name: "solver1", ...})
solver2, _ := llmagent.New(llmagent.Config{Name: "solver2", ...})
solver3, _ := llmagent.New(llmagent.Config{Name: "solver3", ...})

// Parallel execution
parallelSolvers, _ := parallelagent.New(parallelagent.Config{
    AgentConfig: agent.Config{
        Name:      "parallel_solvers",
        SubAgents: []agent.Agent{solver1, solver2, solver3},
    },
})

// Evaluator agent
evaluator, _ := llmagent.New(llmagent.Config{
    Name:        "evaluator",
    Model:       model,
    Description: "Evaluates and selects best solution",
    Instruction: "Review the solutions and select the best one.",
})

// Sequential: solve in parallel, then evaluate
pipeline, _ := sequentialagent.New(sequentialagent.Config{
    AgentConfig: agent.Config{
        Name:      "solver_pipeline",
        SubAgents: []agent.Agent{parallelSolvers, evaluator},
    },
})
```

### 9.3 Advanced State Management

**Structured State Objects**:

```go
type UserProfile struct {
    Name       string
    Preferences map[string]string
    History    []string
}

func agentRun(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
    return func(yield func(*session.Event, error) bool) {
        // Load structured state
        profileData, _ := ctx.Session().State().Get("user_profile")
        profile := profileData.(UserProfile)

        // Modify
        profile.History = append(profile.History, "new_interaction")

        // Save
        ctx.Session().State().Set("user_profile", profile)

        // ...
    }
}
```

**State Migrations**:

```go
beforeAgent := func(ctx agent.CallbackContext) (*genai.Content, error) {
    // Migrate old state schema to new
    version, err := ctx.ReadonlyState().Get("state_version")
    if err != nil || version.(int) < 2 {
        // Perform migration
        oldData, _ := ctx.ReadonlyState().Get("old_key")
        ctx.State().Set("new_key", transformData(oldData))
        ctx.State().Set("state_version", 2)
    }
    return nil, nil
}
```

### 9.4 Instruction Providers

**Dynamic Instructions**:

```go
import "google.golang.org/adk/agent/llmagent"

type TimeAwareInstructionProvider struct{}

func (p *TimeAwareInstructionProvider) Instruction(ctx llmagent.InstructionContext) (string, error) {
    hour := time.Now().Hour()

    if hour < 12 {
        return "You are a cheerful morning assistant.", nil
    } else if hour < 18 {
        return "You are a professional afternoon assistant.", nil
    } else {
        return "You are a relaxed evening assistant.", nil
    }
}

llmagent.New(llmagent.Config{
    Name:                "time_aware_agent",
    Model:               model,
    InstructionProvider: &TimeAwareInstructionProvider{},
})
```

### 9.5 Output Schema (Structured Output)

**Force LLM to Return JSON**:

```go
import "google.golang.org/genai"

outputSchema := &genai.Schema{
    Type: genai.TypeObject,
    Properties: map[string]*genai.Schema{
        "city": {
            Type:        genai.TypeString,
            Description: "City name",
        },
        "temperature": {
            Type:        genai.TypeNumber,
            Description: "Temperature in Celsius",
        },
        "conditions": {
            Type: genai.TypeArray,
            Items: &genai.Schema{
                Type: genai.TypeString,
            },
        },
    },
    Required: []string{"city", "temperature"},
}

agent, _ := llmagent.New(llmagent.Config{
    Name:         "weather_agent",
    Model:        model,
    OutputSchema: outputSchema,
    OutputKey:    "weather_data",  // Extract from this key
})
```

**Parsing Structured Output**:

```go
for event, err := range agent.Run(ctx) {
    if event.IsFinalResponse() {
        // event.LLMResponse.Content.Parts[0] contains structured JSON
        jsonData := event.LLMResponse.Content.Parts[0].Text
        var weather WeatherData
        json.Unmarshal([]byte(jsonData), &weather)
    }
}
```

### 9.6 Include Contents (Conversation History Control)

```go
type IncludeContents string

const (
    IncludeContentsAll              IncludeContents = "all"
    IncludeContentsNone             IncludeContents = "none"
    IncludeContentsSameAgent        IncludeContents = "same_agent"
    IncludeContentsSameBranch       IncludeContents = "same_branch"
    IncludeContentsSameOrParentAgent IncludeContents = "same_or_parent_agent"
)

llmagent.New(llmagent.Config{
    Name:            "isolated_agent",
    Model:           model,
    IncludeContents: llmagent.IncludeContentsNone,  // Fresh context each time
})
```

---

## 10. Testing and Development Workflow

### 10.1 Unit Testing Agents

**Mock Context**:

```go
package myagent_test

import (
    "testing"
    "google.golang.org/adk/agent"
    "google.golang.org/adk/session"
    "google.golang.org/genai"
)

type mockInvocationContext struct {
    agent.InvocationContext
    state map[string]any
}

func (m *mockInvocationContext) Session() session.Session {
    return &mockSession{state: m.state}
}

func TestCustomAgent(t *testing.T) {
    // Create agent
    a, err := agent.New(agent.Config{
        Name: "test_agent",
        Run:  myAgentRun,
    })
    if err != nil {
        t.Fatal(err)
    }

    // Create mock context
    ctx := &mockInvocationContext{
        state: map[string]any{"key": "value"},
    }

    // Run and verify
    var events []*session.Event
    for event, err := range a.Run(ctx) {
        if err != nil {
            t.Fatal(err)
        }
        events = append(events, event)
    }

    if len(events) != 1 {
        t.Errorf("Expected 1 event, got %d", len(events))
    }

    if events[0].LLMResponse.Content.Parts[0].Text != "expected output" {
        t.Errorf("Unexpected output")
    }
}
```

### 10.2 Testing Tools

```go
package mytool_test

import (
    "testing"
    "google.golang.org/adk/tool"
    "google.golang.org/adk/tool/functiontool"
)

type mockToolContext struct {
    tool.Context
    state map[string]any
}

func TestCalculatorTool(t *testing.T) {
    calcTool, err := functiontool.New(functiontool.Config{
        Name: "calculator",
    }, calculatorHandler)
    if err != nil {
        t.Fatal(err)
    }

    input := CalculatorInput{Operation: "add", A: 5, B: 3}
    ctx := &mockToolContext{}

    output, err := calculatorHandler(ctx, input)
    if err != nil {
        t.Fatal(err)
    }

    if output.Result != 8 {
        t.Errorf("Expected 8, got %f", output.Result)
    }
}
```

### 10.3 Integration Testing with Runner

```go
func TestAgentIntegration(t *testing.T) {
    ctx := context.Background()

    // Use in-memory services
    r, err := runner.New(runner.Config{
        AppName:         "test_app",
        Agent:           myAgent,
        SessionService:  session.InMemoryService(),
        ArtifactService: artifact.InMemoryService(),
    })
    if err != nil {
        t.Fatal(err)
    }

    userMsg := &genai.Content{
        Role:  genai.RoleUser,
        Parts: []*genai.Part{{Text: "test query"}},
    }

    runConfig := agent.RunConfig{StreamingMode: agent.StreamingModeNone}

    var finalEvent *session.Event
    for event, err := range r.Run(ctx, "testuser", "testsession", userMsg, runConfig) {
        if err != nil {
            t.Fatal(err)
        }
        if event.IsFinalResponse() {
            finalEvent = event
        }
    }

    if finalEvent == nil {
        t.Fatal("No final response")
    }

    // Assert response content
    // ...
}
```

### 10.4 Debugging Tips

**Enable Verbose Logging**:

```go
beforeModel := func(ctx llmagent.BeforeModelContext, req *model.LLMRequest) (*model.LLMResponse, error) {
    log.Printf("=== LLM Request ===")
    log.Printf("Model: %s", req.Model)
    for i, content := range req.Contents {
        log.Printf("Content[%d]: Role=%s, Parts=%d", i, content.Role, len(content.Parts))
        for j, part := range content.Parts {
            if part.Text != "" {
                log.Printf("  Part[%d]: Text=%q", j, part.Text)
            } else if part.FunctionCall != nil {
                log.Printf("  Part[%d]: FunctionCall=%s", j, part.FunctionCall.Name)
            }
        }
    }
    return nil, nil
}

afterModel := func(ctx llmagent.AfterModelContext, req *model.LLMRequest, resp *model.LLMResponse) (*model.LLMResponse, error) {
    log.Printf("=== LLM Response ===")
    if resp.UsageMetadata != nil {
        log.Printf("Tokens: prompt=%d, response=%d",
            resp.UsageMetadata.PromptTokenCount,
            resp.UsageMetadata.CandidatesTokenCount)
    }
    return nil, nil
}
```

**Session Inspection**:

```go
afterAgent := func(ctx agent.CallbackContext) (*genai.Content, error) {
    log.Printf("=== Session State ===")
    for key, value := range ctx.ReadonlyState().All() {
        log.Printf("%s = %v", key, value)
    }
    return nil, nil
}
```

### 10.5 Linting and Code Quality

**golangci-lint Configuration** (`.golangci.yml`):

```yaml
version: "2"

formatters:
  enable:
    - goimports

linters:
  enable:
    - goheader

  settings:
    goheader:
      values:
        const:
          COMPANY: Google LLC
        regexp:
          YEAR: 20\d\d
      template: |-
        Copyright {{ YEAR }} {{ COMPANY }}

        Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at

            http://www.apache.org/licenses/LICENSE-2.0

        Unless required by applicable law or agreed to in writing, software
        distributed under the License is distributed on an "AS IS" BASIS,
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        See the License for the specific language governing permissions and
        limitations under the License.
```

**Run Linter**:

```bash
golangci-lint run
```

---

## 11. Troubleshooting and Best Practices

### 11.1 Common Issues

#### Issue: "Agent name cannot be 'user'"

**Cause**: Reserved keyword.

**Solution**:

```go
// BAD
agent.New(agent.Config{Name: "user", ...})

// GOOD
agent.New(agent.Config{Name: "user_agent", ...})
```

#### Issue: "Duplicate agent name in tree"

**Cause**: Two agents with same name in agent hierarchy.

**Solution**: Ensure all agent names are unique across the entire tree.

```go
// BAD
subAgent1, _ := agent.New(agent.Config{Name: "helper", ...})
subAgent2, _ := agent.New(agent.Config{Name: "helper", ...})  // Duplicate!

// GOOD
subAgent1, _ := agent.New(agent.Config{Name: "helper_1", ...})
subAgent2, _ := agent.New(agent.Config{Name: "helper_2", ...})
```

#### Issue: State not persisting across invocations

**Cause**: Using in-memory session service without persistence.

**Solution**: Implement custom `session.Service` with database backend, or use GCS artifact service for stateful data.

#### Issue: Tool not being called by LLM

**Debugging**:

1. Check tool description is clear and specific
2. Verify tool is added to agent's Tools list
3. Use BeforeModelCallback to inspect LLM request
4. Check for tool conflicts (Gemini API limitations)

**Workaround**: Use agent-as-tool pattern to combine different tool types.

### 11.2 Best Practices

#### Agent Design

✅ **DO**:
- Keep agent names short, descriptive, unique
- Write specific, one-line descriptions
- Use sub-agents for task delegation
- Implement Before/After callbacks for cross-cutting concerns

❌ **DON'T**:
- Nest agents more than 3-4 levels deep
- Create circular agent references
- Overload single agent with too many tools (>10)
- Use agent name "user" or names with special characters

#### Tool Design

✅ **DO**:
- Provide clear, specific descriptions
- Use JSON schema annotations for input validation
- Return structured outputs
- Handle errors gracefully
- Keep tools focused (single responsibility)

❌ **DON'T**:
- Create tools with side effects without documentation
- Return raw error messages to LLM (sanitize sensitive info)
- Block indefinitely in tool handlers

#### State Management

✅ **DO**:
- Use state for session-scoped data
- Use artifacts for file-like data with versioning
- Use memory for cross-session knowledge
- Document state schema

❌ **DON'T**:
- Store large blobs in state (use artifacts)
- Use state for temporary computation (use local variables)
- Assume state keys exist (check errors)

#### Performance

✅ **DO**:
- Use streaming for long responses
- Implement caching in BeforeModelCallbacks
- Use ParallelAgent for independent tasks
- Profile token usage with AfterModelCallbacks

❌ **DON'T**:
- Make synchronous external API calls in event loop
- Include entire conversation history if not needed (use IncludeContents)
- Ignore streaming errors

### 11.3 Production Considerations

**Security**:

```go
// Sanitize user input in BeforeAgent callback
beforeAgent := func(ctx agent.CallbackContext) (*genai.Content, error) {
    userInput := ctx.UserContent()
    if containsInjection(userInput) {
        return &genai.Content{
            Parts: []*genai.Part{{Text: "Input rejected due to security policy"}},
        }, nil
    }
    return nil, nil
}

// Redact sensitive info in logs
afterModel := func(ctx llmagent.AfterModelContext, req *model.LLMRequest, resp *model.LLMResponse) (*model.LLMResponse, error) {
    sanitizedResp := redactPII(resp)
    logToMonitoring(sanitizedResp)
    return nil, nil
}
```

**Rate Limiting**:

```go
var rateLimiter = rate.NewLimiter(rate.Limit(10), 20)  // 10 req/s, burst 20

beforeModel := func(ctx llmagent.BeforeModelContext, req *model.LLMRequest) (*model.LLMResponse, error) {
    if err := rateLimiter.Wait(ctx); err != nil {
        return nil, fmt.Errorf("rate limit exceeded")
    }
    return nil, nil
}
```

**Error Recovery**:

```go
func robustAgentRun(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
    return func(yield func(*session.Event, error) bool) {
        defer func() {
            if r := recover(); r != nil {
                log.Printf("Panic in agent: %v", r)
                event := session.NewEvent(ctx.InvocationID())
                event.LLMResponse = model.LLMResponse{
                    Content: &genai.Content{
                        Parts: []*genai.Part{{Text: "An error occurred. Please try again."}},
                    },
                }
                yield(event, nil)
            }
        }()

        // Normal agent logic with error handling
        // ...
    }
}
```

**Monitoring**:

```go
import (
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/trace"
)

func instrumentedAgentRun(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
    tracer := otel.Tracer("agent")

    return func(yield func(*session.Event, error) bool) {
        ctx, span := tracer.Start(ctx, "agent.run")
        defer span.End()

        span.SetAttributes(
            attribute.String("agent.name", ctx.Agent().Name()),
            attribute.String("user.id", ctx.Session().UserID()),
        )

        // Agent logic...
    }
}
```

---

## 12. Extension and Customization

### 12.1 Custom Session Service

**Implement Persistent Sessions**:

```go
package mysession

import (
    "context"
    "database/sql"
    "google.golang.org/adk/session"
)

type PostgresSessionService struct {
    db *sql.DB
}

func NewPostgresSessionService(connString string) (session.Service, error) {
    db, err := sql.Open("postgres", connString)
    if err != nil {
        return nil, err
    }
    return &PostgresSessionService{db: db}, nil
}

func (s *PostgresSessionService) Get(ctx context.Context, req *session.GetRequest) (*session.GetResponse, error) {
    // 1. Query database for session
    row := s.db.QueryRowContext(ctx,
        "SELECT id, app_name, user_id, state, events FROM sessions WHERE id=$1 AND app_name=$2 AND user_id=$3",
        req.SessionID, req.AppName, req.UserID)

    var sessionData SessionData
    err := row.Scan(&sessionData.ID, &sessionData.AppName, &sessionData.UserID, &sessionData.State, &sessionData.Events)
    if err == sql.ErrNoRows {
        // Create new session
        return s.createNewSession(ctx, req)
    } else if err != nil {
        return nil, err
    }

    // 2. Deserialize and return
    sess := deserializeSession(sessionData)
    return &session.GetResponse{Session: sess}, nil
}

func (s *PostgresSessionService) Save(ctx context.Context, req *session.SaveRequest) error {
    // Serialize session state and events
    data := serializeSession(req.Session)

    // Upsert to database
    _, err := s.db.ExecContext(ctx,
        "INSERT INTO sessions (id, app_name, user_id, state, events) VALUES ($1, $2, $3, $4, $5) ON CONFLICT (id) DO UPDATE SET state=$4, events=$5",
        data.ID, data.AppName, data.UserID, data.State, data.Events)
    return err
}

// Implement List, Delete...
```

### 12.2 Custom Artifact Service

**GCS Implementation Reference** (`artifact/gcsartifact/service.go`):

```go
// Key methods to implement:
// - Save: Upload to blob storage with versioning
// - Load: Download from blob storage
// - Versions: List all versions of an artifact
// - Delete: Remove artifact
// - List: List all artifacts in session
```

### 12.3 Custom Memory Service

**Vector Database Integration**:

```go
package mymemory

import (
    "context"
    "google.golang.org/adk/memory"
    "google.golang.org/adk/session"
)

type VectorMemoryService struct {
    vectorDB VectorDBClient
    embedder EmbeddingService
}

func (m *VectorMemoryService) AddSession(ctx context.Context, s session.Session) error {
    // 1. Extract meaningful content from session events
    var texts []string
    for event := range s.Events().All() {
        for _, part := range event.LLMResponse.Content.Parts {
            if part.Text != "" {
                texts = append(texts, part.Text)
            }
        }
    }

    // 2. Generate embeddings
    embeddings, err := m.embedder.Embed(ctx, texts)
    if err != nil {
        return err
    }

    // 3. Store in vector DB with metadata
    for i, text := range texts {
        err := m.vectorDB.Insert(ctx, VectorEntry{
            UserID:    s.UserID(),
            AppName:   s.AppName(),
            Text:      text,
            Embedding: embeddings[i],
            Timestamp: time.Now(),
        })
        if err != nil {
            return err
        }
    }

    return nil
}

func (m *VectorMemoryService) Search(ctx context.Context, req *memory.SearchRequest) (*memory.SearchResponse, error) {
    // 1. Embed query
    queryEmbedding, err := m.embedder.Embed(ctx, []string{req.Query})
    if err != nil {
        return nil, err
    }

    // 2. Vector similarity search
    results, err := m.vectorDB.Search(ctx, SearchRequest{
        Embedding: queryEmbedding[0],
        UserID:    req.UserID,
        AppName:   req.AppName,
        TopK:      5,
    })
    if err != nil {
        return nil, err
    }

    // 3. Convert to memory entries
    var memories []memory.Entry
    for _, result := range results {
        memories = append(memories, memory.Entry{
            Content: &genai.Content{
                Parts: []*genai.Part{{Text: result.Text}},
            },
            Author:    "memory",
            Timestamp: result.Timestamp,
        })
    }

    return &memory.SearchResponse{Memories: memories}, nil
}
```

### 12.4 Custom Launcher

**Create Domain-Specific Launcher**:

```go
package mylauncher

import (
    "google.golang.org/adk/cmd/launcher"
    "google.golang.org/adk/cmd/launcher/console"
    "google.golang.org/adk/cmd/launcher/universal"
)

type MyLauncher struct {
    universal.Launcher
}

func NewMyLauncher() *MyLauncher {
    l := &MyLauncher{
        Launcher: universal.NewLauncher(),
    }

    // Add custom modes
    l.AddMode("console", console.Mode())
    l.AddMode("mymode", myCustomMode())

    return l
}

func myCustomMode() universal.Mode {
    return universal.Mode{
        Name:        "mymode",
        Description: "Custom deployment mode",
        Run: func(ctx context.Context, config *launcher.Config, args []string) error {
            // Custom execution logic
            return nil
        },
    }
}
```

### 12.5 Contributing to ADK Go

**Workflow** (from `CONTRIBUTING.md`):

1. **Fork and Clone**:
   ```bash
   git clone https://github.com/your-fork/adk-go.git
   ```

2. **Create Branch**:
   ```bash
   git checkout -b feature/my-feature
   ```

3. **Make Changes**:
   - Follow [Google Go Style Guide](https://google.github.io/styleguide/go/index)
   - Add unit tests
   - Run `golangci-lint run`

4. **Test**:
   ```bash
   go test ./...
   ```

5. **Manual E2E Test**:
   ```bash
   go run ./examples/quickstart/main.go console
   # Test your changes interactively
   ```

6. **Commit and Push**:
   ```bash
   git commit -m "Add feature: description"
   git push origin feature/my-feature
   ```

7. **Create PR**:
   - Reference related issue
   - Include testing plan
   - Provide logs/screenshots

**CLA Requirement**: Sign Google CLA at https://cla.developers.google.com/

**Alignment**: Refer to [adk-python](https://github.com/google/adk-python) for validation.

---

## Appendix A: Complete Example - Multi-Agent System

```go
package main

import (
    "context"
    "fmt"
    "log"
    "os"

    "google.golang.org/adk/agent"
    "google.golang.org/adk/agent/llmagent"
    "google.golang.org/adk/agent/workflowagents/sequentialagent"
    "google.golang.org/adk/cmd/launcher"
    "google.golang.org/adk/cmd/launcher/full"
    "google.golang.org/adk/model/gemini"
    "google.golang.org/adk/tool"
    "google.golang.org/adk/tool/functiontool"
    "google.golang.org/adk/tool/geminitool"
    "google.golang.org/genai"
)

// Custom tool: Database query
type QueryInput struct {
    SQL string `json:"sql" jsonschema:"description=SQL query to execute"`
}

type QueryOutput struct {
    Rows []map[string]any `json:"rows"`
}

func queryHandler(ctx tool.Context, input QueryInput) (QueryOutput, error) {
    // Simulate database query
    return QueryOutput{
        Rows: []map[string]any{
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        },
    }, nil
}

func main() {
    ctx := context.Background()

    // Create Gemini model
    model, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{
        APIKey: os.Getenv("GOOGLE_API_KEY"),
    })
    if err != nil {
        log.Fatalf("Failed to create model: %v", err)
    }

    // Custom database tool
    dbTool, err := functiontool.New(functiontool.Config{
        Name:        "database_query",
        Description: "Executes SQL queries on the database",
    }, queryHandler)
    if err != nil {
        log.Fatal(err)
    }

    // Agent 1: Research agent with Google Search
    researchAgent, err := llmagent.New(llmagent.Config{
        Name:        "research_agent",
        Model:       model,
        Description: "Conducts web research on topics",
        Instruction: "You research topics using Google Search and provide comprehensive summaries.",
        Tools:       []tool.Tool{geminitool.GoogleSearch{}},
    })
    if err != nil {
        log.Fatal(err)
    }

    // Agent 2: Database agent
    databaseAgent, err := llmagent.New(llmagent.Config{
        Name:        "database_agent",
        Model:       model,
        Description: "Queries the database for information",
        Instruction: "You execute SQL queries and format results.",
        Tools:       []tool.Tool{dbTool},
    })
    if err != nil {
        log.Fatal(err)
    }

    // Agent 3: Synthesis agent
    synthesisAgent, err := llmagent.New(llmagent.Config{
        Name:        "synthesis_agent",
        Model:       model,
        Description: "Synthesizes information from multiple sources",
        Instruction: "You combine research findings and database results into a coherent report.",
    })
    if err != nil {
        log.Fatal(err)
    }

    // Workflow: Sequential execution
    workflow, err := sequentialagent.New(sequentialagent.Config{
        AgentConfig: agent.Config{
            Name:        "research_workflow",
            Description: "Multi-step research and reporting workflow",
            SubAgents:   []agent.Agent{researchAgent, databaseAgent, synthesisAgent},
        },
    })
    if err != nil {
        log.Fatal(err)
    }

    // Orchestrator with workflow as sub-agent
    orchestrator, err := llmagent.New(llmagent.Config{
        Name:        "orchestrator",
        Model:       model,
        Description: "Coordinates research, database queries, and synthesis",
        Instruction: "You coordinate the research workflow. Delegate to the research_workflow for comprehensive analysis.",
        SubAgents:   []agent.Agent{workflow},

        // Callbacks for monitoring
        BeforeAgentCallbacks: []agent.BeforeAgentCallback{
            func(ctx agent.CallbackContext) (*genai.Content, error) {
                log.Printf("[Orchestrator] Starting for user: %s, session: %s",
                    ctx.UserID(), ctx.SessionID())
                return nil, nil
            },
        },
        AfterAgentCallbacks: []agent.AfterAgentCallback{
            func(ctx agent.CallbackContext) (*genai.Content, error) {
                log.Printf("[Orchestrator] Completed")
                return nil, nil
            },
        },
    })
    if err != nil {
        log.Fatal(err)
    }

    // Configure launcher
    config := &launcher.Config{
        AgentLoader: agent.NewSingleLoader(orchestrator),
    }

    // Run with full launcher
    l := full.NewLauncher()
    if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
        log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
    }
}
```

**Run**:

```bash
# Console mode
go run main.go console

# REST API mode
go run main.go restapi --port 8080

# A2A mode
go run main.go a2a --port 8081

# WebUI mode
go run main.go webui --port 8080
```

---

## Appendix B: API Reference Quick Lookup

### Core Packages

| Package | Primary Types | Purpose |
|---------|---------------|---------|
| `agent` | `Agent`, `Config`, `InvocationContext` | Agent creation and execution |
| `agent/llmagent` | `Config`, `BeforeModelCallback`, `AfterModelCallback` | LLM-backed agents |
| `agent/workflowagents/*` | `sequentialagent`, `parallelagent`, `loopagent` | Orchestration patterns |
| `agent/remoteagent` | `A2AConfig`, `NewA2A` | Remote agent communication |
| `tool` | `Tool`, `Context`, `Toolset` | Tool interface and context |
| `tool/functiontool` | `Config`, `New` | Custom Go function tools |
| `tool/geminitool` | `GoogleSearch`, `New` | Gemini native tools |
| `tool/mcptoolset` | `Config`, `New` | MCP integration |
| `tool/agenttool` | `New` | Agent-as-tool wrapper |
| `session` | `Session`, `Service`, `Event`, `State` | Session and conversation management |
| `artifact` | `Service`, `SaveRequest`, `LoadRequest` | File storage with versioning |
| `memory` | `Service`, `SearchRequest`, `Entry` | Cross-session knowledge |
| `runner` | `Runner`, `Config`, `Run` | Execution engine |
| `model` | `LLM`, `LLMRequest`, `LLMResponse` | Model interface |
| `model/gemini` | `NewModel` | Gemini implementation |
| `cmd/launcher/*` | `Config`, `Execute` | Deployment launchers |
| `server/adkrest` | `NewHandler` | REST API server |
| `server/adka2a` | `NewExecutor` | A2A protocol server |

### Key Constants

```go
// genai.Role
genai.RoleUser
genai.RoleModel

// agent.StreamingMode
agent.StreamingModeNone
agent.StreamingModeText
agent.StreamingModeEvents

// llmagent.IncludeContents
llmagent.IncludeContentsAll
llmagent.IncludeContentsNone
llmagent.IncludeContentsSameAgent
llmagent.IncludeContentsSameBranch
```

---

## Appendix C: Further Resources

### Official Documentation

- **ADK Docs**: https://google.github.io/adk-docs/
- **Go Package Docs**: https://pkg.go.dev/google.golang.org/adk
- **Python ADK** (reference implementation): https://github.com/google/adk-python
- **Java ADK**: https://github.com/google/adk-java
- **ADK Web**: https://github.com/google/adk-web
- **ADK Samples**: https://github.com/google/adk-samples

### Community

- **Reddit**: https://www.reddit.com/r/agentdevelopmentkit/
- **GitHub Issues**: https://github.com/google/adk-go/issues
- **DeepWiki**: https://deepwiki.com/google/adk-go

### Related Technologies

- **Gemini API**: https://ai.google.dev/
- **A2A Protocol**: https://github.com/a2aproject
- **Model Context Protocol**: https://github.com/modelcontextprotocol

---

## Conclusion

This guide has provided comprehensive coverage of the Google Agent Development Kit for Go, from basic concepts to advanced patterns. You should now be able to:

✅ Design and implement complex multi-agent systems
✅ Create custom tools with proper schemas and error handling
✅ Manage sessions, state, artifacts, and memory
✅ Deploy agents via console, REST API, A2A protocol, or WebUI
✅ Test, debug, and monitor agent systems in production
✅ Extend ADK with custom models, services, and launchers

For continued learning:
1. Explore the `examples/` directory in the repository
2. Review the Python ADK for architectural patterns
3. Contribute to the project via GitHub
4. Join the community on Reddit and GitHub Discussions

**Remember**: ADK is designed for **code-first, production-grade agent development**. Embrace Go's type safety, concurrency, and simplicity to build robust, scalable agent systems.

---

**Document Version**: 1.0
**Last Updated**: 2025-11-19
**ADK Go Version**: Latest (google.golang.org/adk)
**License**: Apache 2.0
