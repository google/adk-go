# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agent Development Kit (ADK) for Go is a flexible, code-first framework for building, deploying, and orchestrating AI agents. It applies software development principles to agent creation with a focus on idiomatic Go, leveraging Go's strengths in concurrency and cloud-native applications.

**Key characteristics:**
- Model-agnostic (optimized for Gemini via google.golang.org/genai)
- Deployment-agnostic (strong support for containerization and Google Cloud Run)
- Uses Go 1.24.4+ features including iterators (iter.Seq2)
- Licensed under Apache 2.0
- Aligned with adk-python as the source of truth

## Development Commands

### Building and Testing
```bash
# Build all packages
go build -mod=readonly -v ./...

# Run all tests
go test -mod=readonly -v ./...

# Run tests with race detection (nightly pattern)
go test -race -mod=readonly -v -count=1 -shuffle=on ./...

# Verify dependencies
go mod tidy -diff

# Run a specific test
go test -mod=readonly -v ./path/to/package -run TestName
```

### Linting and Code Quality
```bash
# Run linter (requires golangci-lint v2.3.1+)
golangci-lint run

# The project uses .golangci.yml with:
# - goimports formatter
# - goheader linter (enforces Apache 2.0 license headers)
# - Custom staticcheck configuration
```

### Running Examples
```bash
# General pattern
go run ./examples/<example_name>/main.go [options]

# Quickstart example with different launchers
go run ./examples/quickstart/main.go help          # Show available options
go run ./examples/quickstart/main.go console       # Console mode
go run ./examples/quickstart/main.go restapi       # REST API server
go run ./examples/quickstart/main.go a2a           # Agent-to-Agent server
go run ./examples/quickstart/main.go webui         # Web UI

# Examples directory includes:
# - quickstart: Simple weather/time agent
# - tools: Custom tool examples
# - workflowagents: Loop, parallel, and sequential agent patterns
# - rest, a2a, web: Server deployment examples
# - mcp: Model Context Protocol integration
# - vertexai: Vertex AI integration
```

## Architecture

### Core Package Structure

**Primary packages** (public API):
- `agent`: Base Agent interface and constructors
  - `agent/llmagent`: LLM-powered agents with model interaction
  - `agent/remoteagent`: Agents running on remote servers
  - `agent/workflowagents`: Orchestration patterns (loopagent, parallelagent, sequentialagent)
- `session`: Session management and event handling
  - `session/database`: Persistent session storage
- `runner`: Runtime for executing agents within sessions
- `tool`: Tool interface and implementations
  - `tool/functiontool`: Custom function-based tools
  - `tool/geminitool`: Gemini-specific tools (GoogleSearch, etc.)
  - `tool/agenttool`: Tools for agent delegation
  - `tool/mcptoolset`: Model Context Protocol toolsets
  - `tool/loadartifactstool`: Artifact loading utilities
- `server`: Protocol implementations for serving agents
  - `server/adkrest`: REST API server
  - `server/adka2a`: Agent-to-Agent protocol server
- `model`: LLM model interface
  - `model/gemini`: Gemini model implementation
- `memory`: Semantic memory service interface
- `artifact`: Artifact storage interface
  - `artifact/gcsartifact`: Google Cloud Storage implementation
- `telemetry`: OpenTelemetry integration
- `util`: Shared utilities

**Internal packages** (`internal/`):
- Not part of public API; implementation details subject to change
- Includes converters, testutil, and package-specific internals

**Command-line tools** (`cmd/`):
- `cmd/launcher`: Launcher framework for different deployment modes
- `cmd/adkgo`: CLI tool

### Agent Execution Model

1. **Agent**: Core abstraction implementing Run(InvocationContext) iter.Seq2[*session.Event, error]
   - Returns an iterator sequence of events
   - Can have SubAgents for hierarchical delegation
   - Supports BeforeAgentCallbacks and AfterAgentCallbacks

2. **Runner**: Manages agent execution within a session
   - Requires: AppName, root Agent, SessionService
   - Optional: ArtifactService, MemoryService
   - Orchestrates event processing and service interactions

3. **Session**: Tracks user-agent interactions
   - Contains: SessionID, UserID, State (key-value store), Events
   - Events are immutable records of interactions
   - State is mutable for agent-specific data

4. **Tools**: Extend agent capabilities
   - Synchronous or long-running
   - Receive tool.Context with access to session state, memory search, and actions
   - Can be grouped into Toolsets with dynamic tool selection

### Launchers

Two main launcher types in `cmd/launcher`:
- **Full launcher** (`full.NewLauncher`): Supports console, REST API, A2A, and WebUI
- **Production launcher** (`prod.NewLauncher`): Supports REST API and A2A only

Typical pattern:
```go
config := &launcher.Config{
    AgentLoader: agent.NewSingleLoader(myAgent),
}
l := full.NewLauncher()
err := l.Execute(ctx, config, os.Args[1:])
```

## Code Style and Conventions

### License Headers
All new Go files must include the Apache 2.0 license header (enforced by goheader linter):
```go
// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
```

### Style Guidelines
- Follow [Google Go Style Guide](https://google.github.io/styleguide/go/index)
- Use goimports for formatting
- Write clear, descriptive test names
- Prefer table-driven tests
- Keep PRs small and focused (one concern per PR)
- Include godoc comments for all exported types, functions, and constants

### Testing Requirements
- Add unit tests for all new features and bug fixes
- Cover edge cases, error conditions, and typical use cases
- Tests should be fast, isolated, and use mocks/fixtures (no external dependencies)
- For agent changes, provide manual E2E test evidence in PR (console output or screenshots)
- Aim for high readability and maintainability

## Important Patterns

### Creating an LLM Agent
```go
agent, err := llmagent.New(llmagent.Config{
    Name:        "agent_name",
    Model:       model,                  // model.LLM interface
    Description: "Agent's purpose",
    Instruction: "Detailed instructions",
    Tools:       []tool.Tool{...},
})
```

### Creating Custom Agents
```go
agent, err := agent.New(agent.Config{
    Name:        "custom_agent",
    Description: "Custom logic agent",
    SubAgents:   []agent.Agent{...},
    Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
        // Custom implementation returning event iterator
    },
})
```

### Workflow Agents
- **LoopAgent**: Repeatedly executes sub-agents until exit condition (requires exitlooptool)
- **ParallelAgent**: Executes multiple sub-agents concurrently
- **SequentialAgent**: Executes sub-agents in order

### Event-Driven Architecture
- Agents communicate via `session.Event` objects
- Events are immutable records in the session
- Use `session.EventActions` to modify state or transfer control
- Events flow through iterators for streaming and backpressure control

## Alignment with adk-python

Per CONTRIBUTING.md, adk-python is the source of truth. When implementing features:
1. Check adk-python for reference implementation
2. Maintain API compatibility where possible
3. Adapt to Go idioms (e.g., iterators instead of async generators)
4. Validate behavior matches Python version

## Common Development Tasks

### Adding a New Tool
1. Implement `tool.Tool` interface (Name, Description, IsLongRunning)
2. For function tools, use `functiontool.New` or implement call logic
3. Add tests covering tool execution and error cases
4. Update example if the tool is user-facing

### Adding a New Agent Type
1. Consider if it should extend existing types (llmagent, workflow agents)
2. Implement `agent.Agent` interface or use `agent.New` with custom Run function
3. Ensure proper event generation and error handling
4. Add comprehensive tests including SubAgent delegation if applicable
5. Consider alignment with adk-python

### Modifying Core Packages
1. Check if changes affect public API (backward compatibility)
2. Update relevant godoc comments
3. Add migration notes if breaking changes are necessary
4. Test across multiple examples to ensure no regressions
5. Update documentation in adk-docs repository if user-facing

## Notes

- **internal/httprr** has its own license (see internal/httprr/LICENSE)
- Examples in this repo are minimal tests; see [adk-samples](https://github.com/google/adk-samples) for production-ready examples
- Official docs: https://google.github.io/adk-docs/
- Related projects: adk-python, adk-java, adk-web
