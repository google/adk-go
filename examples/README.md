# ADK Go Examples

This directory contains examples demonstrating various features and capabilities of the ADK (Agent Development Kit) for Go. The examples are designed to be minimal and focused, each showcasing specific scenarios or features.

**Note**: This is different from the [google/adk-samples](https://github.com/google/adk-samples) repository, which hosts more complex end-to-end samples for production use.

## Table of Contents

- [Getting Started](#getting-started)
- [Basic Examples](#basic-examples)
- [Integration Examples](#integration-examples)
- [Tool Examples](#tool-examples)
- [Workflow Agent Examples](#workflow-agent-examples)
- [Advanced Examples](#advanced-examples)
- [Launcher System](#launcher-system)

## Getting Started

Most examples can be run using the Go command:

```bash
# Run in console mode (interactive)
go run ./examples/<example-name>/main.go console

# Run with REST API server
go run ./examples/<example-name>/main.go restapi

# Run with Agent-to-Agent (A2A) protocol
go run ./examples/<example-name>/main.go a2a

# Run with Web UI
go run ./examples/<example-name>/main.go webui

# Get help for available commands
go run ./examples/<example-name>/main.go help
```

Before running examples, ensure you have:

- Set up your `GOOGLE_API_KEY` or Google Cloud credentials
- Installed required dependencies (`go mod download`)

## Basic Examples

### [quickstart](./quickstart)

A minimal example to get started with ADK Go. Demonstrates:

- Creating a simple LLM agent
- Using Gemini tools (code execution, search)
- Basic launcher configuration

**Run it:**

```bash
go run ./examples/quickstart/main.go console
```

## Integration Examples

### [a2a](./a2a)

Demonstrates the Agent-to-Agent (A2A) protocol integration. Shows:

- Setting up A2A server and client
- Remote agent communication
- Multi-agent collaboration

**Run it:**

```bash
# Start the A2A server
go run ./examples/a2a/main.go a2a

# In another terminal, connect as client
go run ./examples/a2a/main.go a2a --client
```

### [mcp](./mcp)

Model Context Protocol (MCP) integration example. Features:

- Connecting to MCP servers
- Using MCP tools with ADK agents
- GitHub integration via MCP

**Prerequisites:** Configure MCP server settings before running.

### [web](./web)

Web-based agent interface example. Demonstrates:

- Building web UI for agents
- Artifact handling (code, text, images)
- Interactive agent conversations

**Run it:**

```bash
go run ./examples/web/main.go webui
# Open http://localhost:8080 in your browser
```

## Tool Examples

### [tools/multipletools](./tools/multipletools)

Shows how to register and use multiple custom tools with an agent.

**Key concepts:**

- Tool registration
- Tool parameter handling
- Multiple tool coordination

### [tools/loadartifacts](./tools/loadartifacts)

Demonstrates artifact loading and manipulation.

**Key concepts:**

- Artifact creation and storage
- Loading artifacts into agent context
- Artifact type handling

## Workflow Agent Examples

These examples demonstrate different agent workflow patterns.

### [workflowagents/sequential](./workflowagents/sequential)

Sequential workflow execution pattern.

**Use case:** Tasks that must be executed in order, where each step depends on the previous one.

### [workflowagents/parallel](./workflowagents/parallel)

Parallel workflow execution pattern.

**Use case:** Independent tasks that can run simultaneously for better performance.

### [workflowagents/loop](./workflowagents/loop)

Iterative workflow pattern with loops.

**Use case:** Repetitive tasks or retry logic until a condition is met.

### [workflowagents/sequentialCode](./workflowagents/sequentialCode)

Code-defined sequential workflow (vs configuration-based).

**Use case:** Complex workflows requiring programmatic control.

## Advanced Examples

### [vertexai](./vertexai)

Google Cloud Vertex AI integration examples.

**Subdirectories:**

- `imagegenerator`: Generate images using Vertex AI models

**Prerequisites:**

- Google Cloud project with Vertex AI API enabled
- Application Default Credentials (ADC) configured

## Launcher System

The ADK launcher provides a flexible way to run agents in different modes. Most examples use one of two launcher configurations:

### Full Launcher

```go
l := full.NewLauncher()
err = l.ParseAndRun(ctx, config, os.Args[1:], universal.ErrorOnUnparsedArgs)
if err != nil {
    log.Fatalf("run failed: %v\n\n%s", err, l.FormatSyntax())
}
```

The `full.NewLauncher()` includes all major launch modes:

- **console**: Interactive command-line interface
- **restapi**: REST API server for HTTP requests
- **a2a**: Agent-to-Agent protocol server
- **webui**: Web-based user interface (can run standalone or with restapi/a2a)

### Production Launcher

```go
l := prod.NewLauncher()
```

The `prod.NewLauncher()` includes only production-ready modes:

- **restapi**: REST API server
- **a2a**: Agent-to-Agent protocol

Use this for production deployments where console and webui are not needed.

### Getting Help

Run any example with `help` to see available options:

```bash
go run ./examples/quickstart/main.go help
```

## Common Patterns

### Environment Variables

Many examples use these environment variables:

- `GOOGLE_API_KEY`: Gemini API key for model access
- `GOOGLE_CLOUD_PROJECT`: GCP project ID (for Vertex AI)
- `PORT`: Server port (default: 8080)

### Configuration

Examples typically follow this structure:

```go
// 1. Create agent configuration
config := adk.Config{
    AgentCreator: func(ctx context.Context) (agent.Agent, error) {
        // Agent setup
    },
}

// 2. Create and configure launcher
l := full.NewLauncher()

// 3. Parse and run
err := l.ParseAndRun(ctx, config, os.Args[1:], universal.ErrorOnUnparsedArgs)
```

## Contributing

When adding new examples:

1. Keep them focused on specific features
2. Include comments explaining key concepts
3. Follow the existing launcher pattern
4. Update this README with your example
5. Test in at least one launch mode

## Additional Resources

- [ADK Go Documentation](https://pkg.go.dev/google.golang.org/adk)
- [Google AI for Developers](https://ai.google.dev/)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
