# Gemini Code Assist Context: `adk-go`

This document provides context for the `adk-go` project, the Go implementation of the Agent Development Kit (ADK).

## Project Overview

`adk-go` is a flexible and modular framework for building, deploying, and orchestrating sophisticated AI agents using Go. It is designed to be model-agnostic but is optimized for use with Google's Gemini models. The framework emphasizes a code-first approach, allowing developers to define agent logic, tools, and orchestration directly in Go.

The core concepts are:
- **Agent:** The fundamental building block, representing an entity that can perform tasks. Agents can have sub-agents, creating a hierarchy for complex workflows.
- **Tool:** A capability that an agent can use, such as a Go function, to interact with the world or perform a specific task.
- **Session:** Represents a single conversation or interaction with an agent.
- **Memory:** Provides a mechanism for agents to retain information across sessions.

## Building and Running

The project uses standard Go tooling for building and testing. The primary commands are defined in the `.github/workflows/go.yml` file.

*   **Build the project:**
    ```bash
    go build -mod=readonly -v ./...
    ```

*   **Run tests:**
    ```bash
    go test -mod=readonly -v ./...
    ```

*   **Run linter:**
    The project uses `golangci-lint` for linting.
    ```bash
    golangci-lint run
    ```

## Development Conventions

*   **Go Style:** All code should adhere to the [Google Go Style Guide](https://google.github.io/styleguide/go/index).
*   **Testing:**
    *   Unit tests are mandatory for all new features and bug fixes.
    *   Manual End-to-End (E2E) tests with verifiable evidence (screenshots, logs) are required for pull requests.
*   **Contribution:**
    *   All contributions are made via GitHub Pull Requests and require a signed Contributor License Agreement (CLA).
    *   For significant changes, it is recommended to open an issue first to discuss the proposal.
*   **Documentation:** User-facing documentation changes should be submitted to the [adk-docs](https://github.com/google/adk-docs) repository.
*   **Python ADK Parity:** The [Python ADK](https://github.com/google/adk-python) is considered the reference implementation. New features or changes should align with the Python version where applicable.
