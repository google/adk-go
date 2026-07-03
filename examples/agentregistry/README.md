# ADK Go Agent Registry Example

This directory contains a working demonstration of the ADK Go
[`agentregistry`](../../agentregistry) client. It discovers components
registered in the Google Cloud Agent Registry
(`agentregistry.googleapis.com`) and composes them into a runnable LLM agent.

## Overview

The example in `main.go`:

1.  Creates an **Agent Registry client** for a project/location, authenticated
    with Application Default Credentials (ADC).
2.  **Lists** the A2A agents registered in that project/location.
3.  Optionally builds a **sub-agent** from a registered A2A agent
    (`Client.RemoteAgent`) and/or a **toolset** from a registered MCP server
    (`Client.McpToolset`).
4.  Creates an **LLM Agent** composed of the discovered sub-agent and toolset,
    and serves it through the ADK launcher (`console` or `web`).

## Prerequisites

- **Go**: Go 1.25+.
- **A Google Cloud project** with the **Agent Registry API** enabled:

  ```bash
  gcloud services enable agentregistry.googleapis.com --project YOUR_PROJECT
  ```

  (Or enable it in the Cloud Console:
  `https://console.cloud.google.com/apis/library/agentregistry.googleapis.com`.)
- **Application Default Credentials** for a principal with access to the
  registry:

  ```bash
  gcloud auth application-default login
  ```

- **API Key** for the Gemini model (`GOOGLE_API_KEY`).

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_CLOUD_PROJECT`  | yes | The Google Cloud project ID. |
| `GOOGLE_CLOUD_LOCATION` | yes | The location, e.g. `us-central1`. |
| `GOOGLE_API_KEY`        | yes | API key for the Gemini model. |
| `AGENT_RESOURCE`        | no  | Full agent resource name to add as a sub-agent. |
| `MCP_SERVER_RESOURCE`   | no  | Full MCP server resource name to add as a toolset. |

## Getting Started

Run an interactive terminal chat (`console`):

```bash
GOOGLE_CLOUD_PROJECT=your-project \
GOOGLE_CLOUD_LOCATION=us-central1 \
go run ./examples/agentregistry console
```

On startup it prints the registered agents, for example:

```
Registered agents:
  - Workspace Agent (projects/your-project/locations/us-central1/agents/agentregistry-...)
```

Use `web` instead of `console` to serve the browser UI.

### Composing registry components

To add a discovered agent as a sub-agent and/or an MCP server as a toolset,
set the resource names:

```bash
GOOGLE_CLOUD_PROJECT=your-project \
GOOGLE_CLOUD_LOCATION=us-central1 \
AGENT_RESOURCE=projects/your-project/locations/us-central1/agents/AGENT_ID \
MCP_SERVER_RESOURCE=projects/your-project/locations/us-central1/mcpServers/SERVER_ID \
go run ./examples/agentregistry console
```

## Authentication notes

- **Registry API calls** use ADC. `New` selects the mTLS endpoint based on
  `GOOGLE_API_USE_MTLS_ENDPOINT` / `GOOGLE_API_USE_CLIENT_CERTIFICATE`.
- **Egress to endpoints**: `Client.McpToolset` authenticates `*.googleapis.com`
  targets with the registry's ADC credentials automatically. `Client.RemoteAgent`
  leaves egress auth to the caller, so invoking a registered agent whose endpoint
  requires authentication (e.g. a `*.googleapis.com` A2A endpoint) needs a
  caller-supplied client via the `agentregistry.WithA2AHTTPClient` option;
  otherwise the call may return `401`.

## Project structure

- `main.go`: entry point — creates the registry client, discovers components,
  builds the LLM agent, and starts the launcher.
