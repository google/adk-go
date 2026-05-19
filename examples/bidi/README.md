# ADK Go Bidirectional Streaming Demo

This directory contains a working demonstration of real-time bidirectional streaming using the Agent Development Kit (ADK) for Go. It showcases how to set up an agent with tools and serve it over a WebSocket connection for real-time interaction.

![bidi-demo-screen](assets/bidi-demo-screen.png)

## Overview

The example in `main.go` sets up:
1.  An **LLM Agent** named `bidi-demo` using the `gemini-3.1-flash-live-preview` model.
2.  **Tools**: The agent is equipped with Google Search and a custom `camera_toggle` function tool.
3.  An **HTTP Server**: It serves a web interface and handles WebSocket connections for the bidirectional streaming API.

## Features

- **Bidirectional Streaming**: Real-time communication handled by ADK's `RuntimeAPIController`.
- **Function Calling**: Demonstrates how to register and use custom Go functions as tools.
- **Web UI**: A simple frontend to interact with the agent, located in the `static/` directory.

## Architecture

The application uses ADK's `RuntimeAPIController` to manage the bidirectional streaming session:

```
┌─────────────┐         ┌──────────────────────┐         ┌─────────────┐
│             │         │                      │         │             │
│  WebSocket  │────────▶│ RuntimeAPIController │────────▶│  Live API   │
│   Client    │         │   (RunLiveHandler)   │         │   Session   │
│             │◀────────│                      │◀────────│             │
└─────────────┘         └──────────────────────┘         └─────────────┘
```

- **Upstream**: The client sends audio, text, or media messages over the WebSocket connection.
- **Downstream**: The controller streams model responses and events back to the client in real-time.

## Prerequisites

- **Go**: Ensure you have Go installed (Go 1.23+ recommended).
- **API Key**: You need a Google API Key to access the Gemini models.

## Getting Started

### 1. Set up your API Key

Set the `GOOGLE_API_KEY` environment variable:

```bash
export GOOGLE_API_KEY="your_api_key_here"
```

### 2. Run the Server

You can run the server from the project root or directly from this directory:

**From the project root:**
```bash
go run examples/bidi/main.go
```

**From the `examples/bidi` directory:**
```bash
go run main.go
```

The server will start and serve the UI on `http://localhost:8081`.

### 3. Access the UI

Open your browser and navigate to `http://localhost:8081`. You should see a chat interface where you can interact with the agent.

## Project Structure

- `main.go`: The main entry point that configures a simple agent and starts the server.
- `static/`: Contains the frontend files (HTML, CSS, JS) shared by all examples.
- `streaming_tool/`: Demonstrates a **streaming tool** that yields data over time (counting with delays) and how to handle the `stop_streaming` control signal.
  - **Run**: `go run examples/bidi/streaming_tool/main.go`
- `sequential/`: Demonstrates a **Sequential Agent** flow where control is passed from an 'Idea Generator' agent to a 'Story Teller' agent.
  - **Run**: `go run examples/bidi/sequential/main.go`

## Code Overview

The core setup in `main.go` involves:

- Creating the model:
  ```go
  model, err := gemini.NewModel(ctx, "gemini-3.1-flash-live-preview", &genai.ClientConfig{
      APIKey: os.Getenv("GOOGLE_API_KEY"),
  })
  ```
- Defining a custom tool:
  ```go
  cameraTool, err := functiontool.New(functiontool.Config{
      Name:        "camera_toggle",
      Description: "Turns the camera on or off.",
  }, func(ctx tool.Context, args EmptyArgs) (MessageResult, error) {
      // ...
  })
  ```
- Initializing the agent:
  ```go
  a, err := llmagent.New(llmagent.Config{
      Name:        "bidi-demo",
      Model:       model,
      Instruction: "You are a real-time voice assistant.",
      Tools:       []tool.Tool{geminitool.GoogleSearch{}, cameraTool},
  })
  ```
- Serving the UI and the Live Handler:
  ```go
  controller := controllers.NewRuntimeAPIController(ss, nil, agent.NewSingleLoader(a), nil, 0, runner.PluginConfig{}, true)
  http.HandleFunc("/run_live", func(w http.ResponseWriter, req *http.Request) {
      controller.RunLiveHandler(w, req)
  })
  ```
