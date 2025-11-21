# Confirmation Example

This example demonstrates how to use the confirmation feature in FunctionTools.

## Features Demonstrated

1. **Static Confirmation**: Using `RequireConfirmation: true` in the tool config to always require confirmation
2. **Dynamic Confirmation**: Using `ctx.RequestConfirmation()` within the tool function to request confirmation at runtime

## Prerequisites

Set your Google API key:
```bash
export GOOGLE_API_KEY=your_api_key_here
```

## Running the Example

```bash
cd examples/tools/confirmation
go run main.go
```

## How It Works

1. The example creates two types of tools that perform "file write" operations:
   - A tool that uses static confirmation (defined in config)
   - A tool that requests confirmation dynamically (in the function)

2. When either tool is called by the LLM:
   - The static confirmation tool will always show a note in its description that it requires confirmation
   - The dynamic confirmation tool will pause execution when `RequestConfirmation` is called

3. The confirmation request includes the tool name, a hint, and any associated payload data

This is useful for safety-critical operations like file system modifications or system commands that should require user approval before execution.