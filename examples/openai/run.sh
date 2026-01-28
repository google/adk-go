#!/bin/bash

# Quick launch script for OpenAI adapter example

set -e

# Default configuration
OPENAI_BASE_URL=${OPENAI_BASE_URL:-http://127.0.0.1:1234/v1}
OPENAI_MODEL=${OPENAI_MODEL:-google/gemma-3-12b}

echo "🚀 Starting Weather Agent with OpenAI Adapter"
echo "📡 Endpoint: $OPENAI_BASE_URL"
echo "🤖 Model: $OPENAI_MODEL"
echo ""

# Check if local LLM is accessible
echo "🔍 Checking LLM availability..."
if curl -s --connect-timeout 2 "${OPENAI_BASE_URL}/models" > /dev/null 2>&1; then
    echo "✅ LLM server is accessible"
else
    echo "❌ Cannot connect to LLM server at $OPENAI_BASE_URL"
    echo ""
    echo "Please make sure:"
    echo "  1. LM Studio is running (default port: 1234)"
    echo "     OR"
    echo "  2. Ollama is running (default port: 11434)"
    echo "     Set: export OPENAI_BASE_URL=http://localhost:11434/v1"
    echo "     OR"
    echo "  3. Using OpenAI API"
    echo "     Set: export OPENAI_BASE_URL=https://api.openai.com/v1"
    echo "     Set: export OPENAI_API_KEY=sk-your-key"
    exit 1
fi

# Build if needed
if [ ! -f "./weather_agent" ]; then
    echo "🔨 Building weather_agent..."
    go build -o weather_agent main.go
    echo "✅ Build complete"
fi

echo ""
echo "Starting agent in console mode..."
echo "Try asking: 'What's the weather in London?'"
echo ""

# Run the agent
./weather_agent console
