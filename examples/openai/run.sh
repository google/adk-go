#!/bin/bash

# Quick launch script for OpenAI adapter example

set -e

# Default configuration
OPENAI_BASE_URL=${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}
OPENAI_MODEL=${OPENAI_MODEL:-Qwen/Qwen3.5-4B}

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
    echo "Please make sure one of these is running:"
    echo ""
    echo "  1. vLLM via Docker (recommended):"
    echo "     docker compose up -d    # from project root"
    echo "     (default port: 8000, supports MTP speculative decoding)"
    echo ""
    echo "  2. LM Studio (port 1234):"
    echo "     export OPENAI_BASE_URL=http://localhost:1234/v1"
    echo ""
    echo "  3. Ollama (port 11434):"
    echo "     export OPENAI_BASE_URL=http://localhost:11434/v1"
    echo ""
    echo "  4. OpenAI API:"
    echo "     export OPENAI_BASE_URL=https://api.openai.com/v1"
    echo "     export OPENAI_API_KEY=sk-your-key"
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
