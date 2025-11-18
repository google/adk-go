package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

type Function struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

type Request struct {
	Model      string    `json:"model"`
	Messages   []Message `json:"messages"`
	Tools      []Tool    `json:"tools,omitempty"`
	ToolChoice string    `json:"tool_choice,omitempty"`
}

func main() {
	req := Request{
		Model: "google/gemma-3-12b",
		Messages: []Message{
			{Role: "user", Content: "What's the weather in London?"},
		},
		Tools: []Tool{{
			Type: "function",
			Function: Function{
				Name:        "get_weather",
				Description: "Get weather",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"location": map[string]any{"type": "string"},
					},
				},
			},
		}},
		ToolChoice: "auto",
	}

	jsonData, _ := json.Marshal(req)
	fmt.Println("=== Request ===")
	fmt.Println(string(jsonData))

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	httpReq, _ := http.NewRequestWithContext(ctx, "POST", "http://127.0.0.1:1234/v1/chat/completions", bytes.NewReader(jsonData))
	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	fmt.Println("\n=== Response ===")
	fmt.Println(string(body))

	var result map[string]any
	json.Unmarshal(body, &result)
	prettyJSON, _ := json.MarshalIndent(result, "", "  ")
	fmt.Println("\n=== Formatted ===")
	fmt.Println(string(prettyJSON))
}
