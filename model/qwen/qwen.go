package qwen

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
	"os"
	"strings"

	"github.com/sjzsdu/adk-go/internal/version"
	"github.com/sjzsdu/adk-go/model"
	"google.golang.org/genai"
)

const (
	// 环境变量名
	TokenEnvVarName = "QWEN_API_KEY" //nolint:gosec
	ModelEnvVarName = "QWEN_MODEL"   //nolint:gosec

	// OpenAI兼容模式基础URL
	DefaultBaseURL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

	// 默认模型
	DefaultModel = "qwen-max"
)

// Qwen模型常量
const (
	// ModelQWenTurbo 是通义千问Turbo模型
	ModelQWenTurbo = "qwen-turbo"

	// ModelQWenPlus 是通义千问Plus模型
	ModelQWenPlus = "qwen-plus"

	// ModelQWenMax 是通义千问Max模型
	ModelQWenMax = "qwen-max"

	// ModelQWenVLPlus 是通义千问视觉Plus模型
	ModelQWenVLPlus = "qwen-vl-plus"

	// ModelQWenVLMax 是通义千问视觉Max模型
	ModelQWenVLMax = "qwen-vl-max"
)

// Config holds the configuration for Qwen model initialization.
type Config struct {
	// APIKey is the Qwen API key. If empty, it will be read from QWEN_API_KEY environment variable.
	APIKey string
	// BaseURL is the Qwen API base URL. If empty, it will use DefaultBaseURL.
	BaseURL string
	// Organization is the organization ID (optional for Qwen).
	Organization string
}

// qwenModel implements the model.LLM interface for Qwen models.
type qwenModel struct {
	name    string
	apiKey  string
	baseURL string
	client  *http.Client
}

// ChatMessage represents a message in the chat completion request
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content,omitempty"`
}

// ChatCompletionRequest represents the request to Qwen chat completions API
type ChatCompletionRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Temperature float64       `json:"temperature,omitempty"`
	TopP        float64       `json:"top_p,omitempty"`
	Stream      bool          `json:"stream,omitempty"`
}

// ChatCompletionResponse represents the response from Qwen API
type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   *Usage   `json:"usage,omitempty"`
}

// Choice represents a completion choice
type Choice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	Delta        ChatMessage `json:"delta,omitempty"`
	FinishReason string      `json:"finish_reason"`
}

// Usage represents token usage information
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// APIError represents an error response from the Qwen API
type APIError struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    int    `json:"code,omitempty"`
	} `json:"error"`
}

// NewModel returns [model.LLM], backed by the Qwen API.
//
// It uses the provided modelName and configuration to initialize the underlying
// HTTP client for Qwen API calls.
//
// An error is returned if the configuration is invalid.
func NewModel(ctx context.Context, modelName string, config Config) (model.LLM, error) {
	// Set defaults
	apiKey := config.APIKey
	if apiKey == "" {
		apiKey = os.Getenv(TokenEnvVarName)
		if apiKey == "" {
			return nil, fmt.Errorf("Qwen API key is required, set QWEN_API_KEY environment variable or provide APIKey in config")
		}
	}

	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}

	// Validate model name
	if !isValidQwenModel(modelName) {
		return nil, fmt.Errorf("unsupported Qwen model: %s", modelName)
	}

	// Create and return a new qwenModel instance
	return &qwenModel{
		name:    modelName,
		apiKey:  apiKey,
		baseURL: baseURL,
		client:  &http.Client{},
	}, nil
}

// Name returns the name of the model.
func (m *qwenModel) Name() string {
	return m.name
}

// GenerateContent implements the model.LLM interface.
func (m *qwenModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if stream {
			// 处理流式响应
			responses := m.generateStream(ctx, req)
			for resp, err := range responses {
				if !yield(resp, err) {
					return
				}
				if err != nil {
					return
				}
			}
		} else {
			// 处理非流式响应
			resp, err := m.generate(ctx, req)
			yield(resp, err)
		}
	}
}

// generate makes a non-streaming request to the Qwen API.
func (m *qwenModel) generate(ctx context.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
	chatReq, err := m.buildChatRequest(req, false)
	if err != nil {
		return nil, fmt.Errorf("failed to build chat request: %w", err)
	}

	resp, err := m.callChatAPI(ctx, chatReq)
	if err != nil {
		return nil, fmt.Errorf("API call failed: %w", err)
	}

	return m.convertToLLMResponse(resp), nil
}

// generateStream makes a streaming request to the Qwen API.
func (m *qwenModel) generateStream(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		chatReq, err := m.buildChatRequest(req, true)
		if err != nil {
			yield(nil, fmt.Errorf("failed to build chat request: %w", err))
			return
		}

		err = m.callChatStreamAPI(ctx, chatReq, func(resp *ChatCompletionResponse) error {
			llmResp := m.convertToLLMResponse(resp)
			if !yield(llmResp, nil) {
				return io.EOF // 中断迭代
			}
			return nil
		})

		if err != nil {
			yield(nil, fmt.Errorf("stream API call failed: %w", err))
		}
	}
}

// buildChatRequest converts a model.LLMRequest to a ChatCompletionRequest.
func (m *qwenModel) buildChatRequest(req *model.LLMRequest, stream bool) (*ChatCompletionRequest, error) {
	messages := make([]ChatMessage, 0, len(req.Contents))

	for _, content := range req.Contents {
		role := "user" // 默认角色
		if content.Role == "system" {
			role = "system"
		} else if content.Role == "assistant" {
			role = "assistant"
		}

		// 提取文本内容
		var textContent string
		for _, part := range content.Parts {
			if part != nil && part.Text != "" {
				textContent += part.Text
			}
		}

		messages = append(messages, ChatMessage{
			Role:    role,
			Content: textContent,
		})
	}

	// 设置默认参数
	maxTokens := 1024
	temperature := 0.7
	topP := 0.9

	if req.Config != nil {
		if req.Config.MaxOutputTokens != 0 {
			maxTokens = int(req.Config.MaxOutputTokens)
		}
		if req.Config.Temperature != nil {
			temperature = float64(*req.Config.Temperature)
		}
		if req.Config.TopP != nil {
			topP = float64(*req.Config.TopP)
		}
	}

	return &ChatCompletionRequest{
		Model:       m.name,
		Messages:    messages,
		MaxTokens:   maxTokens,
		Temperature: temperature,
		TopP:        topP,
		Stream:      stream,
	}, nil
}

// callChatAPI makes a non-streaming HTTP request to the Qwen API.
func (m *qwenModel) callChatAPI(ctx context.Context, req *ChatCompletionRequest) (*ChatCompletionResponse, error) {
	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/chat/completions", m.baseURL)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	m.setHeaders(httpReq)

	resp, err := m.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	// 读取响应体
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	// 检查响应状态码
	if resp.StatusCode != http.StatusOK {
		// 尝试解析错误响应
		var apiErr APIError
		if err := json.Unmarshal(respBody, &apiErr); err == nil {
			return nil, fmt.Errorf("Qwen API error (code %d): %s", resp.StatusCode, apiErr.Error.Message)
		}
		return nil, fmt.Errorf("unexpected status code: %d, body: %s", resp.StatusCode, string(respBody))
	}

	// 解析成功响应
	var chatResp ChatCompletionResponse
	if err := json.Unmarshal(respBody, &chatResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &chatResp, nil
}

// callChatStreamAPI makes a streaming HTTP request to the Qwen API.
func (m *qwenModel) callChatStreamAPI(ctx context.Context, req *ChatCompletionRequest, callback func(*ChatCompletionResponse) error) error {
	reqBody, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/chat/completions", m.baseURL)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(reqBody))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	m.setHeaders(httpReq)

	resp, err := m.client.Do(httpReq)
	if err != nil {
		return fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	// 检查响应状态码
	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		// 尝试解析错误响应
		var apiErr APIError
		if err := json.Unmarshal(respBody, &apiErr); err == nil {
			return fmt.Errorf("Qwen API error (code %d): %s", resp.StatusCode, apiErr.Error.Message)
		}
		return fmt.Errorf("unexpected status code: %d, body: %s", resp.StatusCode, string(respBody))
	}

	// 处理流式响应
	scanner := bufio.NewScanner(resp.Body)
	scanner.Split(bufio.ScanLines)

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		// 移除前缀 "data: "
		if strings.HasPrefix(line, "data: ") {
			line = strings.TrimPrefix(line, "data: ")
		}

		// 检查是否是结束标记
		if line == "[DONE]" {
			break
		}

		// 解析响应
		var chatResp ChatCompletionResponse
		if err := json.Unmarshal([]byte(line), &chatResp); err != nil {
			return fmt.Errorf("failed to unmarshal stream response: %w, line: %s", err, line)
		}

		// 调用回调函数
		if err := callback(&chatResp); err != nil {
			if err == io.EOF {
				break // 正常中断
			}
			return err
		}
	}

	return scanner.Err()
}

// setHeaders sets the necessary headers for the Qwen API request.
func (m *qwenModel) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", m.apiKey))
	req.Header.Set("User-Agent", fmt.Sprintf("adk-go/%s", version.Version))
}

// convertToLLMResponse converts a ChatCompletionResponse to a model.LLMResponse.
func (m *qwenModel) convertToLLMResponse(resp *ChatCompletionResponse) *model.LLMResponse {
	if len(resp.Choices) == 0 {
		return &model.LLMResponse{}
	}

	choice := resp.Choices[0]
	content := &genai.Content{
		Role: "assistant",
		Parts: []*genai.Part{
			{Text: choice.Message.Content},
		},
	}

	llmResp := &model.LLMResponse{
		Content: content,
	}

	// 添加使用统计信息
	if resp.Usage != nil {
		llmResp.UsageMetadata = &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     int32(resp.Usage.PromptTokens),
			CandidatesTokenCount: int32(resp.Usage.CompletionTokens),
			TotalTokenCount:      int32(resp.Usage.TotalTokens),
		}
	}

	return llmResp
}

// isValidQwenModel checks if the given model name is a valid Qwen model.
func isValidQwenModel(modelName string) bool {
	validModels := map[string]bool{
		ModelQWenTurbo:  true,
		ModelQWenPlus:   true,
		ModelQWenMax:    true,
		ModelQWenVLPlus: true,
		ModelQWenVLMax:  true,
	}
	return validModels[modelName]
}

// GetSupportedModels returns a list of supported Qwen models.
func GetSupportedModels() []string {
	return []string{
		ModelQWenTurbo,
		ModelQWenPlus,
		ModelQWenMax,
		ModelQWenVLPlus,
		ModelQWenVLMax,
	}
}
