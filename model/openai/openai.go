package openai

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
	"runtime"
	"sort"
	"strings"

	"github.com/sjzsdu/adk-go/internal/llminternal"
	"github.com/sjzsdu/adk-go/internal/version"
	"github.com/sjzsdu/adk-go/model"
	"google.golang.org/genai"
)

const (
	// DefaultBaseURL is the default OpenAI API endpoint
	DefaultBaseURL = "https://api.openai.com/v1"
)

// Config holds the configuration for OpenAI model initialization.
type Config struct {
	// APIKey is the OpenAI API key. If empty, it will be read from OPENAI_API_KEY environment variable.
	APIKey string
	// BaseURL is the OpenAI API base URL. If empty, it will use DefaultBaseURL.
	BaseURL string
	// Organization is the OpenAI organization ID.
	Organization string
	// HTTPClient is the HTTP client to use. If nil, http.DefaultClient will be used.
	HTTPClient *http.Client
}

// openaiModel implements the model.LLM interface for OpenAI models.
type openaiModel struct {
	name               string
	client             *http.Client
	apiKey             string
	baseURL            string
	organization       string
	versionHeaderValue string
}

// ChatMessage represents a message in the chat completion request
type ChatMessage struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	Name       string     `json:"name,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// ToolCall represents a tool call in the message
type ToolCall struct {
	ID       string   `json:"id"`
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

// Function represents a function call
type Function struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ChatCompletionRequest represents the request to OpenAI chat completions API
type ChatCompletionRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Temperature float64       `json:"temperature,omitempty"`
	TopP        float64       `json:"top_p,omitempty"`
	Stream      bool          `json:"stream,omitempty"`
	Tools       []Tool        `json:"tools,omitempty"`
}

// Tool represents a tool that can be called by the model
type Tool struct {
	Type     string       `json:"type"`
	Function FunctionTool `json:"function"`
}

// FunctionTool represents a function tool definition
type FunctionTool struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

// ChatCompletionResponse represents the response from OpenAI
type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
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

// NewModel returns [model.LLM], backed by the OpenAI API.
//
// It uses the provided modelName and configuration to initialize the underlying
// HTTP client for OpenAI API calls.
//
// An error is returned if the configuration is invalid.
func NewModel(ctx context.Context, modelName string, config Config) (model.LLM, error) {
	// Set defaults
	apiKey := config.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
		if apiKey == "" {
			return nil, fmt.Errorf("OpenAI API key is required")
		}
	}

	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}

	client := config.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	// Create header value once, when the model is created
	headerValue := fmt.Sprintf("google-adk/%s gl-go/%s", version.Version,
		strings.TrimPrefix(runtime.Version(), "go"))

	return &openaiModel{
		name:               modelName,
		client:             client,
		apiKey:             apiKey,
		baseURL:            baseURL,
		organization:       config.Organization,
		versionHeaderValue: headerValue,
	}, nil
}

func (m *openaiModel) Name() string {
	return m.name
}

// GenerateContent calls the underlying OpenAI model.
func (m *openaiModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	m.maybeAppendUserContent(req)

	if stream {
		return m.generateStream(ctx, req)
	}

	return func(yield func(*model.LLMResponse, error) bool) {
		resp, err := m.generate(ctx, req)
		yield(resp, err)
	}
}

// generate calls the model synchronously returning result from the first choice.
func (m *openaiModel) generate(ctx context.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
	chatReq, err := m.buildChatRequest(req, false)
	if err != nil {
		return nil, fmt.Errorf("failed to build chat request: %w", err)
	}

	resp, err := m.callChatAPI(ctx, chatReq)
	if err != nil {
		return nil, fmt.Errorf("failed to call OpenAI API: %w", err)
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("empty response from OpenAI")
	}

	return m.convertToLLMResponse(resp), nil
}

// generateStream returns a stream of responses from the model.
func (m *openaiModel) generateStream(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	aggregator := llminternal.NewStreamingResponseAggregator()

	return func(yield func(*model.LLMResponse, error) bool) {
		chatReq, err := m.buildChatRequest(req, true)
		if err != nil {
			yield(nil, fmt.Errorf("failed to build chat request: %w", err))
			return
		}

		// 直接在回调函数中处理yield结果，不通过返回错误来终止
		yieldFailed := false
		err = m.callChatStreamAPI(ctx, chatReq, func(resp *ChatCompletionResponse) error {
			if len(resp.Choices) == 0 {
				return nil // Skip empty responses
			}

			llmResp := m.convertToLLMResponse(resp)
			llmResp.Partial = true
			llmResp.TurnComplete = resp.Choices[0].FinishReason != ""

			// Process through aggregator
			for aggResp, err := range aggregator.ProcessResponse(ctx, m.convertToGenaiResponse(llmResp)) {
				if !yield(aggResp, err) {
					yieldFailed = true
					return io.EOF // 立即终止stream API调用
				}
			}
			return nil
		})

		// 如果yield返回false导致提前终止，直接返回而不处理错误
		if yieldFailed {
			return
		}

		if err != nil && err != io.EOF {
			yield(nil, fmt.Errorf("failed to call OpenAI streaming API: %w", err))
			return
		}

		// Send final aggregated response
		if closeResult := aggregator.Close(); closeResult != nil {
			yield(closeResult, nil)
		}
	}
}

// buildChatRequest converts ADK request to OpenAI chat request
func (m *openaiModel) buildChatRequest(req *model.LLMRequest, stream bool) (*ChatCompletionRequest, error) {
	messages, err := m.convertToOpenAIMessages(req.Contents)
	if err != nil {
		return nil, err
	}

	// 处理systemInstruction并添加到messages的开头
	hasSystemMessage := false
	if req.Config != nil && req.Config.SystemInstruction != nil {
		// 检查messages是否已经包含system消息
		for i, msg := range messages {
			if msg.Role == "system" {
				hasSystemMessage = true
				// 如果已经有system消息，更新它的内容
				systemContent := ""
				for _, part := range req.Config.SystemInstruction.Parts {
					if part.Text != "" {
						systemContent += part.Text
					}
				}
				messages[i].Content = systemContent
				break
			}
		}

		// 如果没有system消息，创建一个新的
		if !hasSystemMessage {
			systemContent := ""
			for _, part := range req.Config.SystemInstruction.Parts {
				if part.Text != "" {
					systemContent += part.Text
				}
			}
			// 将system消息插入到messages的开头
			messages = append([]ChatMessage{{
				Role:    "system",
				Content: systemContent,
			}}, messages...)
		}
	}

	chatReq := &ChatCompletionRequest{
		Model:    m.name,
		Messages: messages,
		Stream:   stream,
	}

	// Set parameters from config if available
	if req.Config != nil {
		// 处理基本参数
		if req.Config.MaxOutputTokens > 0 {
			chatReq.MaxTokens = int(req.Config.MaxOutputTokens)
		}
		if req.Config.Temperature != nil {
			chatReq.Temperature = float64(*req.Config.Temperature)
		}
		if req.Config.TopP != nil {
			chatReq.TopP = float64(*req.Config.TopP)
		}

		// 处理ResponseMIMEType（正确拼写）
		if req.Config.ResponseMIMEType == "application/json" && req.Config.ResponseSchema != nil {
			// OpenAI目前的ChatCompletionRequest结构不支持ResponseFormat字段
			// 这里保留注释供将来参考，当API支持时可以实现
		}

		// 处理TopK（OpenAI没有直接对应的参数）
		if req.Config.TopK != nil && *req.Config.TopK > 0 {
			// OpenAI API不直接支持TopK参数
			// 通常TopP参数提供更细粒度的控制，已经在上面处理
		}
	}

	// Convert tools if available
	if len(req.Tools) > 0 {
		tools, err := m.convertTools(req.Tools)
		if err != nil {
			return nil, fmt.Errorf("failed to convert tools: %w", err)
		}
		chatReq.Tools = tools
	}

	return chatReq, nil
}

// convertToOpenAIMessages converts genai.Content to OpenAI messages
func (m *openaiModel) convertToOpenAIMessages(contents []*genai.Content) ([]ChatMessage, error) {
	messages := make([]ChatMessage, 0, len(contents))

	for _, content := range contents {
		if content == nil {
			continue
		}

		role := m.convertRole(content.Role)

		var pending struct {
			msg       *ChatMessage
			textParts []string
		}

		flushPending := func() {
			if pending.msg == nil {
				return
			}
			if len(pending.textParts) > 0 {
				pending.msg.Content = strings.Join(pending.textParts, " ")
			}
			// 仅在存在文本或工具调用时追加消息，避免产生空消息
			if pending.msg.Content != "" || len(pending.msg.ToolCalls) > 0 {
				messages = append(messages, *pending.msg)
			}
			pending.msg = nil
			pending.textParts = nil
		}

		ensurePending := func() *ChatMessage {
			if pending.msg == nil {
				pending.msg = &ChatMessage{Role: role}
				pending.textParts = nil
			}
			return pending.msg
		}

		for _, part := range content.Parts {
			if part == nil {
				continue
			}
			switch {
			case part.Text != "":
				msg := ensurePending()
				pending.textParts = append(pending.textParts, part.Text)
				msg.Content = strings.Join(pending.textParts, " ")
			case part.FunctionCall != nil:
				msg := ensurePending()
				toolCall, err := m.convertFunctionCall(part.FunctionCall)
				if err != nil {
					return nil, err
				}
				msg.ToolCalls = append(msg.ToolCalls, *toolCall)
			case part.FunctionResponse != nil:
				flushPending()
				toolMsg, err := m.convertFunctionResponse(part.FunctionResponse)
				if err != nil {
					return nil, err
				}
				messages = append(messages, *toolMsg)
			}
		}

		flushPending()
	}

	return messages, nil
}

// convertRole converts genai role to OpenAI role
func (m *openaiModel) convertRole(role string) string {
	switch role {
	case "user":
		return "user"
	case "model", "assistant":
		return "assistant"
	case "system":
		return "system"
	case "function", "tool":
		return "tool"
	default:
		return "user"
	}
}

func (m *openaiModel) convertFunctionCall(fn *genai.FunctionCall) (*ToolCall, error) {
	if fn == nil {
		return nil, fmt.Errorf("function call is nil")
	}

	callID := fn.ID
	if callID == "" {
		callID = fmt.Sprintf("call_%s", fn.Name)
	}

	args := fn.Args
	if args == nil {
		args = map[string]any{}
	}

	argsBytes, err := json.Marshal(args)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal args for function %q: %w", fn.Name, err)
	}

	return &ToolCall{
		ID:   callID,
		Type: "function",
		Function: Function{
			Name:      fn.Name,
			Arguments: string(argsBytes),
		},
	}, nil
}

func (m *openaiModel) convertFunctionResponse(fr *genai.FunctionResponse) (*ChatMessage, error) {
	if fr == nil {
		return nil, fmt.Errorf("function response is nil")
	}
	if fr.ID == "" {
		return nil, fmt.Errorf("function response %q missing ID", fr.Name)
	}

	content := "{}"
	if fr.Response == nil {
		content = ""
	} else if bytes, err := json.Marshal(fr.Response); err == nil {
		content = string(bytes)
	} else {
		return nil, fmt.Errorf("failed to marshal function response %q: %w", fr.Name, err)
	}

	return &ChatMessage{
		Role:       "tool",
		Name:       fr.Name,
		ToolCallID: fr.ID,
		Content:    content,
	}, nil
}

// convertTools converts ADK tools to OpenAI tools
func (m *openaiModel) convertTools(tools map[string]any) ([]Tool, error) {
	if len(tools) == 0 {
		return nil, nil
	}

	type declProvider interface {
		Name() string
		Declaration() *genai.FunctionDeclaration
	}

	keys := make([]string, 0, len(tools))
	for name := range tools {
		keys = append(keys, name)
	}
	sort.Strings(keys)

	result := make([]Tool, 0, len(keys))

	for _, name := range keys {
		rawTool := tools[name]
		provider, ok := rawTool.(declProvider)
		if !ok {
			return nil, fmt.Errorf("tool %q does not expose a function declaration", name)
		}

		decl := provider.Declaration()
		if decl == nil {
			return nil, fmt.Errorf("tool %q has no function declaration", provider.Name())
		}

		var params map[string]any
		if decl.ParametersJsonSchema != nil {
			schemaBytes, err := json.Marshal(decl.ParametersJsonSchema)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal parameters schema for tool %q: %w", provider.Name(), err)
			}
			if len(schemaBytes) > 0 {
				if err := json.Unmarshal(schemaBytes, &params); err != nil {
					return nil, fmt.Errorf("failed to convert parameters schema for tool %q: %w", provider.Name(), err)
				}
			}
		}

		toolName := decl.Name
		if toolName == "" {
			toolName = provider.Name()
		}

		result = append(result, Tool{
			Type: "function",
			Function: FunctionTool{
				Name:        toolName,
				Description: decl.Description,
				Parameters:  params,
			},
		})
	}

	return result, nil
}

// callChatAPI makes a synchronous call to OpenAI chat API
func (m *openaiModel) callChatAPI(ctx context.Context, req *ChatCompletionRequest) (*ChatCompletionResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", m.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	m.setHeaders(httpReq)

	resp, err := m.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("OpenAI API error %d: %s", resp.StatusCode, string(body))
	}

	var chatResp ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &chatResp, nil
}

// callChatStreamAPI makes a streaming call to OpenAI chat API
func (m *openaiModel) callChatStreamAPI(ctx context.Context, req *ChatCompletionRequest, callback func(*ChatCompletionResponse) error) error {
	body, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", m.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	m.setHeaders(httpReq)

	resp, err := m.client.Do(httpReq)
	if err != nil {
		return fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("OpenAI API error %d: %s", resp.StatusCode, string(body))
	}

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var streamResp ChatCompletionResponse
		if err := json.Unmarshal([]byte(data), &streamResp); err != nil {
			continue // Skip malformed chunks
		}

		if err := callback(&streamResp); err != nil {
			return err
		}
	}

	return scanner.Err()
}

// setHeaders sets required headers for OpenAI API
func (m *openaiModel) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+m.apiKey)
	req.Header.Set("User-Agent", m.versionHeaderValue)

	if m.organization != "" {
		req.Header.Set("OpenAI-Organization", m.organization)
	}
}

// convertToLLMResponse converts OpenAI response to ADK response
func (m *openaiModel) convertToLLMResponse(resp *ChatCompletionResponse) *model.LLMResponse {
	if len(resp.Choices) == 0 {
		return &model.LLMResponse{
			ErrorMessage: "empty response from OpenAI",
		}
	}

	choice := resp.Choices[0]
	content := m.convertToGenaiContent(&choice.Message, &choice.Delta)

	// Convert finish reason
	var finishReason genai.FinishReason
	switch choice.FinishReason {
	case "stop":
		finishReason = genai.FinishReasonStop
	case "length":
		finishReason = genai.FinishReasonMaxTokens
	case "content_filter":
		finishReason = genai.FinishReasonSafety
	case "tool_calls", "function_call":
		finishReason = genai.FinishReasonStop
	default:
		finishReason = genai.FinishReasonOther
	}

	llmResponse := &model.LLMResponse{
		Content:      content,
		FinishReason: finishReason,
	}

	// Add usage metadata
	if resp.Usage.TotalTokens > 0 {
		llmResponse.UsageMetadata = &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     int32(resp.Usage.PromptTokens),
			CandidatesTokenCount: int32(resp.Usage.CompletionTokens),
			TotalTokenCount:      int32(resp.Usage.TotalTokens),
		}
	}

	return llmResponse
}

// convertToGenaiContent converts OpenAI message to genai.Content
func (m *openaiModel) convertToGenaiContent(msg *ChatMessage, delta *ChatMessage) *genai.Content {
	content := &genai.Content{
		Role:  "model",
		Parts: []*genai.Part{},
	}

	// Use delta for streaming, message for regular responses
	activeMsg := msg
	if delta != nil && (delta.Content != "" || len(delta.ToolCalls) > 0) {
		activeMsg = delta
	}

	if activeMsg.Content != "" {
		textContent := genai.NewContentFromText(activeMsg.Content, "model")
		content.Parts = append(content.Parts, textContent.Parts...)
	}

	// Convert tool calls
	for _, toolCall := range activeMsg.ToolCalls {
		if toolCall.Type == "function" {
			var args map[string]any
			_ = json.Unmarshal([]byte(toolCall.Function.Arguments), &args)

			fcPart := &genai.Part{
				FunctionCall: &genai.FunctionCall{
					ID:   toolCall.ID,
					Name: toolCall.Function.Name,
					Args: args,
				},
			}
			content.Parts = append(content.Parts, fcPart)
		}
	}

	return content
}

// convertToGenaiResponse converts LLMResponse back to genai response for aggregator
func (m *openaiModel) convertToGenaiResponse(resp *model.LLMResponse) *genai.GenerateContentResponse {
	return &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content:      resp.Content,
				FinishReason: resp.FinishReason,
			},
		},
		UsageMetadata: resp.UsageMetadata,
	}
}

// maybeAppendUserContent appends a user content, so that model can continue to output.
// 注意：在工具调用场景下，助手消息后应该跟tool消息而不是user消息
func (m *openaiModel) maybeAppendUserContent(req *model.LLMRequest) {
	// 简化实现，避免干扰工具调用流程
	// 只在消息为空时添加初始提示
	if len(req.Contents) == 0 {
		req.Contents = append(req.Contents, genai.NewContentFromText("Handle the requests as specified in the System Instruction.", "user"))
	}
	// 不再在助手消息后添加用户消息，避免破坏工具调用流程
}
