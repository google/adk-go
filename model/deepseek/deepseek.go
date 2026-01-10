package deepseek

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"os"

	"github.com/sjzsdu/adk-go/model"
	"github.com/sjzsdu/adk-go/model/openai"
)

const (
	// TokenEnvVarName 是DeepSeek API密钥的环境变量名
	TokenEnvVarName = "DEEPSEEK_API_KEY"
	// ModelEnvVarName 是DeepSeek模型的环境变量名
	ModelEnvVarName = "DEEPSEEK_MODEL"

	// OpenAICompatibleBaseURL 是DeepSeek的OpenAI兼容模式基础URL
	OpenAICompatibleBaseURL = "https://api.deepseek.com/v1"
	// DefaultModel 是默认使用的模型
	DefaultModel = "deepseek-chat"
)

const (
	// ModelDeepSeekChat 是DeepSeek对话模型
	ModelDeepSeekChat = "deepseek-chat"

	// ModelDeepSeekChatPro 是DeepSeek对话专业版模型
	ModelDeepSeekChatPro = "deepseek-chat-pro"

	// ModelDeepSeekCoder 是DeepSeek代码模型
	ModelDeepSeekCoder = "deepseek-coder"

	// ModelDeepSeekCoderPro 是DeepSeek代码专业版模型
	ModelDeepSeekCoderPro = "deepseek-coder-pro"

	// ModelDeepSeekMath 是DeepSeek数学模型
	ModelDeepSeekMath = "deepseek-math"
)

// Config 是DeepSeek模型的配置结构
type Config struct {
	APIKey  string
	BaseURL string
}

// Model 实现了model.LLM接口
type Model struct {
	llm model.LLM
}

// NewModel 创建一个新的DeepSeek模型实例
func NewModel(ctx context.Context, modelName string, cfg Config) (model.LLM, error) {
	// 从环境变量加载配置（如果配置为空）
	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv(TokenEnvVarName)
	}
	if modelName == "" {
		modelName = getEnvOrDefault(ModelEnvVarName, DefaultModel)
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = OpenAICompatibleBaseURL
	}

	// 验证API密钥
	if cfg.APIKey == "" {
		return nil, errors.New("API密钥不能为空，请设置DEEPSEEK_API_KEY环境变量或在配置中提供")
	}

	// 验证模型是否受支持
	supported := false
	for _, m := range GetSupportedModels() {
		if m == modelName {
			supported = true
			break
		}
	}
	if !supported {
		return nil, fmt.Errorf("不支持的模型: %s", modelName)
	}

	// 创建OpenAI兼容模式的客户端
	openaiConfig := openai.Config{
		APIKey:  cfg.APIKey,
		BaseURL: cfg.BaseURL,
	}

	openaiModel, err := openai.NewModel(ctx, modelName, openaiConfig)
	if err != nil {
		return nil, fmt.Errorf("创建OpenAI兼容客户端失败: %w", err)
	}

	return &Model{llm: openaiModel}, nil
}

// GetSupportedModels 返回DeepSeek支持的模型列表
func GetSupportedModels() []string {
	return []string{
		ModelDeepSeekChat,
		ModelDeepSeekChatPro,
		ModelDeepSeekCoder,
		ModelDeepSeekCoderPro,
		ModelDeepSeekMath,
	}
}

// Name 返回模型名称
func (m *Model) Name() string {
	return "deepseek"
}

// GenerateContent 生成内容，实现model.LLM接口
func (m *Model) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	// 调用底层OpenAI模型的GenerateContent方法
	return m.llm.GenerateContent(ctx, req, stream)
}

// getEnvOrDefault 获取环境变量值，如果不存在则返回默认值
func getEnvOrDefault(envVar, defaultValue string) string {
	value := os.Getenv(envVar)
	if value == "" {
		return defaultValue
	}
	return value
}