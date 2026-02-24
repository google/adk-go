package qwen

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
	// TokenEnvVarName 是通义千问API密钥的环境变量名
	TokenEnvVarName = "QWEN_API_KEY"
	// ModelEnvVarName 是通义千问模型的环境变量名
	ModelEnvVarName = "QWEN_MODEL"

	// OpenAICompatibleBaseURL 是通义千问的OpenAI兼容模式基础URL
	OpenAICompatibleBaseURL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
	// DefaultModel 是默认使用的模型
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

// Config 是通义千问模型的配置结构
type Config struct {
	// APIKey 是通义千问API密钥
	APIKey string
	// BaseURL 是API基础URL，默认使用OpenAI兼容模式URL
	BaseURL string
}

// Model 实现了model.LLM接口
type Model struct {
	llm model.LLM
}

// NewModel 创建一个新的通义千问模型实例
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
		return nil, errors.New("API密钥不能为空，请设置QWEN_API_KEY环境变量或在配置中提供")
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

// GetSupportedModels 返回通义千问支持的模型列表
func GetSupportedModels() []string {
	return []string{
		ModelQWenTurbo,
		ModelQWenPlus,
		ModelQWenMax,
		ModelQWenVLPlus,
		ModelQWenVLMax,
	}
}

// Name 返回模型名称
func (m *Model) Name() string {
	return "qwen"
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
