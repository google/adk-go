package kimi

import (
	"context"
	"fmt"
	"os"

	"github.com/sjzsdu/adk-go/model"
	"github.com/sjzsdu/adk-go/model/openai"
)

const (
	// 环境变量名
	TokenEnvVarName = "KIMI_API_KEY" //nolint:gosec
	ModelEnvVarName = "KIMI_MODEL"   //nolint:gosec

	// OpenAI兼容模式基础URL
	DefaultBaseURL = "https://api.moonshot.cn/v1"

	// 默认模型
	DefaultModel = "moonshot-v1-8k"
)

// Kimi模型常量
const (
	// ModelMoonshotV18K 是Moonshot V1 8K模型
	ModelMoonshotV18K = "moonshot-v1-8k"

	// ModelMoonshotV132K 是Moonshot V1 32K模型
	ModelMoonshotV132K = "moonshot-v1-32k"

	// ModelMoonshotV1128K 是Moonshot V1 128K模型
	ModelMoonshotV1128K = "moonshot-v1-128k"

	// ModelMoonshotV1256K 是Moonshot V1 256K模型
	ModelMoonshotV1256K = "moonshot-v1-256k"

	// ModelMoonshotV18K002 是Moonshot V1 8K 002模型
	ModelMoonshotV18K002 = "moonshot-v1-8k-002"

	// ModelMoonshotV132K002 是Moonshot V1 32K 002模型
	ModelMoonshotV132K002 = "moonshot-v1-32k-002"

	// ModelKimiK2 是Kimi K2模型
	ModelKimiK2 = "kimi-k2"

	// ModelKimiK2Multimodal 是Kimi K2多模态模型
	ModelKimiK2Multimodal = "kimi-k2-multimodal"
)

// Config holds the configuration for Kimi model initialization.
type Config struct {
	// APIKey is the Kimi API key. If empty, it will be read from KIMI_API_KEY environment variable.
	APIKey string
	// BaseURL is the Kimi API base URL. If empty, it will use DefaultBaseURL.
	BaseURL string
	// Organization is the organization ID (optional for Kimi).
	Organization string
}

// NewModel returns [model.LLM], backed by the Kimi API using OpenAI-compatible interface.
//
// It uses the provided modelName and configuration to initialize the underlying
// OpenAI client with Kimi-specific endpoints and authentication.
//
// An error is returned if the configuration is invalid.
func NewModel(ctx context.Context, modelName string, config Config) (model.LLM, error) {
	// Set defaults
	apiKey := config.APIKey
	if apiKey == "" {
		apiKey = os.Getenv(TokenEnvVarName)
		if apiKey == "" {
			return nil, fmt.Errorf("Kimi API key is required, set KIMI_API_KEY environment variable or provide APIKey in config")
		}
	}

	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}

	// If model name is empty, try to get from environment variable
	if modelName == "" {
		modelName = os.Getenv(ModelEnvVarName)
		if modelName == "" {
			modelName = DefaultModel
		}
	}

	// Validate model name
	if !isValidKimiModel(modelName) {
		return nil, fmt.Errorf("unsupported Kimi model: %s", modelName)
	}

	// Create OpenAI config with Kimi-specific settings
	openaiConfig := openai.Config{
		APIKey:       apiKey,
		BaseURL:      baseURL,
		Organization: config.Organization,
	}

	// Use OpenAI implementation with Kimi configuration
	return openai.NewModel(ctx, modelName, openaiConfig)
}

// isValidKimiModel checks if the given model name is a valid Kimi model.
func isValidKimiModel(modelName string) bool {
	validModels := map[string]bool{
		ModelMoonshotV18K:       true,
		ModelMoonshotV132K:      true,
		ModelMoonshotV1128K:     true,
		ModelMoonshotV1256K:     true,
		ModelMoonshotV18K002:    true,
		ModelMoonshotV132K002:   true,
		ModelKimiK2:             true,
		ModelKimiK2Multimodal:   true,
	}
	return validModels[modelName]
}

// GetSupportedModels returns a list of supported Kimi models.
func GetSupportedModels() []string {
	return []string{
		ModelMoonshotV18K,
		ModelMoonshotV132K,
		ModelMoonshotV1128K,
		ModelMoonshotV1256K,
		ModelMoonshotV18K002,
		ModelMoonshotV132K002,
		ModelKimiK2,
		ModelKimiK2Multimodal,
	}
}