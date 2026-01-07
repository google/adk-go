package siliconflow

import (
	"context"
	"fmt"
	"os"

	"github.com/sjzsdu/adk-go/model"
	"github.com/sjzsdu/adk-go/model/openai"
)

const (
	// 环境变量名
	TokenEnvVarName = "SILICONFLOW_API_KEY" //nolint:gosec
	ModelEnvVarName = "SILICONFLOW_MODEL"   //nolint:gosec
	// Embedding 环境变量
	EmbeddingModelEnvVarName = "SILICONFLOW_EMBEDDING_MODEL" //nolint:gosec

	// OpenAI兼容模式基础URL
	DefaultBaseURL = "https://api.siliconflow.cn/v1"
	// 默认模型
	DefaultModel = "Qwen/Qwen2.5-72B-Instruct"
)

// 文本生成模型常量
const (
	// ModelQwen2572B 是通义千问2.5-72B指令模型
	ModelQwen2572B = "Qwen/Qwen2.5-72B-Instruct"

	// ModelQwen257B 是通义千问2.5-7B指令模型
	ModelQwen257B = "Qwen/Qwen2.5-7B-Instruct"

	// ModelQwen2532B 是通义千问2.5-32B指令模型
	ModelQwen2532B = "Qwen/Qwen2.5-32B-Instruct"

	// ModelQwen2514B 是通义千问2.5-14B指令模型
	ModelQwen2514B = "Qwen/Qwen2.5-14B-Instruct"

	// ModelDeepSeekV25 是DeepSeek-V2.5模型
	ModelDeepSeekV25 = "deepseek-ai/DeepSeek-V2.5"

	// ModelDeepSeekR1 是DeepSeek-R1推理模型
	ModelDeepSeekR1 = "Pro/deepseek-ai/DeepSeek-R1"

	// ModelDeepSeekV3 是DeepSeek-V3模型
	ModelDeepSeekV3 = "deepseek-ai/DeepSeek-V3"

	// ModelInternLM25 是InternLM2.5-20B-Chat模型
	ModelInternLM25 = "internlm/internlm2_5-20b-chat"

	// ModelGLM49B 是GLM-4-9B-Chat模型
	ModelGLM49B = "ZHIPU/GLM-4-9B-Chat"

	// ModelYi34B 是Yi-1.5-34B-Chat模型
	ModelYi34B = "01-ai/Yi-1.5-34B-Chat"

	// ModelLlama370B 是Llama-3-70B-Instruct模型
	ModelLlama370B = "meta-llama/Meta-Llama-3-70B-Instruct"

	// ModelMistral7B 是Mistral-7B-Instruct模型
	ModelMistral7B = "mistralai/Mistral-7B-Instruct-v0.3"

	// ModelQwQ32B 是QwQ-32B-Preview推理模型
	ModelQwQ32B = "Qwen/QwQ-32B-Preview"
)

// 多模态模型常量
const (
	// ModelQwenVLMax 是通义千问VL-Max多模态模型
	ModelQwenVLMax = "Qwen/Qwen2-VL-72B-Instruct"

	// ModelQwenVL7B 是通义千问VL-7B多模态模型
	ModelQwenVL7B = "Qwen/Qwen2-VL-7B-Instruct"

	// ModelInternVL2 是InternVL2-26B多模态模型
	ModelInternVL2 = "OpenGVLab/InternVL2-26B"
)

// Config holds the configuration for SiliconFlow model initialization.
type Config struct {
	// APIKey is the SiliconFlow API key. If empty, it will be read from SILICONFLOW_API_KEY environment variable.
	APIKey string
	// BaseURL is the SiliconFlow API base URL. If empty, it will use DefaultBaseURL.
	BaseURL string
	// Organization is the organization ID (optional for SiliconFlow).
	Organization string
}

// NewModel returns [model.LLM], backed by the SiliconFlow API using OpenAI-compatible interface.
//
// It uses the provided modelName and configuration to initialize the underlying
// OpenAI client with SiliconFlow-specific endpoints and authentication.
//
// An error is returned if the configuration is invalid.
func NewModel(ctx context.Context, modelName string, config Config) (model.LLM, error) {
	// Set defaults
	apiKey := config.APIKey
	if apiKey == "" {
		apiKey = os.Getenv(TokenEnvVarName)
		if apiKey == "" {
			return nil, fmt.Errorf("SiliconFlow API key is required, set SILICONFLOW_API_KEY environment variable or provide APIKey in config")
		}
	}

	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}

	// Validate model name
	if !isValidSiliconFlowModel(modelName) {
		return nil, fmt.Errorf("unsupported SiliconFlow model: %s", modelName)
	}

	// Create OpenAI config with SiliconFlow-specific settings
	openaiConfig := openai.Config{
		APIKey:       apiKey,
		BaseURL:      baseURL,
		Organization: config.Organization,
	}

	// 可以添加调试信息
	// fmt.Printf("SiliconFlow: Using model %s, BaseURL: %s\n", modelName, baseURL)
	
	// Use OpenAI implementation with SiliconFlow configuration
	return openai.NewModel(ctx, modelName, openaiConfig)
}

// isValidSiliconFlowModel checks if the given model name is a valid SiliconFlow model.
func isValidSiliconFlowModel(modelName string) bool {
	validModels := map[string]bool{
		// 文本生成模型
		ModelQwen2572B:     true,
		ModelQwen257B:      true,
		ModelQwen2532B:     true,
		ModelQwen2514B:     true,
		ModelDeepSeekV25:   true,
		ModelDeepSeekR1:    true,
		ModelDeepSeekV3:    true,
		ModelInternLM25:    true,
		ModelGLM49B:        true,
		ModelYi34B:         true,
		ModelLlama370B:     true,
		ModelMistral7B:     true,
		ModelQwQ32B:        true,
		// 多模态模型
		ModelQwenVLMax:     true,
		ModelQwenVL7B:      true,
		ModelInternVL2:     true,
	}
	return validModels[modelName]
}

// GetSupportedModels returns a list of supported SiliconFlow models.
func GetSupportedModels() []string {
	return []string{
		// 文本生成模型
		ModelQwen2572B,
		ModelQwen257B,
		ModelQwen2532B,
		ModelQwen2514B,
		ModelDeepSeekV25,
		ModelDeepSeekR1,
		ModelDeepSeekV3,
		ModelInternLM25,
		ModelGLM49B,
		ModelYi34B,
		ModelLlama370B,
		ModelMistral7B,
		ModelQwQ32B,
		// 多模态模型
		ModelQwenVLMax,
		ModelQwenVL7B,
		ModelInternVL2,
	}
}