package zhipu

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"os"

	"github.com/sjzsdu/adk-go/model"
	"github.com/sjzsdu/adk-go/model/openai"
	"google.golang.org/genai"
)

const (
	// TokenEnvVarName 是智谱API密钥的环境变量名
	TokenEnvVarName = "ZHIPU_API_KEY"
	// ModelEnvVarName 是智谱模型的环境变量名
	ModelEnvVarName = "ZHIPU_MODEL"

	// OpenAICompatibleBaseURL 是智谱的OpenAI兼容模式基础URL
	OpenAICompatibleBaseURL = "https://open.bigmodel.cn/api/paas/v4/"
	// DefaultModel 是默认使用的模型
	DefaultModel = "glm-4"
)

const (
	// ModelGLM4 是智谱GLM-4模型
	ModelGLM4 = "glm-4"

	// ModelGLM4V 是智谱GLM-4V视觉模型
	ModelGLM4V = "glm-4v"

	// ModelGLM4Air 是智谱GLM-4-Air轻量级模型
	ModelGLM4Air = "glm-4-air"

	// ModelGLM4AirX 是智谱GLM-4-AirX模型
	ModelGLM4AirX = "glm-4-airx"

	// ModelGLM4Flash 是智谱GLM-4-Flash快速模型
	ModelGLM4Flash = "glm-4-flash"

	// ModelGLM3Turbo 是智谱GLM-3-Turbo模型
	ModelGLM3Turbo = "glm-3-turbo"

	// ModelCharGLM3 是智谱CharGLM-3角色扮演模型
	ModelCharGLM3 = "charglm-3"

	// ModelCogView3 是智谱CogView-3图像生成模型
	ModelCogView3 = "cogview-3"
)

// Config 是智谱模型的配置结构
type Config struct {
	APIKey  string
	BaseURL string
}

// Model 实现了model.LLM接口
type Model struct {
	llm model.LLM
}

// NewModel 创建一个新的智谱AI模型实例
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
		// 即使从环境变量加载，我们也已经在前面的代码中检查过了
		return nil, errors.New("API密钥不能为空，请设置ZHIPU_API_KEY环境变量或在配置中提供")
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
		APIKey: cfg.APIKey,
		BaseURL: cfg.BaseURL,
	}

	openaiModel, err := openai.NewModel(ctx, modelName, openaiConfig)
	if err != nil {
		return nil, fmt.Errorf("创建OpenAI兼容客户端失败: %w", err)
	}

	return &Model{llm: openaiModel}, nil
}

// GetSupportedModels 返回智谱AI支持的模型列表
func GetSupportedModels() []string {
	return []string{
		ModelGLM4,
		ModelGLM4V,
		ModelGLM4Air,
		ModelGLM4AirX,
		ModelGLM4Flash,
		ModelGLM3Turbo,
		ModelCharGLM3,
		ModelCogView3,
	}
}

// Name 返回模型名称
func (m *Model) Name() string {
	return "zhipu"
}

// GenerateContent 生成内容，实现model.LLM接口
func (m *Model) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	// 转换包含system消息的请求
	convertedReq := convertRequest(req)
	
	// 调用底层OpenAI模型的GenerateContent方法
	return m.llm.GenerateContent(ctx, convertedReq, stream)
}

// convertRequest 转换请求，处理system消息
func convertRequest(req *model.LLMRequest) *model.LLMRequest {
	if req == nil || len(req.Contents) == 0 {
		return req
	}

	// 创建新的请求副本
	convertedReq := &model.LLMRequest{
		Model:    req.Model,
		Config:   req.Config,
		Tools:    req.Tools,
		Contents: make([]*genai.Content, 0, len(req.Contents)),
	}

	// 处理每个content
	for _, content := range req.Contents {
		if content != nil && content.Role == "system" {
			// 简单地将system消息的role改为user
			// 我们不修改parts，直接使用原始内容
			convertedContent := *content // 复制原始content
			convertedContent.Role = "user" // 修改role为user
			convertedReq.Contents = append(convertedReq.Contents, &convertedContent)
		} else {
			// 非system消息直接添加
			convertedReq.Contents = append(convertedReq.Contents, content)
		}
	}

	return convertedReq
}

// getEnvOrDefault 获取环境变量值，如果不存在则返回默认值
func getEnvOrDefault(envVar, defaultValue string) string {
	value := os.Getenv(envVar)
	if value == "" {
		return defaultValue
	}
	return value
}