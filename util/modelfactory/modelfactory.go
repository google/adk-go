package modelfactory

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/sjzsdu/adk-go/model"
	"github.com/sjzsdu/adk-go/model/deepseek"
	"github.com/sjzsdu/adk-go/model/gemini"
	"github.com/sjzsdu/adk-go/model/kimi"
	"github.com/sjzsdu/adk-go/model/ollama"
	"github.com/sjzsdu/adk-go/model/qwen"
	"github.com/sjzsdu/adk-go/model/siliconflow"
	"github.com/sjzsdu/adk-go/model/zhipu"
	"google.golang.org/genai"
)

// Config contains model factory configuration options
type Config struct {
	ModelType string // Model type to use: gemini, kimi, qwen, siliconflow, zhipu, deepseek, ollama
	ModelName string // Specific model name to use (optional)
}

// CreateModel creates a new LLM model based on the provided configuration.
func CreateModel(ctx context.Context, cfg *Config) (model.LLM, error) {
	// 如果没有提供配置，使用默认配置
	if cfg == nil {
		cfg = &Config{
			ModelType: "gemini", // 默认使用gemini模型
			ModelName: "",       // 模型名称使用各模型的默认值
		}
	}

	log.Printf("Creating %s model...", cfg.ModelType)

	var model model.LLM
	var err error

	// 使用配置中的模型名称或回退到默认值
	modelName := cfg.ModelName

	switch cfg.ModelType {
	case "ollama":
		// Ollama model initialization
		config := ollama.Config{}
		if modelName == "" {
			modelName = ollama.DefaultModel
		}
		model, err = ollama.NewModel(ctx, modelName, config)

	case "kimi":
		// Kimi model initialization
		apiKey := os.Getenv("KIMI_API_KEY")
		if apiKey == "" {
			return nil, fmt.Errorf("KIMI_API_KEY environment variable is required")
		}
		config := kimi.Config{APIKey: apiKey}
		if modelName == "" {
			modelName = kimi.DefaultModel
		}
		model, err = kimi.NewModel(ctx, modelName, config)

	case "qwen":
		// Qwen model initialization
		apiKey := os.Getenv("QWEN_API_KEY")
		if apiKey == "" {
			return nil, fmt.Errorf("QWEN_API_KEY environment variable is required")
		}
		config := qwen.Config{APIKey: apiKey}
		if modelName == "" {
			modelName = qwen.DefaultModel
		}
		model, err = qwen.NewModel(ctx, modelName, config)

	case "siliconflow":
		// SiliconFlow model initialization
		apiKey := os.Getenv("SILICONFLOW_API_KEY")
		if apiKey == "" {
			return nil, fmt.Errorf("SILICONFLOW_API_KEY environment variable is required")
		}
		config := siliconflow.Config{APIKey: apiKey}
		if modelName == "" {
			modelName = siliconflow.DefaultModel
		}
		model, err = siliconflow.NewModel(ctx, modelName, config)

	case "zhipu":
		// Zhipu model initialization
		apiKey := os.Getenv("ZHIPU_API_KEY")
		if apiKey == "" {
			return nil, fmt.Errorf("ZHIPU_API_KEY environment variable is required")
		}
		config := zhipu.Config{APIKey: apiKey}
		if modelName == "" {
			modelName = zhipu.DefaultModel
		}
		model, err = zhipu.NewModel(ctx, modelName, config)

	case "deepseek":
		// DeepSeek model initialization
		apiKey := os.Getenv("DEEPSEEK_API_KEY")
		if apiKey == "" {
			return nil, fmt.Errorf("DEEPSEEK_API_KEY environment variable is required")
		}
		config := deepseek.Config{APIKey: apiKey}
		if modelName == "" {
			modelName = deepseek.DefaultModel
		}
		model, err = deepseek.NewModel(ctx, modelName, config)

	case "gemini":
		fallthrough
	default:
		// Default to Gemini model
		apiKey := os.Getenv("GOOGLE_API_KEY")
		if apiKey == "" {
			return nil, fmt.Errorf("GOOGLE_API_KEY environment variable is required")
		}
		config := &genai.ClientConfig{APIKey: apiKey}
		if modelName == "" {
			modelName = "gemini-2.5-flash"
		}
		model, err = gemini.NewModel(ctx, modelName, config)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to create %s model: %w", cfg.ModelType, err)
	}

	log.Printf("Successfully initialized %s model (name: %s)", cfg.ModelType, modelName)
	return model, nil
}

// MustCreateModel creates a new LLM model and panics on error.
// Useful for examples and quickstart applications where error handling is simplified.
func MustCreateModel(ctx context.Context, cfg *Config) model.LLM {
	model, err := CreateModel(ctx, cfg)
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}
	return model
}
