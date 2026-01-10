package zhipu

import (
	"context"
	"os"
	"testing"
)

func TestNewModel(t *testing.T) {
	// 保存原始环境变量
	origAPIKey := os.Getenv(TokenEnvVarName)
	origModel := os.Getenv(ModelEnvVarName)
	defer func() {
		// 恢复原始环境变量
		os.Setenv(TokenEnvVarName, origAPIKey)
		os.Setenv(ModelEnvVarName, origModel)
	}()

	ctx := context.Background()

	t.Run("从配置创建模型", func(t *testing.T) {
		cfg := Config{
			APIKey:  "test-api-key",
			BaseURL: "http://localhost:8080",
		}
		model, err := NewModel(ctx, ModelGLM4, cfg)
		if err != nil {
			t.Errorf("NewModel() error = %v, want nil", err)
			return
		}
		if model == nil {
			t.Errorf("NewModel() returned nil model")
		}
	})

	t.Run("从环境变量创建模型", func(t *testing.T) {
		// 设置环境变量
		os.Setenv(TokenEnvVarName, "env-api-key")
		os.Setenv(ModelEnvVarName, ModelGLM3Turbo)

		cfg := Config{
			BaseURL: "http://localhost:8080",
		}
		// 空字符串modelName将从环境变量读取
		model, err := NewModel(ctx, "", cfg)
		if err != nil {
			t.Errorf("NewModel() error = %v, want nil", err)
			return
		}
		if model == nil {
			t.Errorf("NewModel() returned nil model")
		}
	})

	t.Run("缺少API密钥", func(t *testing.T) {
		// 确保环境变量也为空
		os.Unsetenv(TokenEnvVarName)
		cfg := Config{}
		model, err := NewModel(ctx, ModelGLM4, cfg)
		if err == nil {
			t.Errorf("NewModel() expected error for missing API key, got nil")
			return
		}
		if model != nil {
			t.Errorf("NewModel() should return nil for missing API key")
		}
	})

	t.Run("不支持的模型", func(t *testing.T) {
		cfg := Config{
			APIKey: "test-api-key",
		}
		model, err := NewModel(ctx, "unsupported-model", cfg)
		if err == nil {
			t.Errorf("NewModel() expected error for unsupported model, got nil")
			return
		}
		if model != nil {
			t.Errorf("NewModel() should return nil for unsupported model")
		}
	})
}

func TestGetSupportedModels(t *testing.T) {
	models := GetSupportedModels()
	if len(models) == 0 {
		t.Errorf("GetSupportedModels() returned empty list")
		return
	}

	// 验证预期的模型是否在列表中
	expectedModels := []string{
		ModelGLM4,
		ModelGLM4V,
		ModelGLM4Air,
		ModelGLM4AirX,
		ModelGLM4Flash,
		ModelGLM3Turbo,
		ModelCharGLM3,
		ModelCogView3,
	}

	for _, expected := range expectedModels {
		found := false
		for _, model := range models {
			if model == expected {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("GetSupportedModels() missing expected model: %s", expected)
		}
	}
}

func TestModel_Name(t *testing.T) {
	ctx := context.Background()
	cfg := Config{
		APIKey:  "test-api-key",
		BaseURL: "http://localhost:8080",
	}

	model, err := NewModel(ctx, ModelGLM4, cfg)
	if err != nil {
		t.Fatalf("NewModel() error = %v, want nil", err)
	}

	name := model.Name()
	if name != "zhipu" {
		t.Errorf("Model.Name() = %v, want %v", name, "zhipu")
	}
}

// TestModel_GenerateContent 是一个简单的初始化测试，实际调用需要mock
func TestModel_GenerateContent(t *testing.T) {
	ctx := context.Background()
	cfg := Config{
		APIKey:  "test-api-key",
		BaseURL: "http://localhost:8080",
	}

	model, err := NewModel(ctx, ModelGLM4, cfg)
	if err != nil {
		t.Fatalf("NewModel() error = %v, want nil", err)
	}

	// 这里只测试初始化，不执行实际的GenerateContent调用
	if model == nil {
		t.Errorf("Model initialization failed")
	}
}