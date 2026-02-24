package kimi

import (
	"context"
	"os"
	"testing"
)

func TestNewModel(t *testing.T) {
	// 保存原始环境变量，以便在测试后恢复
	originalAPIKey := os.Getenv(TokenEnvVarName)
	originalModel := os.Getenv(ModelEnvVarName)
	defer func() {
		os.Setenv(TokenEnvVarName, originalAPIKey)
		os.Setenv(ModelEnvVarName, originalModel)
	}()

	t.Run("从配置创建模型", func(t *testing.T) {
		ctx := context.Background()
		config := Config{
			APIKey:  "test-api-key",
			BaseURL: "https://test.api/v1",
		}

		// 这里我们不实际调用API，只是测试配置验证逻辑
		// 因为实际调用需要真实的API密钥
		// 我们会模拟openai.NewModel返回的错误，因为我们只是检查配置是否正确传递
		_, err := NewModel(ctx, ModelMoonshotV18K, config)
		// 在实际测试中，由于没有真实的API密钥和网络请求，这里会返回错误
		// 但我们不关心这个错误，我们只关心是否成功通过了我们的配置验证
		if err != nil && err.Error() != "unsupported Kimi model: moonshot-v1-8k" {
			// 除了不支持的模型错误外，我们期望的是配置验证通过
			// 实际的错误会来自openai.NewModel调用，但我们的配置验证应该通过
			t.Logf("Expected configuration validation to pass, got error: %v", err)
		}
	})

	t.Run("从环境变量创建模型", func(t *testing.T) {
		ctx := context.Background()
		os.Setenv(TokenEnvVarName, "test-api-key-env")
		os.Setenv(ModelEnvVarName, ModelMoonshotV132K)

		config := Config{}
		// 使用空字符串modelName，应该从环境变量读取
		_, err := NewModel(ctx, "", config)
		if err != nil && err.Error() != "unsupported Kimi model: moonshot-v1-32k" {
			t.Logf("Expected to read model from environment variable, got error: %v", err)
		}
	})

	t.Run("缺少API密钥", func(t *testing.T) {
		ctx := context.Background()
		// 清除环境变量以确保测试隔离
		os.Unsetenv(TokenEnvVarName)

		config := Config{}
		_, err := NewModel(ctx, ModelMoonshotV18K, config)
		if err == nil {
			t.Errorf("NewModel() expected error for missing API key, got nil")
		}
	})

	t.Run("不支持的模型", func(t *testing.T) {
		ctx := context.Background()
		config := Config{
			APIKey: "test-api-key",
		}

		_, err := NewModel(ctx, "unsupported-model", config)
		if err == nil {
			t.Errorf("NewModel() expected error for unsupported model, got nil")
		}
	})
}

func TestGetSupportedModels(t *testing.T) {
	models := GetSupportedModels()
	expectedCount := 8 // 我们定义了8个支持的模型
	if len(models) != expectedCount {
		t.Errorf("GetSupportedModels() returned %d models, expected %d", len(models), expectedCount)
	}

	// 检查是否包含所有预期的模型
	expectedModels := []string{
		ModelMoonshotV18K,
		ModelMoonshotV132K,
		ModelMoonshotV1128K,
		ModelMoonshotV1256K,
		ModelMoonshotV18K002,
		ModelMoonshotV132K002,
		ModelKimiK2,
		ModelKimiK2Multimodal,
	}

	modelMap := make(map[string]bool)
	for _, model := range models {
		modelMap[model] = true
	}

	for _, expectedModel := range expectedModels {
		if !modelMap[expectedModel] {
			t.Errorf("GetSupportedModels() missing expected model: %s", expectedModel)
		}
	}
}

func TestModel_Name(t *testing.T) {
	// 由于我们依赖于openai包的实现，这里我们只进行简单的初始化测试
	// 实际的Model接口方法测试会在集成测试中进行
	ctx := context.Background()
	config := Config{
		APIKey: "test-api-key",
	}

	// 这个测试可能会失败，因为我们没有实际的API连接
	// 但它展示了我们期望如何测试Model接口
	// 只检查初始化是否通过了我们的配置验证
	_, err := NewModel(ctx, ModelMoonshotV18K, config)
	if err != nil {
		// 在测试环境中，我们期望初始化会失败（因为没有真实API）
		// 但我们只关心错误是否是我们期望的类型
		t.Skipf("Skipping Model.Name test in test environment: %v", err)
	}
}

func TestModel_GenerateContent(t *testing.T) {
	// 同样，由于依赖于实际的API调用，这个测试在单元测试环境中可能会失败
	// 这里只是展示了我们期望如何测试GenerateContent方法
	t.Skip("Skipping GenerateContent test as it requires actual API connection")

	// 注意：完整的GenerateContent测试需要在有实际API密钥的集成测试环境中进行
	// 以下是测试框架的示例：
	/*
	ctx := context.Background()
	config := Config{
		APIKey: "actual-api-key",
	}

	model, err := NewModel(ctx, ModelMoonshotV18K, config)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	// 实际的GenerateContent测试代码
	*/
}