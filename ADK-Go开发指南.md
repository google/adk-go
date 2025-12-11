# ADK-Go 开发指南

## 1. 概述

ADK-Go 是一个强大的 AI 代理开发工具包，允许开发者构建、评估和部署复杂的 AI 代理系统。本开发指南将指导您如何使用 ADK-Go 框架开发自己的代理应用，从基础概念到高级功能。

## 2. 开发环境设置

### 2.1 安装依赖

要开始使用 ADK-Go，您需要安装 Go 1.21 或更高版本。然后，您可以通过以下命令将 ADK-Go 添加到您的项目中：

```bash
go get github.com/sjzsdu/adk-go
```

### 2.2 创建新项目

创建一个新的 Go 项目并添加 ADK-Go 依赖：

```bash
mkdir my-adk-agent
cd my-adk-agent
go mod init

# 创建一个新的 Go 模块

添加 ADK-Go 依赖项

```bash
go mod init my-adk-agent
go get github.com/sjzsdu/adk-go
```

## 3. 创建基础代理

### 3.1 简单 LLM 代理

让我们创建一个基本的 LLM 代理，该代理使用 Gemini 模型来响应查询：

```go
package main

import (
	"context"
	"log"
	"os"

	"github.com/sjzsdu/adk-go/agent/llmagent"
	"github.com/sjzsdu/adk-go/cmd/launcher/full"
	"github.com/sjzsdu/adk-go/cmd/launcher/universal"
	"github.com/sjzsdu/adk-go/model/gemini"
)

func main() {
	ctx := context.Background()

	// 创建 Gemini 模型
	model, err := gemini.New(ctx, gemini.WithModel("gemini-1.5-flash"))
	if err != nil {
		log.Fatalf("无法创建模型: %v", err)
	}

	// 创建 LLM 代理
	llmAgent, err := llmagent.New(
		llmagent.WithModel(model),
		llmagent.WithInstruction("您是一个有用的助手，能回答各种问题。"),
	)
	if err != nil {
		log.Fatalf("无法创建代理: %v", err)
	}

	// 创建配置和启动器
	config := full.NewEmptyConfig()
	config.Agent = llmAgent

	l := full.NewLauncher()
	err = l.ParseAndRun(ctx, config, os.Args[1:], universal.ErrorOnUnparsedArgs)
	if err != nil {
		log.Fatalf("运行失败: %v\n\n%s", err, l.FormatSyntax())
	}
}
```

### 3.2 运行代理

使用 CLI 模式运行代理：

```bash
export GOOGLE_API_KEY="your-api-key"
go run main.go console
```

或者使用 Web UI：

```bash
go run main.go webui
```

## 4. 实现自定义工具

工具扩展了代理的能力。让我们创建一个自定义工具，用于计算数字的平方：

### 4.1 定义工具

```go
package main

import (
	"context"
	"fmt"
	"strconv"

	"github.com/sjzsdu/adk-go/tool"
	"github.com/sjzsdu/adk-go/tool/functiontool"
)

// SquareTool 计算给定数字的平方
func SquareTool() tool.Tool {
	return functiontool.New(
		"square",
		"计算数字的平方",
		func(ctx context.Context, input map[string]any) (any, error) {
			numStr, ok := input["number"].(string)
			if !ok {
				return nil, fmt.Errorf("需要数字参数")
			}

			num, err := strconv.ParseFloat(numStr, 64)
			if err != nil {
				return nil, fmt.Errorf("无效数字: %v", err)
			}

			return map[string]any{
				"number": num,
				"square": num * num,
			}, nil
		},
		functiontool.WithSchema(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"number": map[string]any{
					"type":        "string",
					"description": "要计算平方的数字",
				},
			},
			"required": []string{"number"},
		}),
	)
}
```

### 4.2 在代理中使用工具

```go
// 创建带工具的 LLM 代理
llmAgent, err := llmagent.New(
	llmagent.WithModel(model),
	llmagent.WithInstruction("您是一个有用的助手，能回答各种问题。必要时使用工具。"),
	llmagent.WithTools(SquareTool()),
)
if err != nil {
	log.Fatalf("无法创建代理: %v", err)
}
```

## 5. 构建工作流代理

ADK-Go 支持三种工作流代理：顺序、并行和循环。

### 5.1 顺序代理

顺序执行子代理：

```go
package main

import (
	"context"
	"log"
	"os"

	"github.com/sjzsdu/adk-go/agent/llmagent"
	"github.com/sjzsdu/adk-go/agent/workflowagents/sequentialagent"
	"github.com/sjzsdu/adk-go/cmd/launcher/full"
	"github.com/sjzsdu/adk-go/cmd/launcher/universal"
	"github.com/sjzsdu/adk-go/model/gemini"
)

func main() {
	ctx := context.Background()
	model, err := gemini.New(ctx, gemini.WithModel("gemini-1.5-flash"))
	if err != nil {
		log.Fatalf("无法创建模型: %v", err)
	}

	// 创建子代理
	agent1, err := llmagent.New(
		llmagent.WithModel(model),
		llmagent.WithInstruction("将以下文本翻译成法语: %s"),
	)
	if err != nil {
		log.Fatalf("无法创建代理 1: %v", err)
	}

	agent2, err := llmagent.New(
		llmagent.WithModel(model),
		llmagent.WithInstruction("总结以下文本: %s"),
	)
	if err != nil {
		log.Fatalf("无法创建代理 2: %v", err)
	}

	// 创建顺序代理
	seqAgent, err := sequentialagent.New(
		sequentialagent.WithAgents(agent1, agent2),
	)
	if err != nil {
		log.Fatalf("无法创建顺序代理: %v", err)
	}

	// 运行代理
	config := full.NewEmptyConfig()
	config.Agent = seqAgent

	l := full.NewLauncher()
	err = l.ParseAndRun(ctx, config, os.Args[1:], universal.ErrorOnUnparsedArgs)
	if err != nil {
		log.Fatalf("运行失败: %v\n\n%s", err, l.FormatSyntax())
	}
}
```

### 5.2 并行代理

并行执行所有子代理：

```go
// 创建并行代理
parallelAgent, err := parallelagent.New(
	parallelagent.WithAgents(agent1, agent2, agent3),
)
if err != nil {
	log.Fatalf("无法创建并行代理: %v", err)
}
```

### 5.3 循环代理

重复执行子代理直到满足条件：

```go
// 创建循环代理
loopAgent, err := loopagent.New(
	loopagent.WithAgent(agent1),
	loopagent.WithMaxIterations(5),
)
if err != nil {
	log.Fatalf("无法创建循环代理: %v", err)
}
```

## 6. 集成 LLM 模型

ADK-Go 支持多种 LLM 模型。以下是如何集成不同模型的示例：

### 6.1 使用 OpenAI 模型

```go
import "github.com/sjzsdu/adk-go/model/openai"

// 创建 OpenAI 模型
model, err := openai.New(ctx, 
	openai.WithModel("gpt-4"),
	openai.WithAPIKey("your-openai-api-key"),
)
if err != nil {
	log.Fatalf("无法创建 OpenAI 模型: %v", err)
}
```

### 6.2 使用 DeepSeek 模型

```go
import "github.com/sjzsdu/adk-go/model/deepseek"

// 创建 DeepSeek 模型
model, err := deepseek.New(ctx, 
	deepseek.WithModel("deepseek-chat"),
	deepseek.WithAPIKey("your-deepseek-api-key"),
)
if err != nil {
	log.Fatalf("无法创建 DeepSeek 模型: %v", err)
}
```

### 6.3 使用 Kimi 模型

```go
import "github.com/sjzsdu/adk-go/model/kimi"

// 创建 Kimi 模型
model, err := kimi.New(ctx, 
	kimi.WithModel("moonshot-v1-8k"),
	kimi.WithAPIKey("your-kimi-api-key"),
)
if err != nil {
	log.Fatalf("无法创建 Kimi 模型: %v", err)
}
```

## 7. 调试和测试技巧

### 7.1 启用详细日志

在开发过程中启用详细日志：

```go
import "log"

func main() {
	// 设置日志级别为调试
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	// ...
}
```

### 7.2 使用示例测试

ADK-Go 包含多个示例，可用于测试和学习：

```bash
go run ./examples/quickstart/main.go console
```

### 7.3 编写单元测试

为您的代理和工具编写单元测试：

```go
package main

import (
	"context"
	"testing"

	"github.com/sjzsdu/adk-go/agent"
	"github.com/sjzsdu/adk-go/agent/llmagent"
	"github.com/sjzsdu/adk-go/model/gemini"
)

func TestMyAgent(t *testing.T) {
	ctx := context.Background()

	// 创建测试模型（可使用模拟模型）
	model, err := gemini.New(ctx, gemini.WithModel("gemini-1.5-flash"))
	if err != nil {
		t.Skip("跳过测试，需要有效的 API 密钥")
	}

	// 创建代理
	agent, err := llmagent.New(
		llmagent.WithModel(model),
		llmagent.WithInstruction("测试代理"),
	)
	if err != nil {
		t.Fatalf("无法创建代理: %v", err)
	}

	// 验证代理实现了 Agent 接口
	var _ agent.Agent = agent
}
```

## 8. 最佳实践

### 8.1 代理设计

- **单一职责原则**: 每个代理应专注于一个特定任务
- **模块化**: 将复杂代理拆分为更小、可重用的组件
- **清晰指令**: 为 LLM 代理提供明确、详细的指令
- **错误处理**: 实现适当的错误处理和重试机制

### 8.2 工具实现

- **类型安全**: 使用强类型输入和输出
- **文档化**: 为每个工具提供清晰的描述和示例
- **幂等性**: 设计幂等工具，可安全重试
- **性能考虑**: 优化长时间运行的工具，考虑异步执行

### 8.3 工作流设计

- **选择合适的工作流**: 根据任务需求选择顺序、并行或循环代理
- **限制循环迭代**: 为循环代理设置最大迭代次数，防止无限循环
- **监控执行**: 监控代理执行时间和资源使用

### 8.4 部署考虑

- **容器化**: 使用 Docker 容器化您的代理应用
- **配置外部化**: 将配置（如 API 密钥）外部化，不硬编码
- **监控**: 实现适当的监控和日志记录
- **扩展性**: 设计支持水平扩展的代理

## 9. 下一步

- 探索 [ADK-Go 示例](./examples/) 目录，了解更多用例
- 阅读 [ADK-Go 架构分析文档](./ADK-Go架构分析文档.md)，深入了解框架内部
- 查看 [ADK-Go API 参考](./ADK-Go-API参考.md)，了解详细的 API 文档
- 尝试构建更复杂的代理系统，结合多种代理类型和工具

通过遵循本指南，您应该能够使用 ADK-Go 框架开发各种复杂程度的 AI 代理应用。祝您构建出强大而灵活的 AI 代理系统！