# ADK-Go 示例库详解

## 1. 概述

ADK-Go 示例库包含了一系列示例应用，用于展示 ADK-Go 框架的各种功能和使用场景。这些示例通常设计得比较简洁，专注于演示一个或几个特定的功能点。与 [google/adk-samples](https://github.com/google/adk-samples) 仓库中的复杂端到端示例不同，本示例库中的示例更加轻量级，适合用于学习和测试。

### 1.1 示例库结构

```
examples/
├── a2a/                  # A2A 代理示例
├── mcp/                  # MCP 示例
├── quickstart/           # 快速入门示例
├── rest/                 # REST API 示例
├── tools/                # 工具示例
│   ├── loadartifacts/    # 加载制品示例
│   └── multipletools/    # 多工具示例
├── vertexai/             # Vertex AI 示例
│   └── imagegenerator/   # 图像生成器示例
├── web/                  # Web UI 示例
│   ├── agents/           # Web 代理定义
│   └── web.md            # Web 示例说明
├── workflowagents/       # 工作流代理示例
│   ├── loop/             # 循环代理示例
│   ├── parallel/         # 并行代理示例
│   ├── sequential/       # 顺序代理示例
│   └── sequentialCode/   # 顺序代码代理示例
├── EXAMPLE.md            # 示例模板
└── README.md             # 示例库说明
```

### 1.2 启动器使用说明

许多示例中都使用了 ADK-Go 的启动器（Launcher），它允许您选择不同的运行方式：

```go
l := full.NewLauncher()
err = l.ParseAndRun(ctx, config, os.Args[1:], universal.ErrorOnUnparsedArgs)
if err != nil {
    log.Fatalf("run failed: %v\n\n%s", err, l.FormatSyntax())
}
```

`full.NewLauncher()` 包含了所有主要的运行方式：
- `console`：命令行方式
- `restapi`：REST API 方式
- `a2a`：A2A 代理方式
- `webui`：Web UI 方式（可以独立运行，也可以与 restapi 或 a2a 一起运行）

您也可以使用 `prod.NewLauncher()`，它只包含 restapi 和 a2a 启动器，适合生产环境使用。

运行 `go run ./example/quickstart/main.go help` 可以查看详细的使用说明。

## 2. 快速入门示例 (quickstart/)

### 2.1 示例目的

快速入门示例是一个简单的 LLM 代理应用，用于演示 ADK-Go 框架的基本使用方法。

### 2.2 示例代码

```go
package main

import (
	"context"
	"log"
	"os"

	"github.com/sjzsdu/adk-go/agent/llmagent"
	"github.com/sjzsdu/adk-go/full"
	"github.com/sjzsdu/adk-go/runner"
)

func main() {
	ctx := context.Background()

	// Create a simple LLM agent
	llmAgent, err := llmagent.New(llmagent.Config{
		Name:        "quickstart",
		Description: "A simple LLM agent",
		Model:       "gemini-1.5-flash",
		Instruction: "You are a helpful assistant.",
	})
	if err != nil {
		log.Fatalf("Failed to create LLM agent: %v", err)
	}

	// Create a runner
	runnerCfg := runner.Config{
		AppName:   "quickstart-app",
		RootAgent: llmAgent,
	}

	// Launcher handles parsing command line arguments and running the app
	l := full.NewLauncher()
	if err := l.ParseAndRun(ctx, runnerCfg, os.Args[1:]); err != nil {
		log.Fatalf("Failed to run: %v\n\n%s", err, l.FormatSyntax())
	}
}
```

### 2.3 运行说明

```bash
# 运行命令行版本
go run ./examples/quickstart/main.go console

# 运行 REST API 版本
go run ./examples/quickstart/main.go restapi

# 运行 Web UI 版本
go run ./examples/quickstart/main.go webui

# 查看帮助信息
go run ./examples/quickstart/main.go help
```

### 2.4 使用场景

- 了解 ADK-Go 框架的基本结构
- 学习如何创建简单的 LLM 代理
- 体验不同的运行方式

## 3. REST API 示例 (rest/)

### 3.1 示例目的

REST API 示例演示了如何使用 ADK-Go 框架创建一个 REST API 服务，允许客户端通过 HTTP 请求与代理交互。

### 3.2 示例代码

```go
package main

import (
	"context"
	"log"

	"github.com/sjzsdu/adk-go/agent/llmagent"
	"github.com/sjzsdu/adk-go/server/rest"
	"github.com/sjzsdu/adk-go/runner"
)

func main() {
	ctx := context.Background()

	// Create LLM agent
	llmAgent, err := llmagent.New(llmagent.Config{
		Name:        "rest-agent",
		Description: "A simple LLM agent for REST API",
		Model:       "gemini-1.5-flash",
		Instruction: "You are a helpful assistant.",
	})
	if err != nil {
		log.Fatalf("Failed to create LLM agent: %v", err)
	}

	// Create runner
	runner, err := runner.New(runner.Config{
		AppName:   "rest-app",
		RootAgent: llmAgent,
	})
	if err != nil {
		log.Fatalf("Failed to create runner: %v", err)
	}

	// Create REST server
	restServer := rest.New(runner)

	// Run server
	log.Println("Starting REST server on :8080...")
	if err := restServer.Run(rest.RunConfig{
		Port: 8080,
	}); err != nil {
		log.Fatalf("Failed to run REST server: %v", err)
	}
}
```

### 3.3 运行说明

```bash
go run ./examples/rest/main.go
```

然后可以使用 curl 或其他 HTTP 客户端与 REST API 交互：

```bash
curl -X POST http://localhost:8080/v1/run \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test-user", "session_id": "test-session", "content": "Hello, ADK-Go!"}'
```

### 3.4 使用场景

- 学习如何创建 REST API 服务
- 了解如何与其他系统集成
- 构建基于 HTTP 的代理服务

## 4. A2A 代理示例 (a2a/)

### 4.1 示例目的

A2A 代理示例演示了如何使用 ADK-Go 框架创建一个 A2A（Agent-to-Agent）代理，允许代理与远程代理通信。

### 4.2 示例代码

```go
package main

import (
	"context"
	"log"
	"os"

	"github.com/sjzsdu/adk-go/agent/remoteagent"
	"github.com/sjzsdu/adk-go/full"
	"github.com/sjzsdu/adk-go/runner"
)

func main() {
	ctx := context.Background()

	// Create a remote A2A agent
	remoteAgent, err := remoteagent.NewA2A(remoteagent.A2AConfig{
		Name:        "a2a-agent",
		Description: "A remote A2A agent",
		AgentCardSource: "https://example.com/agent-card.json", // Replace with actual agent card source
	})
	if err != nil {
		log.Fatalf("Failed to create remote A2A agent: %v", err)
	}

	// Create a runner
	runnerCfg := runner.Config{
		AppName:   "a2a-app",
		RootAgent: remoteAgent,
	}

	// Launcher handles parsing command line arguments and running the app
	l := full.NewLauncher()
	if err := l.ParseAndRun(ctx, runnerCfg, os.Args[1:]); err != nil {
		log.Fatalf("Failed to run: %v\n\n%s", err, l.FormatSyntax())
	}
}
```

### 4.3 运行说明

```bash
# 运行 A2A 代理
go run ./examples/a2a/main.go a2a

# 查看帮助信息
go run ./examples/a2a/main.go help
```

### 4.4 使用场景

- 学习如何创建远程 A2A 代理
- 了解 A2A 协议的使用
- 构建分布式代理系统

## 5. Web UI 示例 (web/)

### 5.1 示例目的

Web UI 示例演示了如何使用 ADK-Go 框架创建一个带有 Web 界面的代理应用。

### 5.2 示例结构

```
web/
├── agents/           # Web 代理定义
│   ├── image_generator.go   # 图像生成器代理
│   └── llmauditor.go        # LLM 审核器代理
├── main.go           # 主程序
└── web.md            # Web 示例说明
```

### 5.3 示例代码

主程序代码：

```go
package main

import (
	"context"
	"log"
	"os"

	"github.com/sjzsdu/adk-go/examples/web/agents"
	"github.com/sjzsdu/adk-go/full"
	"github.com/sjzsdu/adk-go/runner"
)

func main() {
	ctx := context.Background()

	// Create image generator agent
	imageGenAgent, err := agents.NewImageGenerator()
	if err != nil {
		log.Fatalf("Failed to create image generator agent: %v", err)
	}

	// Create LLM auditor agent
	auditorAgent, err := agents.NewLLMAuditor()
	if err != nil {
		log.Fatalf("Failed to create LLM auditor agent: %v", err)
	}

	// Create a runner with both agents
	runnerCfg := runner.Config{
		AppName:   "web-app",
		RootAgent: imageGenAgent, // Set image generator as root agent
	}

	// Launcher handles parsing command line arguments and running the app
	l := full.NewLauncher()
	if err := l.ParseAndRun(ctx, runnerCfg, os.Args[1:]); err != nil {
		log.Fatalf("Failed to run: %v\n\n%s", err, l.FormatSyntax())
	}
}
```

### 5.4 运行说明

```bash
# 运行 Web UI
go run ./examples/web/main.go webui

# 同时运行 Web UI 和 REST API
go run ./examples/web/main.go webui restapi

# 查看帮助信息
go run ./examples/web/main.go help
```

### 5.5 使用场景

- 学习如何创建带有 Web 界面的代理应用
- 了解如何集成多个代理
- 构建可视化的代理服务

## 6. 工作流代理示例 (workflowagents/)

### 6.1 顺序代理示例 (sequential/)

#### 6.1.1 示例目的

顺序代理示例演示了如何使用 ADK-Go 框架创建一个顺序代理，按顺序执行其子代理。

#### 6.1.2 示例代码

```go
package main

import (
	"context"
	"log"
	"os"

	"github.com/sjzsdu/adk-go/agent"
	"github.com/sjzsdu/adk-go/agent/llmagent"
	"github.com/sjzsdu/adk-go/agent/workflowagents/sequentialagent"
	"github.com/sjzsdu/adk-go/full"
	"github.com/sjzsdu/adk-go/runner"
)

func main() {
	ctx := context.Background()

	// Create first agent
	agent1, err := llmagent.New(llmagent.Config{
		Name:        "agent-1",
		Description: "First agent",
		Model:       "gemini-1.5-flash",
		Instruction: "You are the first agent. Greet the user.",
	})
	if err != nil {
		log.Fatalf("Failed to create agent 1: %v", err)
	}

	// Create second agent
	agent2, err := llmagent.New(llmagent.Config{
		Name:        "agent-2",
		Description: "Second agent",
		Model:       "gemini-1.5-flash",
		Instruction: "You are the second agent. Ask the user what they want to do.",
	})
	if err != nil {
		log.Fatalf("Failed to create agent 2: %v", err)
	}

	// Create third agent
	agent3, err := llmagent.New(llmagent.Config{
		Name:        "agent-3",
		Description: "Third agent",
		Model:       "gemini-1.5-flash",
		Instruction: "You are the third agent. Summarize the conversation.",
	})
	if err != nil {
		log.Fatalf("Failed to create agent 3: %v", err)
	}

	// Create sequential agent
	seqAgent, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{
			Name:        "sequential-agent",
			Description: "A sequential workflow agent",
			SubAgents:   []agent.Agent{agent1, agent2, agent3},
		},
	})
	if err != nil {
		log.Fatalf("Failed to create sequential agent: %v", err)
	}

	// Create a runner
	runnerCfg := runner.Config{
		AppName:   "sequential-app",
		RootAgent: seqAgent,
	}

	// Launcher handles parsing command line arguments and running the app
	l := full.NewLauncher()
	if err := l.ParseAndRun(ctx, runnerCfg, os.Args[1:]); err != nil {
		log.Fatalf("Failed to run: %v\n\n%s", err, l.FormatSyntax())
	}
}
```

#### 6.1.3 运行说明

```bash
# 运行顺序代理
go run ./examples/workflowagents/sequential/main.go console

# 查看帮助信息
go run ./examples/workflowagents/sequential/main.go help
```

#### 6.1.4 使用场景

- 学习如何创建顺序代理
- 了解工作流代理的基本概念
- 构建按顺序执行的工作流

### 6.2 并行代理示例 (parallel/)

#### 6.2.1 示例目的

并行代理示例演示了如何使用 ADK-Go 框架创建一个并行代理，并行执行其子代理。

#### 6.2.2 示例代码

```go
package main

import (
	"context"
	"log"
	"os"

	"github.com/sjzsdu/adk-go/agent"
	"github.com/sjzsdu/adk-go/agent/llmagent"
	"github.com/sjzsdu/adk-go/agent/workflowagents/parallelagent"
	"github.com/sjzsdu/adk-go/full"
	"github.com/sjzsdu/adk-go/runner"
)

func main() {
	ctx := context.Background()

	// Create multiple agents
	agents := make([]agent.Agent, 3)
	for i := 0; i < 3; i++ {
		agent, err := llmagent.New(llmagent.Config{
			Name:        fmt.Sprintf("agent-%d", i+1),
			Description: fmt.Sprintf("Agent %d", i+1),
			Model:       "gemini-1.5-flash",
			Instruction: fmt.Sprintf("You are agent %d. Provide a unique perspective on the user's query.", i+1),
		})
		if err != nil {
			log.Fatalf("Failed to create agent %d: %v", i+1, err)
		}
		agents[i] = agent
	}

	// Create parallel agent
	parallelAgent, err := parallelagent.New(parallelagent.Config{
		AgentConfig: agent.Config{
			Name:        "parallel-agent",
			Description: "A parallel workflow agent",
			SubAgents:   agents,
		},
	})
	if err != nil {
		log.Fatalf("Failed to create parallel agent: %v", err)
	}

	// Create a runner
	runnerCfg := runner.Config{
		AppName:   "parallel-app",
		RootAgent: parallelAgent,
	}

	// Launcher handles parsing command line arguments and running the app
	l := full.NewLauncher()
	if err := l.ParseAndRun(ctx, runnerCfg, os.Args[1:]); err != nil {
		log.Fatalf("Failed to run: %v\n\n%s", err, l.FormatSyntax())
	}
}
```

#### 6.2.3 运行说明

```bash
# 运行并行代理
go run ./examples/workflowagents/parallel/main.go console

# 查看帮助信息
go run ./examples/workflowagents/parallel/main.go help
```

#### 6.2.4 使用场景

- 学习如何创建并行代理
- 了解并行执行的工作流
- 构建需要多个视角的工作流

### 6.3 循环代理示例 (loop/)

#### 6.3.1 示例目的

循环代理示例演示了如何使用 ADK-Go 框架创建一个循环代理，循环执行其子代理。

#### 6.3.2 示例代码

```go
package main

import (
	"context"
	"log"
	"os"

	"github.com/sjzsdu/adk-go/agent"
	"github.com/sjzsdu/adk-go/agent/llmagent"
	"github.com/sjzsdu/adk-go/agent/workflowagents/loopagent"
	"github.com/sjzsdu/adk-go/full"
	"github.com/sjzsdu/adk-go/runner"
)

func main() {
	ctx := context.Background()

	// Create a code refinement agent
	codeAgent, err := llmagent.New(llmagent.Config{
		Name:        "code-refiner",
		Description: "Code refinement agent",
		Model:       "gemini-1.5-pro",
		Instruction: "You are a code refinement agent. Improve the provided code based on the feedback.",
	})
	if err != nil {
		log.Fatalf("Failed to create code agent: %v", err)
	}

	// Create loop agent with max iterations
	loopAgent, err := loopagent.New(loopagent.Config{
		AgentConfig: agent.Config{
			Name:        "loop-agent",
			Description: "A loop workflow agent for code refinement",
			SubAgents:   []agent.Agent{codeAgent},
		},
		MaxIterations: 5, // Max 5 iterations
	})
	if err != nil {
		log.Fatalf("Failed to create loop agent: %v", err)
	}

	// Create a runner
	runnerCfg := runner.Config{
		AppName:   "loop-app",
		RootAgent: loopAgent,
	}

	// Launcher handles parsing command line arguments and running the app
	l := full.NewLauncher()
	if err := l.ParseAndRun(ctx, runnerCfg, os.Args[1:]); err != nil {
		log.Fatalf("Failed to run: %v\n\n%s", err, l.FormatSyntax())
	}
}
```

#### 6.3.3 运行说明

```bash
# 运行循环代理
go run ./examples/workflowagents/loop/main.go console

# 查看帮助信息
go run ./examples/workflowagents/loop/main.go help
```

#### 6.3.4 使用场景

- 学习如何创建循环代理
- 了解循环执行的工作流
- 构建需要迭代优化的工作流，如代码修改

### 6.4 顺序代码代理示例 (sequentialCode/)

#### 6.4.1 示例目的

顺序代码代理示例演示了如何使用 ADK-Go 框架创建一个顺序代码代理，按顺序执行代码逻辑。

#### 6.4.2 示例代码

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/sjzsdu/adk-go/agent"
	"github.com/sjzsdu/adk-go/agent/llmagent"
	"github.com/sjzsdu/adk-go/agent/workflowagents/sequentialagent"
	"github.com/sjzsdu/adk-go/full"
	"github.com/sjzsdu/adk-go/runner"
)

func main() {
	ctx := context.Background()

	// Create code agent 1
	codeAgent1, err := agent.New(agent.Config{
		Name:        "code-agent-1",
		Description: "First code agent",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				event := session.NewEvent(ctx.InvocationID())
				event.Content = genai.NewContentFromText("Running code agent 1")
				event.Author = ctx.Agent().Name()
				yield(event, nil)
			}
		},
	})
	if err != nil {
		log.Fatalf("Failed to create code agent 1: %v", err)
	}

	// Create code agent 2
	codeAgent2, err := agent.New(agent.Config{
		Name:        "code-agent-2",
		Description: "Second code agent",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				event := session.NewEvent(ctx.InvocationID())
				event.Content = genai.NewContentFromText("Running code agent 2")
				event.Author = ctx.Agent().Name()
				yield(event, nil)
			}
		},
	})
	if err != nil {
		log.Fatalf("Failed to create code agent 2: %v", err)
	}

	// Create sequential code agent
	seqCodeAgent, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{
			Name:        "sequential-code-agent",
			Description: "A sequential code workflow agent",
			SubAgents:   []agent.Agent{codeAgent1, codeAgent2},
		},
	})
	if err != nil {
		log.Fatalf("Failed to create sequential code agent: %v", err)
	}

	// Create a runner
	runnerCfg := runner.Config{
		AppName:   "sequential-code-app",
		RootAgent: seqCodeAgent,
	}

	// Launcher handles parsing command line arguments and running the app
	l := full.NewLauncher()
	if err := l.ParseAndRun(ctx, runnerCfg, os.Args[1:]); err != nil {
		log.Fatalf("Failed to run: %v\n\n%s", err, l.FormatSyntax())
	}
}
```

#### 6.4.3 运行说明

```bash
# 运行顺序代码代理
go run ./examples/workflowagents/sequentialCode/main.go console

# 查看帮助信息
go run ./examples/workflowagents/sequentialCode/main.go help
```

#### 6.4.4 使用场景

- 学习如何创建顺序代码代理
- 了解如何在工作流中执行代码逻辑
- 构建需要按顺序执行代码的工作流

## 7. 工具示例 (tools/)

### 7.1 加载制品示例 (loadartifacts/)

#### 7.1.1 示例目的

加载制品示例演示了如何使用 ADK-Go 框架创建一个代理，加载和使用制品。

#### 7.1.2 示例代码

```go
package main

import (
	"context"
	"log"
	"os"

	"github.com/sjzsdu/adk-go/agent/llmagent"
	"github.com/sjzsdu/adk-go/full"
	"github.com/sjzsdu/adk-go/runner"
	"github.com/sjzsdu/adk-go/tool"
	"github.com/sjzsdu/adk-go/tool/builtintools"
)

func main() {
	ctx := context.Background()

	// Create load artifacts tool
	loadArtifactsTool, err := builtintools.NewLoadArtifacts()
	if err != nil {
		log.Fatalf("Failed to create load artifacts tool: %v", err)
	}

	// Create LLM agent with load artifacts tool
	llmAgent, err := llmagent.New(llmagent.Config{
		Name:        "load-artifacts-agent",
		Description: "Agent that can load artifacts",
		Model:       "gemini-1.5-pro",
		Instruction: "You are an agent that can load and analyze artifacts.",
		Tools:       []tool.Tool{loadArtifactsTool},
	})
	if err != nil {
		log.Fatalf("Failed to create LLM agent: %v", err)
	}

	// Create a runner
	runnerCfg := runner.Config{
		AppName:   "load-artifacts-app",
		RootAgent: llmAgent,
	}

	// Launcher handles parsing command line arguments and running the app
	l := full.NewLauncher()
	if err := l.ParseAndRun(ctx, runnerCfg, os.Args[1:]); err != nil {
		log.Fatalf("Failed to run: %v\n\n%s", err, l.FormatSyntax())
	}
}
```

#### 7.1.3 运行说明

```bash
# 运行加载制品代理
go run ./examples/tools/loadartifacts/main.go console

# 查看帮助信息
go run ./examples/tools/loadartifacts/main.go help
```

#### 7.1.4 使用场景

- 学习如何创建和使用加载制品工具
- 了解制品服务的基本概念
- 构建需要处理制品的代理

### 7.2 多工具示例 (multipletools/)

#### 7.2.1 示例目的

多工具示例演示了如何使用 ADK-Go 框架创建一个代理，使用多个工具。

#### 7.2.2 示例代码

```go
package main

import (
	"context"
	"log"
	"os"

	"github.com/sjzsdu/adk-go/agent/llmagent"
	"github.com/sjzsdu/adk-go/full"
	"github.com/sjzsdu/adk-go/runner"
	"github.com/sjzsdu/adk-go/tool"
	"github.com/sjzsdu/adk-go/tool/builtintools"
)

func main() {
	ctx := context.Background()

	// Create multiple tools
	calculatorTool := builtintools.NewCalculator()
	httpTool := builtintools.NewHTTP()
	fileSystemTool := builtintools.NewFileSystem(".")

	// Create LLM agent with multiple tools
	llmAgent, err := llmagent.New(llmagent.Config{
		Name:        "multiple-tools-agent",
		Description: "Agent with multiple tools",
		Model:       "gemini-1.5-pro",
		Instruction: "You are an agent with multiple tools. Use the appropriate tool for each task.",
		Tools:       []tool.Tool{calculatorTool, httpTool, fileSystemTool},
	})
	if err != nil {
		log.Fatalf("Failed to create LLM agent: %v", err)
	}

	// Create a runner
	runnerCfg := runner.Config{
		AppName:   "multiple-tools-app",
		RootAgent: llmAgent,
	}

	// Launcher handles parsing command line arguments and running the app
	l := full.NewLauncher()
	if err := l.ParseAndRun(ctx, runnerCfg, os.Args[1:]); err != nil {
		log.Fatalf("Failed to run: %v\n\n%s", err, l.FormatSyntax())
	}
}
```

#### 7.2.3 运行说明

```bash
# 运行多工具代理
go run ./examples/tools/multipletools/main.go console

# 查看帮助信息
go run ./examples/tools/multipletools/main.go help
```

#### 7.2.4 使用场景

- 学习如何创建和使用多个工具
- 了解工具系统的基本概念
- 构建需要多种工具的代理

## 8. Vertex AI 示例 (vertexai/)

### 8.1 图像生成器示例 (imagegenerator/)

#### 8.1.1 示例目的

图像生成器示例演示了如何使用 ADK-Go 框架创建一个代理，使用 Vertex AI 进行图像生成。

#### 8.1.2 示例代码

```go
package main

import (
	"context"
	"log"
	"os"

	"github.com/sjzsdu/adk-go/agent/llmagent"
	"github.com/sjzsdu/adk-go/full"
	"github.com/sjzsdu/adk-go/runner"
	"github.com/sjzsdu/adk-go/tool"
	"github.com/sjzsdu/adk-go/tool/builtintools"
)

func main() {
	ctx := context.Background()

	// Create Vertex AI image generator tool
	imageGeneratorTool, err := builtintools.NewVertexAIImageGenerator()
	if err != nil {
		log.Fatalf("Failed to create Vertex AI image generator tool: %v", err)
	}

	// Create LLM agent with image generator tool
	llmAgent, err := llmagent.New(llmagent.Config{
		Name:        "image-generator-agent",
		Description: "Agent that can generate images using Vertex AI",
		Model:       "gemini-1.5-pro",
		Instruction: "You are an agent that can generate images using Vertex AI. Use the image generator tool to generate images based on user requests.",
		Tools:       []tool.Tool{imageGeneratorTool},
	})
	if err != nil {
		log.Fatalf("Failed to create LLM agent: %v", err)
	}

	// Create a runner
	runnerCfg := runner.Config{
		AppName:   "image-generator-app",
		RootAgent: llmAgent,
	}

	// Launcher handles parsing command line arguments and running the app
	l := full.NewLauncher()
	if err := l.ParseAndRun(ctx, runnerCfg, os.Args[1:]); err != nil {
		log.Fatalf("Failed to run: %v\n\n%s", err, l.FormatSyntax())
	}
}
```

#### 8.1.3 运行说明

```bash
# 运行图像生成器代理
go run ./examples/vertexai/imagegenerator/main.go console

# 查看帮助信息
go run ./examples/vertexai/imagegenerator/main.go help
```

#### 8.1.4 使用场景

- 学习如何创建和使用 Vertex AI 图像生成工具
- 了解如何集成 Vertex AI 服务
- 构建图像生成代理

## 9. MCP 示例 (mcp/)

### 9.1 示例目的

MCP 示例演示了如何使用 ADK-Go 框架创建一个 MCP（Model Configuration Protocol）代理。

### 9.2 示例代码

```go
package main

import (
	"context"
	"log"
	"os"

	"github.com/sjzsdu/adk-go/agent/llmagent"
	"github.com/sjzsdu/adk-go/full"
	"github.com/sjzsdu/adk-go/runner"
)

func main() {
	ctx := context.Background()

	// Create MCP agent
	mcpAgent, err := llmagent.New(llmagent.Config{
		Name:        "mcp-agent",
		Description: "MCP agent",
		Model:       "gemini-1.5-pro",
		Instruction: "You are an MCP agent. Respond to MCP requests.",
	})
	if err != nil {
		log.Fatalf("Failed to create MCP agent: %v", err)
	}

	// Create a runner
	runnerCfg := runner.Config{
		AppName:   "mcp-app",
		RootAgent: mcpAgent,
	}

	// Launcher handles parsing command line arguments and running the app
	l := full.NewLauncher()
	if err := l.ParseAndRun(ctx, runnerCfg, os.Args[1:]); err != nil {
		log.Fatalf("Failed to run: %v\n\n%s", err, l.FormatSyntax())
	}
}
```

### 9.3 运行说明

```bash
# 运行 MCP 代理
go run ./examples/mcp/main.go mcp

# 查看帮助信息
go run ./examples/mcp/main.go help
```

### 9.4 使用场景

- 学习如何创建 MCP 代理
- 了解 MCP 协议的基本概念
- 构建 MCP 兼容的代理

## 10. 示例分类和使用场景

| 示例类型 | 示例名称 | 使用场景 |
|----------|----------|----------|
| 基础示例 | quickstart | 了解 ADK-Go 框架的基本结构 |
| 服务器示例 | rest | 构建 REST API 服务 |
| 远程代理示例 | a2a | 构建分布式代理系统 |
| Web 示例 | web | 构建可视化代理服务 |
| 工作流代理示例 | sequential | 构建按顺序执行的工作流 |
| 工作流代理示例 | parallel | 构建并行执行的工作流 |
| 工作流代理示例 | loop | 构建迭代优化的工作流 |
| 工作流代理示例 | sequentialCode | 构建代码驱动的工作流 |
| 工具示例 | loadartifacts | 处理和分析制品 |
| 工具示例 | multipletools | 构建需要多种工具的代理 |
| Vertex AI 示例 | imagegenerator | 构建图像生成代理 |
| MCP 示例 | mcp | 构建 MCP 兼容的代理 |

## 11. 最佳实践

### 11.1 示例学习最佳实践

- 从简单示例开始，逐步学习复杂示例
- 理解每个示例的核心功能和设计思路
- 尝试修改示例，扩展其功能
- 结合文档学习，深入理解 ADK-Go 框架

### 11.2 示例开发最佳实践

- 遵循示例模板（EXAMPLE.md）创建新示例
- 保持示例简洁，专注于演示一个或几个功能
- 添加详细的注释，说明示例的功能和使用方法
- 提供清晰的运行说明

## 12. 总结

ADK-Go 示例库提供了丰富的示例应用，覆盖了框架的各种功能和使用场景。通过学习这些示例，您可以快速了解 ADK-Go 框架的基本结构和使用方法，为开发自己的代理应用打下基础。

建议您按照以下顺序学习示例：
1. quickstart：了解框架的基本结构
2. rest：学习 REST API 服务开发
3. workflowagents/sequential：学习工作流代理的基本概念
4. tools/multipletools：学习工具系统的使用
5. 根据您的需求选择其他示例

通过深入学习和实践这些示例，您将能够熟练使用 ADK-Go 框架开发各种类型的代理应用。
