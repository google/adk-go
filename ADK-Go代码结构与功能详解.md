# ADK-Go 代码结构与功能详解

## 项目概述

ADK-Go (Agent Development Kit for Go) 是 Google 开源的 Go 语言 AI 代理开发框架，用于构建、评估和部署复杂的 AI 代理系统。项目采用模块化设计，支持从简单任务到复杂系统的代理工作流编排。

## 项目基本信息

- **模块名**: `github.com/sjzsdu/adk-go`
- **Go 版本**: 1.24.4
- **许可证**: Apache 2.0
- **主要依赖**: Gemini API、Google Cloud Storage、Gorilla Mux、GORM 等

## 🗂️ 目录结构详解

### 🎯 核心包 (Core Packages)

#### `/agent` - 代理核心包
```
agent/
├── agent.go           # Agent 接口定义和基础实现
├── agent_test.go      # Agent 核心测试
├── context.go         # 调用上下文定义
├── run_config.go      # 运行配置
├── doc.go            # 包文档
├── llmagent/         # LLM 代理实现
├── remoteagent/      # 远程代理 (A2A)
└── workflowagents/   # 工作流代理
    ├── loopagent/       # 循环代理
    ├── parallelagent/   # 并行代理
    └── sequentialagent/ # 顺序代理
```

**功能说明：**
- **Agent 接口**: 定义所有代理的基本行为契约
- **LLM Agent**: 与语言模型交互的智能代理
- **Remote Agent**: 支持分布式部署的远程代理通信
- **Workflow Agents**: 提供不同执行模式的工作流编排

#### `/runner` - 运行器包
```
runner/
├── runner.go         # 核心运行器实现
└── runner_test.go    # 运行器测试
```

**功能说明：**
- 代理执行的核心调度器
- 管理代理生命周期
- 处理会话和上下文
- 提供流式事件处理

#### `/tool` - 工具系统包
```
tool/
├── tool.go              # Tool 接口定义
├── tool_test.go         # 工具测试
├── agenttool/           # 代理工具封装
├── exitlooptool/        # 循环退出工具
├── functiontool/        # 函数工具封装
├── geminitool/          # Gemini API 工具
├── loadartifactstool/   # 制品加载工具
└── mcptoolset/          # MCP 工具集
```

**功能说明：**
- **Function Tool**: 将普通 Go 函数封装为代理工具
- **Agent Tool**: 将其他代理作为工具使用
- **Gemini Tool**: 集成 Google Gemini API
- **MCP Tool**: 支持模型上下文协议 (Model Context Protocol)
- **Artifact Tool**: 管理和加载制品资源

### 💾 存储与服务包 (Storage & Services)

#### `/session` - 会话管理包
```
session/
├── session.go        # Session 接口定义
├── inmemory.go       # 内存会话实现
├── inmemory_test.go  # 内存实现测试
├── service.go        # 会话服务接口
├── vertexai.go       # Vertex AI 集成
├── doc.go           # 包文档
└── database/        # 数据库会话实现
```

**功能说明：**
- 会话状态持久化
- 支持内存和数据库存储
- 会话生命周期管理
- Vertex AI 平台集成

#### `/artifact` - 制品管理包
```
artifact/
├── service.go               # 制品服务接口
├── inmemory.go             # 内存制品存储
├── inmemory_test.go        # 内存存储测试
├── artifact_key_test.go    # 制品键测试
├── request_validation_test.go # 请求验证测试
└── gcs/                    # Google Cloud Storage 实现
    ├── gcs_client.go          # GCS 客户端
    ├── gcs_test.go           # GCS 测试
    └── service.go            # GCS 服务实现
```

**功能说明：**
- 代理产生的制品存储和管理
- 支持内存和 Google Cloud Storage
- 制品版本控制和访问控制
- 制品共享和分发

#### `/memory` - 内存管理包
```
memory/
├── service.go        # 内存服务接口
├── inmemory.go       # 内存实现
└── inmemory_test.go  # 内存测试
```

**功能说明：**
- 代理执行期间的内存管理
- 上下文信息存储
- 临时数据缓存

### 🤖 模型与集成包 (Model & Integration)

#### `/model` - 模型抽象包
```
model/
├── llm.go         # LLM 接口定义
├── llm_test.go    # LLM 测试
└── gemini/        # Gemini 模型实现
```

**功能说明：**
- LLM 模型抽象接口
- Google Gemini API 实现
- 支持扩展其他 LLM 提供商
- 模型配置和参数管理

#### `/server` - 服务器包
```
server/
├── doc.go         # 包文档
├── adka2a/        # A2A 服务器实现
└── restapi/       # REST API 服务器
```

**功能说明：**
- **REST API Server**: 提供 HTTP REST API 接口
- **A2A Server**: Agent-to-Agent 通信服务器
- 支持多种部署模式

### 🛠️ 命令行工具包 (CLI Tools)

#### `/cmd` - 命令行工具
```
cmd/
├── adkgo/            # ADK-Go 主命令
│   ├── adkgo.go         # 主程序入口
│   └── internal/        # 内部实现
└── launcher/         # 启动器框架
    ├── launcher.go      # 启动器主程序
    ├── console/         # 控制台启动器
    ├── full/           # 完整功能启动器
    ├── prod/           # 生产环境启动器
    ├── universal/      # 通用启动器
    └── web/           # Web 启动器
```

**功能说明：**
- **adkgo**: 主命令行工具，提供代理管理功能
- **Launcher**: 多模式启动器，支持控制台、Web、API 等运行模式
- 灵活的启动配置和参数解析

### 📚 示例与工具包 (Examples & Utilities)

#### `/examples` - 示例代码
```
examples/
├── README.md           # 示例说明文档
├── quickstart/         # 快速开始示例
│   └── main.go
├── a2a/               # Agent-to-Agent 示例
│   └── main.go
├── mcp/               # MCP 协议示例
│   └── main.go
├── tools/             # 工具使用示例
│   ├── loadartifacts/    # 制品加载示例
│   └── multipletools/    # 多工具使用示例
├── vertexai/          # Vertex AI 集成示例
│   ├── agent.go
│   └── imagegenerator/
├── web/               # Web 应用示例
│   ├── main.go
│   └── agents/
└── workflowagents/    # 工作流代理示例
    ├── loop/             # 循环代理示例
    ├── parallel/         # 并行代理示例
    ├── sequential/       # 顺序代理示例
    └── sequentialCode/   # 顺序代码示例
```

**功能说明：**
- **Quickstart**: 最小化的入门示例
- **A2A**: 分布式代理通信示例
- **MCP**: 模型上下文协议集成
- **Tools**: 各种工具的使用方法
- **Vertex AI**: Google Cloud 平台集成
- **Web**: Web 应用开发示例
- **Workflow**: 不同工作流模式的实际应用

#### `/util` - 实用工具包
```
util/
└── instructionutil/    # 指令处理工具
```

**功能说明：**
- 指令解析和处理
- 文本格式化和验证
- 通用工具函数

### 🔧 内部实现包 (Internal Packages)

#### `/internal` - 内部实现
```
internal/
├── style_test.go       # 代码风格测试
├── agent/             # 代理内部实现
│   ├── state.go          # 状态管理
│   ├── parentmap/        # 父子关系映射
│   └── runconfig/        # 运行配置
├── artifact/          # 制品内部实现
├── cli/               # CLI 内部工具
├── context/           # 上下文内部实现
├── converters/        # 数据转换器
├── httprr/           # HTTP 记录回放
├── llminternal/      # LLM 内部处理
├── memory/           # 内存内部实现
├── sessioninternal/  # 会话内部实现
├── sessionutils/     # 会话工具
├── telemetry/        # 遥测数据
├── testutil/         # 测试工具
├── toolinternal/     # 工具内部实现
├── typeutil/         # 类型工具
├── utils/           # 通用工具
└── version/         # 版本管理
```

**功能说明：**
- 框架内部实现细节
- 不对外暴露的工具和辅助函数
- 测试支持和开发工具
- 遥测和监控支持

#### `/telemetry` - 遥测包
```
telemetry/
└── telemetry.go       # 遥测数据收集
```

**功能说明：**
- 性能指标收集
- 使用情况统计
- 错误追踪和监控

## 🚀 核心功能特性

### 1. 多类型代理支持
- **LLM Agent**: 基于大语言模型的智能代理
- **Custom Agent**: 自定义业务逻辑代理
- **Remote Agent**: 分布式远程代理
- **Workflow Agent**: 工作流编排代理

### 2. 丰富的工具生态
- **Function Tool**: 函数封装工具
- **Agent Tool**: 代理嵌套工具
- **Gemini Tool**: Gemini API 集成
- **MCP Tool**: 模型上下文协议支持
- **Artifact Tool**: 制品管理工具

### 3. 灵活的存储后端
- **In-Memory**: 内存存储，适用于开发和测试
- **Database**: 数据库存储，支持持久化
- **Google Cloud Storage**: 云存储，适用于生产环境

### 4. 多种部署模式
- **Console**: 命令行交互模式
- **Web UI**: Web 界面模式
- **REST API**: HTTP API 服务模式
- **A2A**: Agent-to-Agent 分布式模式

### 5. 企业级特性
- **流式处理**: 实时响应和事件流
- **会话管理**: 状态持久化和恢复
- **错误处理**: 完善的错误处理和重试机制
- **遥测监控**: 性能监控和使用统计
- **安全性**: 访问控制和数据保护

## 🏗️ 架构设计原则

### 1. 接口导向设计
- 所有核心组件都定义了清晰的接口
- 支持可插拔的实现和扩展
- 便于测试和模拟

### 2. 分层架构
- 清晰的层次结构和职责分离
- 从应用层到存储层的完整抽象
- 易于维护和扩展

### 3. 并发安全
- 充分利用 Go 的并发特性
- 支持高并发代理执行
- 线程安全的状态管理

### 4. 云原生支持
- 容器化友好的设计
- 支持 Kubernetes 部署
- 与 Google Cloud 服务深度集成

## 📈 使用场景

### 1. 简单任务自动化
使用 LLM Agent 处理单一任务，如文档生成、数据分析等。

### 2. 复杂业务流程
使用工作流代理编排多个子任务，实现复杂的业务逻辑。

### 3. 分布式系统
使用 A2A 代理实现大规模分布式处理，提高系统可扩展性。

### 4. Web 应用集成
通过 Web 服务器和 REST API 为前端应用提供代理服务。

### 5. 企业级部署
结合云存储和数据库，构建企业级的 AI 代理平台。

## 🔄 开发流程

### 1. 环境配置
```bash
go get github.com/sjzsdu/adk-go
```

### 2. 创建代理
```go
agent := llmagent.New(llmagent.Config{
    Model: model,
    Instruction: "You are a helpful assistant",
    Tools: []tool.Tool{...},
})
```

### 3. 配置运行器
```go
runner, err := runner.New(runner.Config{
    AppName: "my-app",
    RootAgent: agent,
})
```

### 4. 执行代理
```go
for event := range runner.Run(ctx, userID, sessionID, message, nil) {
    // 处理事件
}
```

## 📋 总结

ADK-Go 是一个功能全面、设计优雅的 AI 代理开发框架。其模块化的架构设计、丰富的功能特性和灵活的部署选项，使其能够满足从原型开发到企业级部署的各种需求。框架充分利用了 Go 语言的优势，为开发者提供了高性能、高并发的 AI 代理开发平台。