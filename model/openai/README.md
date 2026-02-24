# OpenAI Model Implementation for ADK-Go

这是一个基于 OpenAI API 的 ADK-Go 模型实现，直接调用 OpenAI 的 HTTP API，不依赖第三方库。

## 特性

- ✅ 直接使用 OpenAI HTTP API，无额外依赖
- ✅ 支持同步和流式响应
- ✅ 支持工具调用 (Function Calling)
- ✅ 完全兼容 ADK-Go 的 `model.LLM` 接口
- ✅ 与现有 `gemini.go` 实现风格保持一致
- ✅ 支持自定义 HTTP 客户端和配置

## 使用示例

### 基本用法

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/sjzsdu/adk-go/model/openai"
    "github.com/sjzsdu/adk-go/model"
    "google.golang.org/genai"
)

func main() {
    ctx := context.Background()

    // 创建 OpenAI 模型
    config := openai.Config{
        APIKey: "your-api-key", // 或通过环境变量 OPENAI_API_KEY 设置
    }
    
    model, err := openai.NewModel(ctx, "gpt-4", config)
    if err != nil {
        log.Fatal(err)
    }

    // 创建请求
    req := &model.LLMRequest{
        Contents: []*genai.Content{
            genai.NewContentFromText("你好，请介绍一下自己", "user"),
        },
    }

    // 生成内容
    for response, err := range model.GenerateContent(ctx, req, false) {
        if err != nil {
            log.Fatal(err)
        }
        
        for _, part := range response.Content.Parts {
            if part.Text != "" {
                fmt.Println(part.Text)
            }
        }
        break
    }
}
```

### 流式响应

```go
// 启用流式响应
for response, err := range model.GenerateContent(ctx, req, true) {
    if err != nil {
        log.Fatal(err)
    }
    
    if response != nil && response.Content != nil {
        for _, part := range response.Content.Parts {
            if part.Text != "" {
                fmt.Print(part.Text) // 实时输出
            }
        }
    }
}
```

### 高级配置

```go
config := openai.Config{
    APIKey:       "your-api-key",
    BaseURL:      "https://api.openai.com/v1", // 自定义端点
    Organization: "your-org-id",               // 组织 ID
    HTTPClient:   &http.Client{               // 自定义 HTTP 客户端
        Timeout: 30 * time.Second,
    },
}
```

### 使用生成配置

```go
req := &model.LLMRequest{
    Contents: []*genai.Content{
        genai.NewContentFromText("写一首诗", "user"),
    },
    Config: &genai.GenerateContentConfig{
        MaxOutputTokens: 1000,
        Temperature:     genai.Ptr(0.7),
        TopP:           genai.Ptr(0.9),
    },
}
```

## 配置选项

### Config 结构体

- **APIKey**: OpenAI API 密钥，如果为空会从 `OPENAI_API_KEY` 环境变量读取
- **BaseURL**: API 端点，默认为 `https://api.openai.com/v1`
- **Organization**: OpenAI 组织 ID（可选）
- **HTTPClient**: 自定义 HTTP 客户端（可选）

### 支持的模型

- `gpt-4`
- `gpt-4-turbo`
- `gpt-3.5-turbo`
- 以及其他 OpenAI 支持的模型

### 生成参数

通过 `genai.GenerateContentConfig` 支持：

- **MaxOutputTokens**: 最大输出令牌数
- **Temperature**: 控制随机性 (0.0-2.0)
- **TopP**: 核采样参数 (0.0-1.0)
- **Tools**: 工具/函数定义

## 实现特点

### 1. 直接 HTTP API 调用
- 不依赖第三方 SDK，减少依赖
- 完全控制 HTTP 请求和响应处理
- 易于调试和自定义

### 2. 流式处理支持
- 支持 Server-Sent Events (SSE) 流式响应
- 集成 ADK-Go 的流式响应聚合器
- 实时返回部分结果

### 3. 工具调用支持
- 自动转换 ADK-Go 工具定义到 OpenAI 格式
- 支持函数调用和响应处理
- 完整的工具执行流程

### 4. 错误处理
- 完整的 HTTP 状态码检查
- JSON 解析错误处理
- 详细的错误信息返回

### 5. 与 Gemini 实现一致
- 相同的接口风格和方法签名
- 统一的错误处理模式
- 相似的配置方式

## 架构设计

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   ADK-Go Core   │───▶│  OpenAI Model    │───▶│   OpenAI API    │
│                 │    │   Implementation │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │  HTTP Client     │
                       │  Request/Response│
                       │  Processing      │
                       └──────────────────┘
```

## 测试

运行测试前，请确保设置了 `OPENAI_API_KEY` 环境变量：

```bash
export OPENAI_API_KEY="your-api-key"
go test ./model/openai
```

如果没有 API 密钥，测试会被跳过。

## 与 LangChain 方案对比

| 特性 | 直接 API 实现 (推荐) | LangChain 实现 |
|------|---------------------|---------------|
| 依赖复杂度 | 低 (仅标准库 + HTTP) | 高 (第三方库) |
| 代码可读性 | 高 (直观明了) | 中 (多层抽象) |
| 调试难度 | 低 (直接控制) | 高 (抽象层多) |
| 自定义能力 | 强 (完全控制) | 受限 (依赖接口) |
| 维护成本 | 低 (自主控制) | 高 (依赖更新) |
| 性能 | 更好 (直接调用) | 一般 (多层转换) |
| 风格一致性 | 高 (与 gemini.go 一致) | 低 (不同风格) |

## 总结

这个 OpenAI 实现采用直接调用 HTTP API 的方式，提供了：

- 🚀 高性能：直接 HTTP 调用，无额外转换开销
- 🎯 高可控性：完全控制请求和响应处理
- 🔧 易维护：代码简洁，逻辑清晰
- 🎨 风格统一：与现有 Gemini 实现保持一致
- 📦 轻依赖：最小化外部依赖

这种实现方式更符合 ADK-Go 的设计理念，为开发者提供了清爽、高效的 OpenAI 模型集成方案。