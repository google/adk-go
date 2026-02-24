# ADK-Go Web 示例解读

## 项目概述

本示例展示了如何使用ADK-Go框架构建一个包含多个智能代理的Web应用程序。该应用程序集成了天气时间查询、LLM答案审核和图像生成等功能，通过A2A协议提供服务。

## 项目结构

```
examples/web/
├── main.go              # 主程序入口
├── web.md               # 本说明文档
└── agents/              # 代理实现目录
    ├── image_generator.go  # 图像生成代理
    └── llmauditor.go       # LLM答案审核代理
```

## 核心功能组件

### 1. 主程序 (main.go)

主程序负责初始化各个组件并启动服务：

- **模型初始化**：通过命令行参数创建LLM模型配置
- **代理创建**：初始化三个主要代理
- **会话和工件服务**：使用内存存储服务管理会话和生成的内容
- **身份认证**：实现简单的身份认证拦截器
- **服务启动**：使用full launcher启动包含A2A和Web UI的完整服务

### 2. 代理设计

本示例包含三个主要代理：

#### 2.1 天气时间代理 (weather_time_agent)

- **功能**：回答关于城市天气和时间的问题
- **工具集成**：集成Google搜索工具以获取实时信息
- **回调处理**：使用saveReportfunc保存LLM生成的响应到工件服务

#### 2.2 LLM审核代理 (llm_auditor)

- **功能**：评估LLM生成答案的准确性和可靠性
- **架构**：使用sequentialagent编排两个子代理
  - **critic_agent**：扮演专业调查记者角色，识别和验证答案中的每个声明
  - **reviser_agent**：扮演编辑角色，根据审核结果修订答案
- **工作流程**：
  1. 识别答案中的所有声明(claims)
  2. 验证每个声明的准确性
  3. 提供整体评估
  4. 根据评估结果修订答案
- **元数据处理**：收集和格式化引用信息，增强答案的可信度

#### 2.3 图像生成代理 (image_generator)

- **功能**：根据用户提示生成图像并保存
- **工具集成**：
  - **generate_image**：调用Google Imagen模型生成图像
  - **loadartifactstool**：加载保存的图像
- **技术实现**：使用Google Cloud Vertex AI的Imagen-3.0模型
- **工件管理**：将生成的图像保存到工件服务中

## 关键技术点

### 1. 工具集成机制

- 使用`functiontool`包将Go函数包装为代理可调用的工具
- 通过`tool.Context`提供对工件服务等资源的访问

### 2. 代理编排

- 使用`sequentialagent`实现多代理的顺序执行
- 通过`AfterModelCallback`处理LLM响应的后处理逻辑

### 3. 工件管理

- 使用`artifact.InMemoryService()`存储生成的内容
- 通过UUID生成唯一标识符管理工件

### 4. 身份认证

- 实现`AuthInterceptor`接口处理A2A请求的身份验证
- 设置固定用户名以简化演示

## 工作流程

1. 用户通过Web界面或A2A客户端发送请求
2. 请求被路由到相应的代理进行处理
3. 代理根据需要使用工具获取信息或执行操作
4. 生成的响应通过回调函数进行处理和增强
5. 结果返回给用户并可选地保存到工件服务

## 扩展与优化

1. **生产环境配置**：
   - 替换内存存储为持久化存储服务
   - 添加更复杂的身份认证机制
   - 实现错误处理和重试逻辑

2. **功能扩展**：
   - 添加更多专业领域的代理
   - 实现代理间的智能路由
   - 增强图像生成的参数控制

3. **性能优化**：
   - 添加请求缓存机制
   - 实现异步处理长时间运行的任务
   - 优化LLM提示以提高响应质量

## 运行示例

要运行此示例，需要设置以下环境变量：

```bash
# Google Cloud配置（用于图像生成）
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_LOCATION=your-region

# LLM模型配置（通过命令行参数或环境变量）
export MODEL_TYPE=your-model-type
export MODEL_API_KEY=your-api-key

# 启动应用
cd /path/to/adk-go/examples/web
go run main.go
```

启动后，应用将提供Web界面和A2A接口，可以通过浏览器或A2A客户端与之交互。