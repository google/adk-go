# Agent Development Kit (ADK) Go 语言版

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Go Doc](https://img.shields.io/badge/Go%20Package-Doc-blue.svg)](https://pkg.go.dev/google.golang.org/adk/v2)
[![Nightly Check](https://github.com/google/adk-go/actions/workflows/nightly.yml/badge.svg)](https://github.com/google/adk-go/actions/workflows/nightly.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![View Code Wiki](https://www.gstatic.com/_/boq-sdlc-agents-ui/_/r/YUi5dj2UWvE.svg)](https://codewiki.google/github.com/google/adk-go)

<p align="center">
  <a href="https://github.com/google/adk-go/blob/main/README.md">English</a> · <b>简体中文</b>
</p>

<html>
    <h2 align="center">
      <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256"/>
    </h2>
    <h3 align="center">
      开源、代码优先的 Go 语言智能体开发工具包，助力灵活、可控地构建、评测与部署复杂 AI 智能体。
    </h3>
    <h3 align="center">
      重要链接：
      <a href="https://google.github.io/adk-docs/">官方文档</a> &
      <a href="https://github.com/google/adk-go/tree/main/examples">示例代码</a> &
      <a href="https://github.com/google/adk-python">Python 版 ADK</a> &
      <a href="https://github.com/google/adk-java">Java 版 ADK</a> &
      <a href="https://github.com/google/adk-kotlin">Kotlin 版 ADK</a> &
      <a href="https://github.com/google/adk-js">TypeScript 版 ADK</a> &
      <a href="https://github.com/google/adk-web">ADK Web</a>。
    </h3>
</html>

智能体开发套件（Agent Development Kit，简称 ADK）是一个灵活且模块化的框架，将经典软件工程开发原则引入 AI 智能体（Agent）的创建过程中。它旨在简化智能体工作流的构建、部署与编排，涵盖从单项简单任务到复杂多智能体协同系统的全场景应用。ADK 在针对 Gemini 深度优化的同时，具备模型无关（Model-agnostic）与部署无关（Deployment-agnostic）特性，并与现有各类框架保持良好兼容。

ADK 的 Go 语言实现充分发挥了 Go 语言在并发性与运行性能上的优势，非常适合开发者构建云原生（Cloud-native）智能体应用。

---

## ✨ 核心特性

*   **符合 Go 语言习惯（Idiomatic Go）：** 契合 Go 语言编程哲学，充分释放 Go 语言的并发与性能潜能。
*   **丰富的工具生态（Rich Tool Ecosystem）：** 支持利用预置工具、自定义函数或集成现有生态工具，为智能体赋予多样化的执行能力。
*   **代码优先开发（Code-First Development）：** 直接在 Go 代码中定义智能体业务逻辑、调用工具及工作流编排，具备极高的灵活性、可测试性与版本管理能力。
*   **模块化多智能体系统（Modular Multi-Agent Systems）：** 通过组合多个专业分工的智能体，轻松设计高可扩展性的复杂应用。
*   **全场景随时部署（Deploy Anywhere）：** 易于容器化打包并部署，针对 Google Cloud Run 等云原生运行环境提供开箱即用的深度支持。

## 🚀 安装指南

ADK Go 要求 Go 1.26.10 或更高版本（详见 [go.mod](go.mod)）。

若要在项目中添加并使用 ADK Go，请运行：

```bash
go get google.golang.org/adk/v2
```

## 📖 面向 AI 编码智能体的文档索引

[adk.dev/llms.txt](https://adk.dev/llms.txt) 是 ADK 文档的机器可读索引文件，而 [adk.dev/llms-full.txt](https://adk.dev/llms-full.txt) 则将全部文档聚合为单文件。两者均由 [adk-docs](https://github.com/google/adk-docs/) 自动生成，并收录了 Go API 参考规范与丰富示例。你可以将其中任意一个链接提供给 AI 编码智能体作为上下文：

```text
规划并实现一个能够审查客户工单问题并生成总结报告的智能体。使用 ADK Go 框架，并参考 https://adk.dev/llms-full.txt 获取示例代码。
```

如需对本项目源码本身进行贡献开发（而非使用它进行应用构建），请参阅 [AGENTS.md](AGENTS.md)。

## 📄 开源许可证

本项目基于 Apache 2.0 许可证开源 —— 详见 [LICENSE](LICENSE) 文件。

`internal/httprr` 模块除外 —— 请参阅其专属 [LICENSE 文件](internal/httprr/LICENSE)。

---

> _注：本中文文档由社区志愿者翻译与维护，旨在降低中文开发者的阅读门槛。如与英文原版存在差异或更新延迟，请以英文原版 [README.md](https://github.com/google/adk-go/blob/main/README.md) 为唯一权威标准（Single Source of Truth）。_
