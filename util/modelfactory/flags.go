package modelfactory

import (
	"flag"
)

// 定义包级别的标志变量
var (
	// modelTypeFlag 存储命令行中的模型类型
	modelTypeFlag = flag.String("model", "gemini", "Model type to use: gemini, kimi, qwen, siliconflow, zhipu, deepseek")
	// modelNameFlag 存储命令行中的模型名称
	modelNameFlag = flag.String("model-name", "", "Specific model name to use (optional)")
)

// init 在包初始化时自动注册标志
func init() {
	// 标志已经通过包级变量定义
}

// RegisterFlags 注册模型相关的命令行标志
// 这是一个公开函数，允许调用者显式注册这些标志
func RegisterFlags() {
	// 标志已经通过包级变量注册，此函数提供一个明确的接口
}

// NewFromFlags 从命令行标志创建一个新的Config
// 如果调用者希望从命令行参数创建配置，应该先调用flag.Parse()
func NewFromFlags() *Config {
	return &Config{
		ModelType: *modelTypeFlag,
		ModelName: *modelNameFlag,
	}
}

// ParseAndCreateConfig 解析命令行参数并创建配置
// 这是一个便捷函数，它会解析命令行参数并返回对应的配置
func ParseAndCreateConfig() *Config {
	flag.Parse()
	return NewFromFlags()
}