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

// ExtractLauncherArgs 从命令行参数中提取launcher需要的参数，跳过模型相关参数
// 这个函数可以在调用launcher.Execute时使用，避免将模型参数传递给launcher导致参数冲突
func ExtractLauncherArgs(args []string) []string {
	var launcherArgs []string
	for i := 0; i < len(args); i++ {
		arg := args[i]
		// 跳过模型相关参数，同时支持单破折号和双破折号形式
		if arg == "-model" || arg == "--model" || arg == "-model-name" || arg == "--model-name" {
			// 如果参数有值，也跳过下一个参数
			if i+1 < len(args) && args[i+1][0] != '-' {
				i++
			}
			continue
		}
		launcherArgs = append(launcherArgs, arg)
	}
	return launcherArgs
}
