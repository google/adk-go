// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package console provides a simple way to interact with an agent from console application.
package console

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"

	"google.golang.org/genai"

	"github.com/sjzsdu/adk-go/agent"
	"github.com/sjzsdu/adk-go/cmd/launcher"
	"github.com/sjzsdu/adk-go/cmd/launcher/universal"
	"github.com/sjzsdu/adk-go/internal/cli/util"
	"github.com/sjzsdu/adk-go/runner"
	"github.com/sjzsdu/adk-go/session"
)

// consoleConfig contains command-line params for console launcher
type consoleConfig struct {
	streamingMode       agent.StreamingMode
	streamingModeString string // command-line param to be converted to agent.StreamingMode
}

// consoleLauncher allows to interact with an agent in console
type consoleLauncher struct {
	flags  *flag.FlagSet  // flags are used to parse command-line arguments
	config *consoleConfig // config contains parsed command-line parameters
}

// NewLauncher creates new console launcher
func NewLauncher() launcher.SubLauncher {
	config := &consoleConfig{}

	fs := flag.NewFlagSet("console", flag.ContinueOnError)
	fs.StringVar(&config.streamingModeString, "streaming_mode", string(agent.StreamingModeSSE),
		fmt.Sprintf("defines streaming mode (%s|%s)", agent.StreamingModeNone, agent.StreamingModeSSE))

	return &consoleLauncher{config: config, flags: fs}
}

// Run implements launcher.SubLauncher. It starts the console interaction loop.
func (l *consoleLauncher) Run(ctx context.Context, config *launcher.Config) error {
	// userID and appName are not important at this moment, we can just use any
	userID, appName := "console_user", "console_app"

	sessionService := config.SessionService
	if sessionService == nil {
		sessionService = session.InMemoryService()
	}

	resp, err := sessionService.Create(ctx, &session.CreateRequest{
		AppName: appName,
		UserID:  userID,
	})
	if err != nil {
		return fmt.Errorf("failed to create the session service: %v", err)
	}

	rootAgent := config.AgentLoader.RootAgent()

	session := resp.Session

	r, err := runner.New(runner.Config{
		AppName:         appName,
		Agent:           rootAgent,
		SessionService:  sessionService,
		ArtifactService: config.ArtifactService,
	})
	if err != nil {
		return fmt.Errorf("failed to create runner: %v", err)
	}

	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Print("\nUser -> ")

		userInput, err := reader.ReadString('\n')
		if err != nil {
			log.Fatal(err)
		}

		// 处理特殊命令
		trimmedInput := strings.TrimSpace(userInput)
		switch strings.ToLower(trimmedInput) {
		case "q", "quit", "exit":
			fmt.Println("\n正在退出...")
			return nil
		case "vim":
			// 创建临时文件
			tempFile, err := os.CreateTemp("", "adk-vim-input-")
			if err != nil {
				fmt.Printf("创建临时文件失败: %v\n", err)
				continue
			}
			tempFilePath := tempFile.Name()
			tempFile.Close()
			defer os.Remove(tempFilePath) // 确保程序结束时删除临时文件

			// 启动vim编辑器
			fmt.Println("正在启动vim编辑器，请输入您的文本内容...")
			cmd := exec.Command("vim", tempFilePath)
			cmd.Stdin = os.Stdin
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			err = cmd.Run()
			if err != nil {
				fmt.Printf("vim编辑失败: %v\n", err)
				continue
			}

			// 读取vim编辑的内容
			vimContent, err := os.ReadFile(tempFilePath)
			if err != nil {
				fmt.Printf("读取vim编辑内容失败: %v\n", err)
				continue
			}

			userInput = string(vimContent)
			fmt.Println("已成功读取vim编辑的内容")
		}

		// 如果输入为空（只包含空白字符），则保持用户输入状态
		if strings.TrimSpace(userInput) == "" {
			continue
		}

		userMsg := genai.NewContentFromText(userInput, genai.RoleUser)

		streamingMode := l.config.streamingMode
		if streamingMode == "" {
			streamingMode = agent.StreamingModeSSE
		}
		fmt.Print("\nAgent -> ")
		prevText := ""
		for event, err := range r.Run(ctx, userID, session.ID(), userMsg, agent.RunConfig{
			StreamingMode: streamingMode,
		}) {
			if err != nil {
				fmt.Printf("\nAGENT_ERROR: %v\n", err)
			} else {
				if event.LLMResponse.Content == nil {
					continue
				}

				text := ""
				for _, p := range event.LLMResponse.Content.Parts {
					text += p.Text
				}

				if streamingMode != agent.StreamingModeSSE {
					fmt.Print(text)
					continue
				}

				// In SSE mode, always print partial responses and capture them.
				if !event.IsFinalResponse() {
					fmt.Print(text)
					prevText += text
					continue
				}

				// Only print final response if it doesn't match previously captured text.
				if text != prevText {
					fmt.Print(text)
				}

				prevText = ""
			}
		}
	}
}

// Parse implements launcher.SubLauncher. After parsing console-specific
// arguments returns remaining un-parsed arguments
func (l *consoleLauncher) Parse(args []string) ([]string, error) {
	err := l.flags.Parse(args)
	if err != nil || !l.flags.Parsed() {
		return nil, fmt.Errorf("failed to parse flags: %v", err)
	}
	if l.config.streamingModeString != string(agent.StreamingModeNone) &&
		l.config.streamingModeString != string(agent.StreamingModeSSE) {
		return nil, fmt.Errorf("invalid streaming_mode: %v. Should be (%s|%s)", l.config.streamingModeString,
			agent.StreamingModeNone, agent.StreamingModeSSE)
	}
	l.config.streamingMode = agent.StreamingMode(l.config.streamingModeString)
	return l.flags.Args(), nil
}

// Keyword implements launcher.SubLauncher. Returns the command-line keyword for this launcher.
func (l *consoleLauncher) Keyword() string {
	return "console"
}

// CommandLineSyntax implements launcher.SubLauncher. Returns the command-line syntax for the console launcher.
func (l *consoleLauncher) CommandLineSyntax() string {
	return util.FormatFlagUsage(l.flags)
}

// SimpleDescription implements launcher.SubLauncher. Returns a simple description of the console launcher.
func (l *consoleLauncher) SimpleDescription() string {
	return "runs an agent in console mode."
}

// Execute implements launcher.Launcher. It parses arguments and runs the launcher.
func (l *consoleLauncher) Execute(ctx context.Context, config *launcher.Config, args []string) error {
	remainingArgs, err := l.Parse(args)
	if err != nil {
		return fmt.Errorf("cannot parse args: %w", err)
	}
	// do not accept additional arguments
	err = universal.ErrorOnUnparsedArgs(remainingArgs)
	if err != nil {
		return fmt.Errorf("cannot parse all the arguments: %w", err)
	}
	return l.Run(ctx, config)
}
