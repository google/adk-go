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

package main

import (
	"context"
	"flag"
	"log"
	"os"

	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/google/uuid"

	"github.com/sjzsdu/adk-go/agent"
	"github.com/sjzsdu/adk-go/agent/llmagent"
	"github.com/sjzsdu/adk-go/artifact"
	"github.com/sjzsdu/adk-go/cmd/launcher"
	"github.com/sjzsdu/adk-go/cmd/launcher/full"
	"github.com/sjzsdu/adk-go/examples/web/agents"
	"github.com/sjzsdu/adk-go/model"
	"github.com/sjzsdu/adk-go/session"
	"github.com/sjzsdu/adk-go/tool"
	"github.com/sjzsdu/adk-go/tool/geminitool"
	"github.com/sjzsdu/adk-go/util/modelfactory"
)

func saveReportfunc(ctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error) {
	if llmResponse == nil || llmResponse.Content == nil || llmResponseError != nil {
		return llmResponse, llmResponseError
	}
	for _, part := range llmResponse.Content.Parts {
		_, err := ctx.Artifacts().Save(ctx, uuid.NewString(), part)
		if err != nil {
			return nil, err
		}
	}
	return llmResponse, llmResponseError
}

// AuthInterceptor sets 'user' name needed for both a2a and webui launchers which sharing the same sessions service.
type AuthInterceptor struct {
	a2asrv.PassthroughCallInterceptor
}

// Before implements a before request callback.
func (a *AuthInterceptor) Before(ctx context.Context, callCtx *a2asrv.CallContext, req *a2asrv.Request) (context.Context, error) {
	callCtx.User = &a2asrv.AuthenticatedUser{
		UserName: "user",
	}
	return ctx, nil
}

func main() {
	ctx := context.Background()
	flag.Parse()

	// 从命令行参数创建模型配置
	modelConfig := modelfactory.NewFromFlags()
	model := modelfactory.MustCreateModel(ctx, modelConfig)

	sessionService := session.InMemoryService()
	rootAgent, err := llmagent.New(llmagent.Config{
		Name:        "weather_time_agent",
		Model:       model,
		Description: "Agent to answer questions about the time and weather in a city.",
		Instruction: "I can answer your questions about the time and weather in a city.",
		Tools: []tool.Tool{
			geminitool.GoogleSearch{},
		},
		AfterModelCallbacks: []llmagent.AfterModelCallback{saveReportfunc},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}
	llmAuditor := agents.GetLLMAuditorAgent(ctx, model)
	imageGeneratorAgent := agents.GetImageGeneratorAgent(ctx, model)

	agentLoader, err := agent.NewMultiLoader(
		rootAgent,
		llmAuditor,
		imageGeneratorAgent,
	)
	if err != nil {
		log.Fatalf("Failed to create agent loader: %v", err)
	}

	artifactservice := artifact.InMemoryService()

	config := &launcher.Config{
		ArtifactService: artifactservice,
		SessionService:  sessionService,
		AgentLoader:     agentLoader,
		A2AOptions: []a2asrv.RequestHandlerOption{
			a2asrv.WithCallInterceptor(&AuthInterceptor{}),
		},
	}

	l := full.NewLauncher()
	// 过滤掉-model和-model-name参数，避免与launcher参数冲突
	launcherArgs := modelfactory.ExtractLauncherArgs(os.Args[1:])
	if err = l.Execute(ctx, config, launcherArgs); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
