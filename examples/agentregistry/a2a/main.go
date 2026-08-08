// Copyright 2026 Google LLC
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

// Package main closes the Agent Registry loop for A2A: it serves an agent over
// A2A, resolves that same agent back out of the registry, and exchanges a
// message with it.
//
// One process plays both roles so the loop runs in one terminal. In production
// they are separate deployments, and only the publishing half moves — the
// consuming half below is what any caller writes.
package main

import (
	"cmp"
	"context"
	"fmt"
	"iter"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agentregistry"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	adka2a "google.golang.org/adk/v2/server/adka2a/v2"
	"google.golang.org/adk/v2/session"
)

func main() {
	ctx := context.Background()

	project := os.Getenv("GOOGLE_CLOUD_PROJECT")
	if project == "" {
		log.Fatal("GOOGLE_CLOUD_PROJECT must be set")
	}
	location := cmp.Or(os.Getenv("GOOGLE_CLOUD_LOCATION"), "global")
	name := os.Getenv("REGISTRY_AGENT")
	if name == "" {
		log.Fatal("REGISTRY_AGENT must be set to the agent resource name from registration; see the README")
	}
	// Must match the URL in the card that was registered.
	addr := cmp.Or(os.Getenv("A2A_ADDR"), "localhost:8765")

	// Publisher half.
	srv, err := serve(addr)
	if err != nil {
		log.Fatalf("Failed to start the A2A server: %v", err)
	}
	defer func() { _ = srv.Close() }()
	log.Printf("Serving an agent over A2A at http://%s", addr)

	// Consumer half: everything below only knows the registry resource name.
	client, err := agentregistry.New(ctx, agentregistry.Config{
		ProjectID: project,
		Location:  location,
	})
	if err != nil {
		log.Fatalf("Failed to create the registry client: %v", err)
	}

	// No egress options: the registered card points at localhost, which is not
	// behind IAM. An agent that is needs WithA2AHTTPClient.
	remote, err := client.RemoteAgent(ctx, name)
	if err != nil {
		log.Fatalf("Failed to resolve %q: %v", name, err)
	}
	log.Printf("Resolved %q from the registry; the URL and transport came from its card", remote.Name())

	reply, err := ask(ctx, remote, "Hello from the registry!")
	if err != nil {
		log.Fatalf("Failed to reach the agent over A2A: %v", err)
	}
	fmt.Printf("\n<<< %s\n", reply)
}

// ask runs one turn against the agent and returns its text.
func ask(ctx context.Context, a agent.Agent, prompt string) (string, error) {
	r, err := runner.New(runner.Config{
		AppName:           a.Name(),
		Agent:             a,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		return "", err
	}
	fmt.Printf("\n>>> %s\n", prompt)

	var reply strings.Builder
	for event, err := range r.Run(ctx, "user", "session", genai.NewContentFromText(prompt, genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			return "", err
		}
		// A failed A2A call arrives as an event carrying an error and no
		// content, which would otherwise read as a silent empty answer.
		if event.LLMResponse.ErrorMessage != "" {
			return "", fmt.Errorf("remote agent: %s", event.LLMResponse.ErrorMessage)
		}
		if event.LLMResponse.Content != nil {
			for _, part := range event.LLMResponse.Content.Parts {
				reply.WriteString(part.Text)
			}
		}
	}
	if reply.Len() == 0 {
		return "", fmt.Errorf("empty reply")
	}
	return reply.String(), nil
}

// serve exposes a canned agent over A2A. It echoes the request so a successful
// run proves both directions of the exchange, and needs no model credentials.
func serve(addr string) (*http.Server, error) {
	echo, err := agent.New(agent.Config{
		Name: "registry_echo",
		Run: func(ictx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				var got strings.Builder
				if c := ictx.UserContent(); c != nil {
					for _, part := range c.Parts {
						got.WriteString(part.Text)
					}
				}
				yield(&session.Event{LLMResponse: model.LLMResponse{
					Content: genai.NewContentFromText("echo: "+got.String(), genai.RoleModel),
				}}, nil)
			}
		},
	})
	if err != nil {
		return nil, err
	}

	handler := a2asrv.NewHandler(adka2a.NewExecutor(adka2a.ExecutorConfig{
		RunnerConfig: runner.Config{
			AppName:        echo.Name(),
			Agent:          echo,
			SessionService: session.InMemoryService(),
		},
	}))
	// The card registered for this agent declares HTTP+JSON, so serve REST.
	srv := &http.Server{Addr: addr, Handler: a2asrv.NewRESTHandler(handler)}

	failed := make(chan error, 1)
	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			failed <- err
		}
	}()
	select {
	case err := <-failed:
		return nil, err
	case <-time.After(300 * time.Millisecond):
		return srv, nil
	}
}
