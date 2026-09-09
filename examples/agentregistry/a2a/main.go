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
// One process plays both roles so the loop runs in one terminal: it publishes
// the agent, then consumes it through the registry. In production those are
// separate programs, and the consuming code below is unchanged — it is what any
// caller writes.
package main

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"log"
	"net"
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
	// Output here is a transcript, not a log: the timestamp prefix is noise.
	log.SetFlags(0)
	// run owns the listener; log.Fatal inside it would skip the deferred close.
	if err := run(context.Background()); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context) error {
	project := os.Getenv("GOOGLE_CLOUD_PROJECT")
	if project == "" {
		return errors.New("GOOGLE_CLOUD_PROJECT must be set")
	}
	location := cmp.Or(os.Getenv("GOOGLE_CLOUD_LOCATION"), "global")
	name := os.Getenv("REGISTRY_AGENT")
	if name == "" {
		return errors.New("REGISTRY_AGENT must be set to the agent resource name from registration; see the README")
	}
	// Must match the URL in the card that was registered.
	addr := cmp.Or(os.Getenv("A2A_ADDR"), "localhost:8765")

	// Publishing: in production this is the agent owner's deployment.
	srv, err := serve(addr)
	if err != nil {
		return fmt.Errorf("failed to start the A2A server: %w", err)
	}
	defer func() { _ = srv.Close() }()
	log.Printf("Serving an agent over A2A at http://%s", addr)

	// Consuming: everything below knows only the registry resource name.
	client, err := agentregistry.New(ctx, agentregistry.Config{
		ProjectID: project,
		Location:  location,
	})
	if err != nil {
		return fmt.Errorf("failed to create the registry client: %w", err)
	}

	// No egress options: the registered card points at localhost, which is not
	// behind IAM. An agent that is needs WithA2AHTTPClient.
	remote, err := client.RemoteAgent(ctx, name)
	if err != nil {
		return fmt.Errorf("failed to resolve %q: %s", name, explain(err))
	}
	log.Printf("Resolved %q from the registry; the URL and transport came from its card", remote.Name())

	reply, err := ask(ctx, remote, "Hello from the registry!")
	if err != nil {
		return fmt.Errorf("failed to reach the agent over A2A: %w", err)
	}
	fmt.Printf("\n<<< %s\n", reply)
	return nil
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
		// A failed A2A call arrives as an event carrying an error, which would
		// otherwise read as the agent simply having nothing to say.
		if event.LLMResponse.ErrorMessage != "" {
			return "", fmt.Errorf("remote agent: %s", event.LLMResponse.ErrorMessage)
		}
		// A streaming agent sends each chunk as a partial event and then the
		// whole text again in one non-partial event; progress notes arrive as
		// partials too. Summing both would return the answer twice.
		if event.LLMResponse.Partial {
			continue
		}
		if event.LLMResponse.Content != nil {
			for _, part := range event.LLMResponse.Content.Parts {
				reply.WriteString(part.Text)
			}
		}
	}
	if reply.Len() == 0 {
		return "", errors.New("empty reply")
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
	srv := &http.Server{
		// The card registered for this agent declares HTTP+JSON, so serve REST.
		Handler: a2asrv.NewRESTHandler(handler),
		// Only localhost here, but readers copy sample servers into deployments.
		ReadHeaderTimeout: 10 * time.Second,
	}

	// Bind before returning: net.Listen either takes the port or says why not,
	// so the caller never races a server that has not come up yet.
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, err
	}
	go func() {
		if err := srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Printf("A2A server stopped: %v", err)
		}
	}()
	return srv, nil
}

// explain renders a registry failure as something a human can act on. The
// service reports a denial as a screenful of JSON, so unwrap the typed
// [agentregistry.APIError] and keep the parts that identify the fix.
//
// It returns text, not an error: wrapping the result back into one with %w
// would re-append the envelope it exists to suppress.
func explain(err error) string {
	var apiErr *agentregistry.APIError
	if !errors.As(err, &apiErr) {
		return err.Error()
	}
	var envelope struct {
		Error struct {
			Message string `json:"message"`
			Status  string `json:"status"`
		} `json:"error"`
	}
	detail := apiErr.Body
	if json.Unmarshal([]byte(apiErr.Body), &envelope) == nil && envelope.Error.Message != "" {
		detail = envelope.Error.Message
		if envelope.Error.Status != "" {
			detail = envelope.Error.Status + ": " + detail
		}
	}
	if apiErr.StatusCode == http.StatusForbidden {
		detail += "\nGrant roles/agentregistry.viewer on this project, or point GOOGLE_CLOUD_PROJECT at one where you have it."
	}
	return fmt.Sprintf("HTTP %d — %s", apiErr.StatusCode, detail)
}
