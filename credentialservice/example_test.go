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

package credentialservice_test

import (
	"context"
	"fmt"
	"iter"
	"log"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/credentialservice"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/toolauth"
)

// A runner with an in-memory credential service. The tool loads its
// credential with agent.LoadCredential; when there is none it obtains one and
// saves it with agent.SaveCredential. A later invocation of the same user, in
// another session, loads the saved credential; another user's invocation
// does not see it.
func ExampleInMemoryService() {
	cfg := &toolauth.Config{
		AuthScheme:    toolauth.Scheme{Type: toolauth.SchemeTypeHTTP, Scheme: "bearer"},
		CredentialKey: "profile-api",
	}
	profile, err := functiontool.New(functiontool.Config{Name: "get_profile", Description: "Returns the user's profile."},
		func(ctx agent.Context, _ struct{}) (map[string]any, error) {
			cred, err := agent.LoadCredential(ctx, cfg)
			if err != nil {
				return nil, err
			}
			if cred != nil {
				return map[string]any{"credential": "loaded from the credential service"}, nil
			}
			// A real tool obtains the token here, for example by asking the
			// user to sign in.
			obtained := *cfg
			obtained.ExchangedAuthCredential = &toolauth.Credential{
				AuthType: toolauth.CredentialTypeHTTP,
				HTTP:     &toolauth.HTTPAuth{Scheme: "bearer", Credentials: toolauth.HTTPCredentials{Token: "token-of-" + ctx.UserID()}},
			}
			if err := agent.SaveCredential(ctx, &obtained); err != nil {
				return nil, err
			}
			return map[string]any{"credential": "obtained and saved"}, nil
		})
	if err != nil {
		log.Fatal(err)
	}

	// A scripted model stands in for Gemini: in each invocation it calls the
	// tool, then answers.
	var replies []*genai.Content
	for range 3 {
		replies = append(replies,
			genai.NewContentFromFunctionCall("get_profile", map[string]any{}, genai.RoleModel),
			genai.NewContentFromText("Done.", genai.RoleModel))
	}
	a, err := llmagent.New(llmagent.Config{Name: "profile_agent", Model: &scriptedModel{replies: replies}, Tools: []tool.Tool{profile}})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "profiles",
		Agent:             a,
		SessionService:    session.InMemoryService(),
		CredentialService: credentialservice.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		log.Fatal(err)
	}

	for _, run := range []struct{ userID, sessionID string }{
		{"user-1", "session-a"},
		{"user-1", "session-b"},
		{"user-2", "session-c"},
	} {
		msg := genai.NewContentFromText("Show my profile.", genai.RoleUser)
		for ev, err := range r.Run(context.Background(), run.userID, run.sessionID, msg, agent.RunConfig{}) {
			if err != nil {
				log.Fatal(err)
			}
			for _, p := range ev.Content.Parts {
				if p.FunctionResponse != nil {
					fmt.Printf("%s, %s: %v\n", run.userID, run.sessionID, p.FunctionResponse.Response["credential"])
				}
			}
		}
	}
	// Output:
	// user-1, session-a: obtained and saved
	// user-1, session-b: loaded from the credential service
	// user-2, session-c: obtained and saved
}

// scriptedModel answers each request with the next of its replies.
type scriptedModel struct {
	replies []*genai.Content
}

func (m *scriptedModel) Name() string { return "scripted" }

func (m *scriptedModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if len(m.replies) == 0 {
			yield(nil, fmt.Errorf("scripted model has no reply left"))
			return
		}
		reply := m.replies[0]
		m.replies = m.replies[1:]
		yield(&model.LLMResponse{Content: reply, FinishReason: genai.FinishReasonStop}, nil)
	}
}
