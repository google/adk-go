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

package authflow_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"log"
	"net/http"
	"net/url"
	"strings"

	"golang.org/x/oauth2"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/exp/authflow"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/toolauth"
)

// A tool that needs the user to sign in with OAuth2. On its first call the
// tool finds no answer, asks for the credential and returns; the client signs
// the user in and answers the adk_request_credential call with the redirect
// URL. ADK then runs the tool call again, and GetAuthResponse redeems the
// code from that URL for tokens.
func ExampleRequestCredential() {
	cfg := &toolauth.Config{
		AuthScheme: toolauth.Scheme{
			Type: toolauth.SchemeTypeOAuth2,
			Flows: &toolauth.OAuthFlows{AuthorizationCode: &toolauth.OAuthFlow{
				AuthorizationURL: "https://auth.example.com/authorize",
				TokenURL:         "https://auth.example.com/token",
				Scopes:           map[string]string{"calendar.read": "Read your calendar"},
			}},
		},
		RawAuthCredential: &toolauth.Credential{
			AuthType: toolauth.CredentialTypeOAuth2,
			OAuth2: &toolauth.OAuth2Auth{
				ClientID:     "example-client",
				ClientSecret: "example-secret",
				RedirectURI:  "https://app.example.com/callback",
			},
		},
	}
	calendar, err := functiontool.New(functiontool.Config{Name: "list_events", Description: "Lists the user's calendar events."},
		func(ctx agent.Context, _ struct{}) (map[string]any, error) {
			cred, err := authflow.GetAuthResponse(ctx, cfg)
			if err != nil {
				return nil, err
			}
			if cred == nil {
				if err := authflow.RequestCredential(ctx, cfg); err != nil {
					return nil, err
				}
				return map[string]any{"status": "waiting for the user to sign in"}, nil
			}
			// A real tool calls the calendar API with cred.OAuth2.AccessToken.
			return map[string]any{"events": "standup", "signed_in": cred.OAuth2.AccessToken != ""}, nil
		})
	if err != nil {
		log.Fatal(err)
	}

	// A scripted model stands in for Gemini: it calls the tool, then answers.
	llm := &scriptedModel{replies: []*genai.Content{
		genai.NewContentFromFunctionCall("list_events", map[string]any{}, genai.RoleModel),
		genai.NewContentFromText("You have standup.", genai.RoleModel),
	}}
	a, err := llmagent.New(llmagent.Config{Name: "calendar_agent", Model: llm, Tools: []tool.Tool{calendar}})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: "calendar", Agent: a, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		log.Fatal(err)
	}
	// The token request goes to an in-memory endpoint instead of the network.
	ctx := context.WithValue(context.Background(), oauth2.HTTPClient, &http.Client{Transport: memoryTokenEndpoint{}})

	fmt.Println("run 1")
	var request *genai.FunctionCall
	msg := genai.NewContentFromText("What is on my calendar?", genai.RoleUser)
	for ev, err := range r.Run(ctx, "user-1", "s1", msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
		for _, p := range ev.Content.Parts {
			if p.FunctionResponse != nil {
				fmt.Println("tool:", p.FunctionResponse.Response)
			}
			if p.FunctionCall != nil && p.FunctionCall.Name == toolauth.RequestCredentialFunctionCallName {
				fmt.Println("client: ADK asks for a credential")
				request = p.FunctionCall
			}
		}
	}

	// The client reads the config from the request and sends the user to its
	// sign-in link. The provider then redirects the user to the redirect URI
	// with a code and the state the link carried.
	var args toolauth.ToolArguments
	b, err := json.Marshal(request.Args)
	if err != nil {
		log.Fatal(err)
	}
	if err := json.Unmarshal(b, &args); err != nil {
		log.Fatal(err)
	}
	signIn := args.AuthConfig.ExchangedAuthCredential.OAuth2
	fmt.Println("client: sign-in link goes to auth.example.com:", strings.HasPrefix(signIn.AuthURI, "https://auth.example.com/authorize?"))
	fmt.Println("client: request carries the client secret:", signIn.ClientSecret != "")
	link, err := url.Parse(signIn.AuthURI)
	if err != nil {
		log.Fatal(err)
	}
	signIn.AuthResponseURI = "https://app.example.com/callback?code=code-1&state=" + url.QueryEscape(link.Query().Get("state"))

	// The client answers the request with the config, as a function response.
	b, err = json.Marshal(args.AuthConfig)
	if err != nil {
		log.Fatal(err)
	}
	var response map[string]any
	if err := json.Unmarshal(b, &response); err != nil {
		log.Fatal(err)
	}
	answer := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
		ID:       request.ID,
		Name:     toolauth.RequestCredentialFunctionCallName,
		Response: response,
	}}}}

	fmt.Println("run 2")
	for ev, err := range r.Run(ctx, "user-1", "s1", answer, agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
		for _, p := range ev.Content.Parts {
			switch {
			case p.FunctionResponse != nil:
				fmt.Println("tool:", p.FunctionResponse.Response)
			case p.Text != "":
				fmt.Println("model:", p.Text)
			}
		}
	}
	// Output:
	// run 1
	// tool: map[status:waiting for the user to sign in]
	// client: ADK asks for a credential
	// client: sign-in link goes to auth.example.com: true
	// client: request carries the client secret: false
	// run 2
	// token endpoint: grant_type=authorization_code code=code-1
	// tool: map[events:standup signed_in:true]
	// model: You have standup.
}

// memoryTokenEndpoint is an OAuth2 token endpoint in memory. It prints what it
// receives and returns a token.
type memoryTokenEndpoint struct{}

func (memoryTokenEndpoint) RoundTrip(req *http.Request) (*http.Response, error) {
	if err := req.ParseForm(); err != nil {
		return nil, err
	}
	fmt.Printf("token endpoint: grant_type=%s code=%s\n", req.PostForm.Get("grant_type"), req.PostForm.Get("code"))
	body := `{"access_token":"access-token-1","token_type":"Bearer","expires_in":3600}`
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": {"application/json"}},
		Body:       io.NopCloser(strings.NewReader(body)),
		Request:    req,
	}, nil
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
