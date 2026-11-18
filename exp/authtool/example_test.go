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

package authtool_test

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

	"github.com/google/jsonschema-go/jsonschema"
	"golang.org/x/oauth2"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/credentialservice"
	"google.golang.org/adk/v2/exp/authtool"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/toolauth"
)

// A tool that calls an API with an API key. The function receives the key as
// a credential; the model sees only the city parameter.
func ExampleNew() {
	type weatherArgs struct {
		City string `json:"city"`
	}
	weather, err := authtool.New(authtool.Config{
		Config: functiontool.Config{Name: "get_weather", Description: "Returns the weather in a city."},
		AuthConfig: &toolauth.Config{
			AuthScheme:        toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey, In: "header", Name: "X-Api-Key"},
			RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "example-key"},
		},
	}, func(ctx agent.Context, a weatherArgs, cred *toolauth.Credential, authErr error) (map[string]any, error) {
		// A real tool sends its request with the header X-Api-Key: cred.APIKey.
		return map[string]any{"city": a.City, "forecast": "sunny", "key_set": cred.APIKey != ""}, nil
	})
	if err != nil {
		log.Fatal(err)
	}

	// A scripted model stands in for Gemini: it calls the tool, then answers.
	llm := &scriptedModel{replies: []*genai.Content{
		genai.NewContentFromFunctionCall("get_weather", map[string]any{"city": "Paris"}, genai.RoleModel),
		genai.NewContentFromText("It is sunny in Paris.", genai.RoleModel),
	}}
	a, err := llmagent.New(llmagent.Config{Name: "weather_agent", Model: llm, Tools: []tool.Tool{weather}})
	if err != nil {
		log.Fatal(err)
	}
	sessions := session.InMemoryService()
	r, err := runner.New(runner.Config{AppName: "weather", Agent: a, SessionService: sessions})
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()
	s, err := sessions.Create(ctx, &session.CreateRequest{AppName: "weather", UserID: "user-1"})
	if err != nil {
		log.Fatal(err)
	}

	msg := genai.NewContentFromText("What is the weather in Paris?", genai.RoleUser)
	for ev, err := range r.Run(ctx, "user-1", s.Session.ID(), msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
		if ev.Content == nil {
			continue
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
	// tool: map[city:Paris forecast:sunny key_set:true]
	// model: It is sunny in Paris.
}

// weatherHandler is the handler of a tool that is not built from a typed Go
// function. It reads its arguments from the map the model sent.
type weatherHandler struct{}

func (weatherHandler) Call(ctx agent.Context, args any, cred *toolauth.Credential, authErr error) (map[string]any, error) {
	city, _ := args.(map[string]any)["city"].(string)
	// A real tool sends its request with the header X-Api-Key: cred.APIKey.
	return map[string]any{"city": city, "forecast": "rain", "key_set": cred.APIKey != ""}, nil
}

// A custom authenticated tool: the tool declares its own input schema and a
// BaseHandler receives the arguments as the model sent them, with the
// credential.
func ExampleNewBaseTool() {
	weather, err := authtool.NewBaseTool(authtool.Config{
		Config: functiontool.Config{
			Name:        "get_weather",
			Description: "Returns the weather in a city.",
			InputSchema: &jsonschema.Schema{
				Type:       "object",
				Properties: map[string]*jsonschema.Schema{"city": {Type: "string"}},
				Required:   []string{"city"},
			},
		},
		AuthConfig: &toolauth.Config{
			AuthScheme:        toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey, In: "header", Name: "X-Api-Key"},
			RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "example-key"},
		},
	}, weatherHandler{})
	if err != nil {
		log.Fatal(err)
	}

	llm := &scriptedModel{replies: []*genai.Content{
		genai.NewContentFromFunctionCall("get_weather", map[string]any{"city": "Oslo"}, genai.RoleModel),
		genai.NewContentFromText("It is raining in Oslo.", genai.RoleModel),
	}}
	a, err := llmagent.New(llmagent.Config{Name: "weather_agent", Model: llm, Tools: []tool.Tool{weather}})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: "weather", Agent: a, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		log.Fatal(err)
	}

	msg := genai.NewContentFromText("What is the weather in Oslo?", genai.RoleUser)
	for ev, err := range r.Run(context.Background(), "user-1", "s1", msg, agent.RunConfig{}) {
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
	// tool: map[city:Oslo forecast:rain key_set:true]
	// model: It is raining in Oslo.
}

// A tool that uses a CredentialManager itself. The manager finds no
// credential, so the tool asks the user to sign in. When the client answers
// with the redirect URL, the manager redeems the code and saves the tokens in
// the runner's credential service, so the same user's next session needs no
// sign-in.
func ExampleNewCredentialManager() {
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
	manager, err := authtool.NewCredentialManager(cfg, nil)
	if err != nil {
		log.Fatal(err)
	}
	calendar, err := functiontool.New(functiontool.Config{Name: "list_events", Description: "Lists the user's calendar events."},
		func(ctx agent.Context, _ struct{}) (map[string]any, error) {
			cred, err := manager.GetAuthCredential(ctx)
			if err != nil {
				return nil, err
			}
			if cred == nil {
				if err := manager.RequestCredential(ctx); err != nil {
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

	llm := &scriptedModel{replies: []*genai.Content{
		genai.NewContentFromFunctionCall("list_events", map[string]any{}, genai.RoleModel),
		genai.NewContentFromText("You have standup.", genai.RoleModel),
		genai.NewContentFromFunctionCall("list_events", map[string]any{}, genai.RoleModel),
		genai.NewContentFromText("You have standup.", genai.RoleModel),
	}}
	a, err := llmagent.New(llmagent.Config{Name: "calendar_agent", Model: llm, Tools: []tool.Tool{calendar}})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "calendar",
		Agent:             a,
		SessionService:    session.InMemoryService(),
		CredentialService: credentialservice.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		log.Fatal(err)
	}
	// The token request goes to an in-memory endpoint instead of the network.
	ctx := context.WithValue(context.Background(), oauth2.HTTPClient, &http.Client{Transport: memoryTokenEndpoint{}})

	// run prints the tool results of one invocation and returns the
	// credential request, if ADK sent one.
	run := func(sessionID string, msg *genai.Content) *genai.FunctionCall {
		var request *genai.FunctionCall
		for ev, err := range r.Run(ctx, "user-1", sessionID, msg, agent.RunConfig{}) {
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
		return request
	}

	fmt.Println("session-a, first message")
	request := run("session-a", genai.NewContentFromText("What is on my calendar?", genai.RoleUser))
	fmt.Println("session-a, the client answers")
	run("session-a", signInAnswer(request, "code-1"))
	fmt.Println("session-b")
	run("session-b", genai.NewContentFromText("What is on my calendar?", genai.RoleUser))
	// Output:
	// session-a, first message
	// tool: map[status:waiting for the user to sign in]
	// client: ADK asks for a credential
	// session-a, the client answers
	// token endpoint: grant_type=authorization_code code=code-1
	// tool: map[events:standup signed_in:true]
	// session-b
	// tool: map[events:standup signed_in:true]
}

// signInAnswer is what a client sends after the user signed in: the config of
// the adk_request_credential call request, with the URL the provider
// redirected the user to, carrying code and the state of the sign-in link.
func signInAnswer(request *genai.FunctionCall, code string) *genai.Content {
	var args toolauth.ToolArguments
	b, err := json.Marshal(request.Args)
	if err != nil {
		log.Fatal(err)
	}
	if err := json.Unmarshal(b, &args); err != nil {
		log.Fatal(err)
	}
	signIn := args.AuthConfig.ExchangedAuthCredential.OAuth2
	link, err := url.Parse(signIn.AuthURI)
	if err != nil {
		log.Fatal(err)
	}
	signIn.AuthResponseURI = "https://app.example.com/callback?code=" + code + "&state=" + url.QueryEscape(link.Query().Get("state"))
	b, err = json.Marshal(args.AuthConfig)
	if err != nil {
		log.Fatal(err)
	}
	var response map[string]any
	if err := json.Unmarshal(b, &response); err != nil {
		log.Fatal(err)
	}
	return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
		ID:       request.ID,
		Name:     toolauth.RequestCredentialFunctionCallName,
		Response: response,
	}}}}
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
