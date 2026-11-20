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

package runner_test

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"log"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// echoColorModel stands in for a model: it answers with the last word of
// the user's latest message, and prints the conversation it was sent.
type echoColorModel struct{}

func (echoColorModel) Name() string { return "echo-color" }

func (echoColorModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		var seen []string
		for _, c := range req.Contents {
			seen = append(seen, c.Role+": "+c.Parts[0].Text)
		}
		fmt.Printf("  model sees [%s]\n", strings.Join(seen, ", "))
		words := strings.Fields(req.Contents[len(req.Contents)-1].Parts[0].Text)
		yield(&model.LLMResponse{Content: genai.NewContentFromText(words[len(words)-1], genai.RoleModel)}, nil)
	}
}

// Rewind undoes the invocation it names and every invocation after it: the
// session's state goes back to what it was before that invocation, and
// later model requests leave out the rewound turns, whose events stay in
// the session.
func ExampleRunner_Rewind() {
	assistant, err := llmagent.New(llmagent.Config{
		Name:      "assistant",
		Model:     echoColorModel{},
		OutputKey: "color", // The agent's reply is stored in the session state.
	})
	if err != nil {
		log.Fatal(err)
	}
	sessions := session.InMemoryService()
	r, err := runner.New(runner.Config{
		AppName:           "colors",
		Agent:             assistant,
		SessionService:    sessions,
		AutoCreateSession: true,
	})
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()

	// turn runs one user turn and returns its invocation ID.
	turn := func(text string) string {
		fmt.Println("user:", text)
		invocationID := ""
		msg := genai.NewContentFromText(text, genai.RoleUser)
		for ev, err := range r.Run(ctx, "user", "s1", msg, agent.RunConfig{}) {
			if err != nil {
				log.Fatal(err)
			}
			invocationID = ev.InvocationID
		}
		return invocationID
	}
	color := func() {
		resp, err := sessions.Get(ctx, &session.GetRequest{AppName: "colors", UserID: "user", SessionID: "s1"})
		if err != nil {
			log.Fatal(err)
		}
		value, _ := resp.Session.State().Get("color")
		fmt.Println("  color:", value)
	}

	turn("set the color to red")
	color()
	blue := turn("change the color to blue")
	color()

	// Undo the second turn.
	if err := r.Rewind(ctx, "user", "s1", blue, nil); err != nil {
		log.Fatal(err)
	}
	fmt.Println("rewound")
	color()

	turn("now make it green")
	color()

	// Output:
	// user: set the color to red
	//   model sees [user: set the color to red]
	//   color: red
	// user: change the color to blue
	//   model sees [user: set the color to red, model: red, user: change the color to blue]
	//   color: blue
	// rewound
	//   color: red
	// user: now make it green
	//   model sees [user: set the color to red, model: red, user: now make it green]
	//   color: green
}

// Rewind refuses an invocation ID that names no event of the session, and
// changes nothing then.
func ExampleRunner_Rewind_unknownInvocation() {
	sessions := session.InMemoryService()
	if _, err := sessions.Create(context.Background(), &session.CreateRequest{AppName: "colors", UserID: "user", SessionID: "s1"}); err != nil {
		log.Fatal(err)
	}
	assistant, err := llmagent.New(llmagent.Config{Name: "assistant", Model: echoColorModel{}})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: "colors", Agent: assistant, SessionService: sessions})
	if err != nil {
		log.Fatal(err)
	}

	err = r.Rewind(context.Background(), "user", "s1", "no-such-invocation", nil)
	fmt.Println(errors.Is(err, runner.ErrInvocationNotFound))

	// Output:
	// true
}
