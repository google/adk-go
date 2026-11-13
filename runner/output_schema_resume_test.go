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
	"path/filepath"
	"reflect"
	"testing"

	"github.com/glebarez/sqlite"
	"google.golang.org/genai"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session/database"
	"google.golang.org/adk/v2/workflow"
)

// resumedJoin is what the resumed turn showed for the join and the node after it.
type resumedJoin struct {
	output  map[string]any
	sawJoin bool
	sawSink bool
}

// runSchemaReplyThroughJoin runs a workflow on the database session service
// where an LlmAgent with an OutputSchema replies with reply, a sibling agent
// pauses for input, and both feed a join. It returns what the turn that
// resumes the sibling emitted.
func runSchemaReplyThroughJoin(t *testing.T, reply string) resumedJoin {
	t.Helper()
	svc, err := database.NewSessionService(sqlite.Open(filepath.Join(t.TempDir(), "sessions.db")),
		&gorm.Config{Logger: logger.Default.LogMode(logger.Silent)})
	if err != nil {
		t.Fatal(err)
	}
	if err := database.AutoMigrate(svc); err != nil {
		t.Fatal(err)
	}
	m := &scriptedModel{responses: []*genai.Content{genai.NewContentFromText(reply, "model")}}
	s, err := llmagent.New(llmagent.Config{Name: "s", Model: m, OutputSchema: &genai.Schema{
		Type:       genai.TypeObject,
		Properties: map[string]*genai.Schema{"title": {Type: genai.TypeString}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	b := joinResumeAgent(t, "b", "B", true)
	sn, bn := joinResumeAgentNode(t, s), joinResumeAgentNode(t, b)
	join := workflow.NewJoinNode("join")
	sink := workflow.NewFunctionNode("sink", func(agent.Context, map[string]any) (string, error) {
		return "done", nil
	}, workflow.NodeConfig{})
	root, err := workflowagent.New(workflowagent.Config{Name: "root", SubAgents: []agent.Agent{s, b}, Edges: []workflow.Edge{
		{From: workflow.Start, To: sn},
		{From: workflow.Start, To: bn},
		{From: sn, To: join},
		{From: bn, To: join},
		{From: join, To: sink},
	}})
	if err != nil {
		t.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: "repro", Agent: root, SessionService: svc, AutoCreateSession: true})
	if err != nil {
		t.Fatal(err)
	}
	for _, err := range r.Run(t.Context(), "u", "s1", genai.NewContentFromText("start", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatal(err)
		}
	}
	var got resumedJoin
	for ev, err := range r.Run(t.Context(), "u", "s1", resumeContent("b", workflow.WorkflowInputFunctionCallName, "approved"), agent.RunConfig{}) {
		if err != nil {
			t.Fatal(err)
		}
		if out, ok := ev.Output.(map[string]any); ok {
			got.sawJoin = true
			got.output = out
		}
		if ev.Output == "done" {
			got.sawSink = true
		}
	}
	return got
}

// A literal null reply has no required fields to reject, so it passes
// validation as a nil map. The database backend stores that value as null and
// reads back no output, so the agent's event has to keep the reply text as the
// node output for the join to see it after the resume.
func TestNullSchemaReplySurvivesResume(t *testing.T) {
	got := runSchemaReplyThroughJoin(t, "null")
	if !got.sawJoin || !got.sawSink {
		t.Fatalf("resume reached join = %v, sink = %v, want both", got.sawJoin, got.sawSink)
	}
	if want := (map[string]any{"s": "null", "b": "approved"}); !reflect.DeepEqual(got.output, want) {
		t.Errorf("join output = %v, want %v", got.output, want)
	}
}

// A parsed reply is stored on the event as the node output, so the join gets
// the record after the resume instead of the reply text.
func TestParsedSchemaReplySurvivesResumeOnDatabase(t *testing.T) {
	got := runSchemaReplyThroughJoin(t, `{"title":"T"}`)
	if !got.sawJoin || !got.sawSink {
		t.Fatalf("resume reached join = %v, sink = %v, want both", got.sawJoin, got.sawSink)
	}
	want := map[string]any{"s": map[string]any{"title": "T"}, "b": "approved"}
	if !reflect.DeepEqual(got.output, want) {
		t.Errorf("join output = %v, want %v", got.output, want)
	}
}
