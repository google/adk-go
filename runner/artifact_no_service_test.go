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

package runner

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// TestRunner_SaveInputBlobsWithoutArtifactService checks that a runner with
// no artifact service skips SaveInputBlobsAsArtifacts: the run succeeds and
// the user event keeps its inline data (adk-python Runner._append_new_message_to_session).
func TestRunner_SaveInputBlobsWithoutArtifactService(t *testing.T) {
	for _, kind := range cmRootKinds {
		t.Run(kind.name, func(t *testing.T) {
			r, svc := cmNewRunner(t, kind.newRoot(t, "done", nil))
			msg := &genai.Content{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					genai.NewPartFromText("here is a file"),
					{InlineData: &genai.Blob{MIMEType: "text/plain", Data: []byte("blob")}},
				},
			}

			cmCollect(t, r.Run(t.Context(), cmUser, "s1", msg, agent.RunConfig{SaveInputBlobsAsArtifacts: true}))

			got, err := svc.Get(t.Context(), &session.GetRequest{AppName: cmApp, UserID: cmUser, SessionID: "s1"})
			if err != nil {
				t.Fatalf("Get() error = %v", err)
			}
			userEvent := got.Session.Events().At(0)
			want := []*genai.Part{
				{Text: "here is a file"},
				{InlineData: &genai.Blob{MIMEType: "text/plain", Data: []byte("blob")}},
			}
			if diff := cmp.Diff(want, userEvent.Content.Parts); diff != "" {
				t.Errorf("user event parts mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
