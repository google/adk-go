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

package llminternal

import (
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

func TestTurnCycleGuard(t *testing.T) {
	tests := []struct {
		name string
		run  func(t *testing.T, g *turnCycleGuard)
	}{
		{
			name: "initial onModelContent returns false",
			run: func(t *testing.T, g *turnCycleGuard) {
				if g.onModelContent() {
					t.Error("expected onModelContent to return false initially")
				}
			},
		},
		{
			name: "after content and TurnComplete, onModelContent returns true",
			run: func(t *testing.T, g *turnCycleGuard) {
				g.onModelContent()
				g.onTurnComplete()
				if !g.onModelContent() {
					t.Error("expected onModelContent to return true (suppress) after content+TurnComplete")
				}
			},
		},
		{
			name: "after reset, onModelContent returns false",
			run: func(t *testing.T, g *turnCycleGuard) {
				g.onModelContent()
				g.onTurnComplete()
				g.reset()
				if g.onModelContent() {
					t.Error("expected onModelContent to return false after reset")
				}
			},
		},
		{
			name: "TurnComplete without prior content does not arm suppression",
			run: func(t *testing.T, g *turnCycleGuard) {
				g.onTurnComplete()
				if g.onModelContent() {
					t.Error("expected onModelContent to return false — TurnComplete without content should not arm")
				}
			},
		},
		{
			name: "multiple onModelContent before TurnComplete all return false",
			run: func(t *testing.T, g *turnCycleGuard) {
				for i := range 5 {
					if g.onModelContent() {
						t.Errorf("onModelContent[%d] should return false before TurnComplete", i)
					}
				}
				g.onTurnComplete()
				if !g.onModelContent() {
					t.Error("expected suppression after multiple content + TurnComplete")
				}
			},
		},
		{
			name: "suppression persists across multiple calls",
			run: func(t *testing.T, g *turnCycleGuard) {
				g.onModelContent()
				g.onTurnComplete()
				for i := range 3 {
					if !g.onModelContent() {
						t.Errorf("onModelContent[%d] should remain suppressed", i)
					}
				}
			},
		},
		{
			name: "reset is idempotent",
			run: func(t *testing.T, g *turnCycleGuard) {
				g.reset()
				g.reset()
				if g.onModelContent() {
					t.Error("expected false after idempotent resets")
				}
			},
		},
		{
			name: "markContentDelivered then onTurnComplete arms",
			run: func(t *testing.T, g *turnCycleGuard) {
				g.markContentDelivered()
				g.onTurnComplete()
				if !g.onModelContent() {
					t.Error("expected suppression after markContentDelivered + onTurnComplete")
				}
			},
		},
		{
			name: "markContentDelivered alone does not arm",
			run: func(t *testing.T, g *turnCycleGuard) {
				g.markContentDelivered()
				if g.onModelContent() {
					t.Error("markContentDelivered without onTurnComplete should not arm")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := &turnCycleGuard{}
			tt.run(t, g)
		})
	}
}

// modelTextResp is a helper for applyTurnCycleGuard tests that returns a
// model-role text response.
func modelTextResp(text string) *model.LLMResponse {
	return &model.LLMResponse{Content: genai.NewContentFromText(text, "model")}
}

func TestApplyTurnCycleGuard_Interrupted(t *testing.T) {
	t.Run("Interrupted bypasses suppression and arms", func(t *testing.T) {
		g := &turnCycleGuard{}
		// Arm the guard via a normal turn.
		_ = applyTurnCycleGuard(&model.LLMResponse{
			Content:      genai.NewContentFromText("A", "model"),
			TurnComplete: true,
		}, g, false)

		// Interrupted message must be delivered (not suppressed) even
		// while the guard is armed.
		out := applyTurnCycleGuard(&model.LLMResponse{
			Content:     genai.NewContentFromText("B", "model"),
			Interrupted: true,
		}, g, false)
		if out == nil {
			t.Fatal("Interrupted message must not be suppressed")
		}
		if out.Content == nil || out.Content.Parts[0].Text != "B" {
			t.Errorf("expected Interrupted content B, got %+v", out.Content)
		}

		// A duplicate that follows must be suppressed: Interrupted armed.
		dup := applyTurnCycleGuard(modelTextResp("B"), g, false)
		if dup != nil {
			t.Error("duplicate after Interrupted must be suppressed (Interrupted should arm)")
		}
	})
}

func TestApplyTurnCycleGuard_PartialDoesNotArm(t *testing.T) {
	g := &turnCycleGuard{}
	// Partial+TurnComplete with content should NOT arm the guard.
	_ = applyTurnCycleGuard(&model.LLMResponse{
		Content:      genai.NewContentFromText("A", "model"),
		Partial:      true,
		TurnComplete: true,
	}, g, false)

	// Subsequent model content with no Partial must NOT be suppressed.
	out := applyTurnCycleGuard(modelTextResp("A"), g, false)
	if out == nil {
		t.Error("partial TurnComplete should not arm guard; subsequent content was incorrectly suppressed")
	}
}

func TestApplyTurnCycleGuard_StripPathMatrix(t *testing.T) {
	cases := []struct {
		name           string
		resp           *model.LLMResponse
		isAudio        bool
		armBeforeCall  bool
		wantNil        bool
		wantContent    bool
		wantOT         bool
		wantIT         bool
		wantUnchanged  bool // if true, returned response must equal input pointer-wise
		descriptionKey string
	}{
		{
			name: "model content only — fully suppressed",
			resp: &model.LLMResponse{
				Content: genai.NewContentFromText("dup", "model"),
			},
			armBeforeCall: true,
			wantNil:       true,
		},
		{
			name: "model content + OT (no IT) — fully suppressed",
			resp: &model.LLMResponse{
				Content:             genai.NewContentFromText("dup", "model"),
				OutputTranscription: &genai.Transcription{Text: "dup", Finished: true},
			},
			armBeforeCall: true,
			wantNil:       true,
		},
		{
			name: "model content + IT — Content stripped, IT preserved",
			resp: &model.LLMResponse{
				Content:            genai.NewContentFromText("dup", "model"),
				InputTranscription: &genai.Transcription{Text: "user said hi", Finished: true},
			},
			armBeforeCall: true,
			wantContent:   false,
			wantOT:        false,
			wantIT:        true,
		},
		{
			name: "model content + OT + IT — Content+OT stripped, IT preserved",
			resp: &model.LLMResponse{
				Content:             genai.NewContentFromText("dup", "model"),
				OutputTranscription: &genai.Transcription{Text: "dup", Finished: true},
				InputTranscription:  &genai.Transcription{Text: "user said hi", Finished: true},
			},
			armBeforeCall: true,
			wantContent:   false,
			wantOT:        false,
			wantIT:        true,
		},
		{
			name: "OT-only (no Content) — bypass: response unchanged",
			resp: &model.LLMResponse{
				OutputTranscription: &genai.Transcription{Text: "out", Finished: true},
			},
			armBeforeCall: true,
			wantUnchanged: true,
		},
		{
			name: "IT-only (no Content) — bypass: response unchanged",
			resp: &model.LLMResponse{
				InputTranscription: &genai.Transcription{Text: "in", Finished: true},
			},
			armBeforeCall: true,
			wantUnchanged: true,
		},
		{
			name: "audio model content — bypass via isAudio carve-out",
			resp: &model.LLMResponse{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{InlineData: &genai.Blob{MIMEType: "audio/pcm", Data: []byte{0x01}}},
					},
				},
			},
			isAudio:       true,
			armBeforeCall: true,
			wantUnchanged: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g := &turnCycleGuard{}
			if tc.armBeforeCall {
				// Arm the guard with a prior turn so suppression is active.
				_ = applyTurnCycleGuard(&model.LLMResponse{
					Content:      genai.NewContentFromText("prior", "model"),
					TurnComplete: true,
				}, g, false)
			}

			out := applyTurnCycleGuard(tc.resp, g, tc.isAudio)

			if tc.wantNil {
				if out != nil {
					t.Errorf("expected nil (fully suppressed), got %+v", out)
				}
				return
			}
			if tc.wantUnchanged {
				if out != tc.resp {
					t.Errorf("expected unchanged response (same pointer), got different value")
				}
				return
			}
			if out == nil {
				t.Fatal("unexpected nil result")
			}
			if tc.wantContent != (out.Content != nil) {
				t.Errorf("Content presence: got %v, want %v", out.Content != nil, tc.wantContent)
			}
			if tc.wantOT != (out.OutputTranscription != nil) {
				t.Errorf("OutputTranscription presence: got %v, want %v", out.OutputTranscription != nil, tc.wantOT)
			}
			if tc.wantIT != (out.InputTranscription != nil) {
				t.Errorf("InputTranscription presence: got %v, want %v", out.InputTranscription != nil, tc.wantIT)
			}
		})
	}
}
