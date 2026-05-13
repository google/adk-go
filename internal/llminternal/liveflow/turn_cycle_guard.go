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

package liveflow

import (
	"google.golang.org/adk/model"
)

// turnCycleGuard tracks turn-cycle boundaries within receiverLoop to
// suppress duplicate model content caused by orphaned tool results.
// NOT goroutine-safe — only accessed from the receiverLoop goroutine.
type turnCycleGuard struct {
	contentDelivered bool // model content emitted since last reset
	suppressActive   bool // suppress subsequent model content
}

func (g *turnCycleGuard) onModelContent() bool {
	if g.suppressActive {
		return true // suppress
	}
	g.contentDelivered = true
	return false
}

func (g *turnCycleGuard) onTurnComplete() {
	if g.contentDelivered {
		g.suppressActive = true
	}
}

// markContentDelivered records that model content was emitted in the current
// turn. Used by the Interrupted bypass path so a subsequent onTurnComplete
// arms suppression for any post-Interrupted duplicates that may follow.
func (g *turnCycleGuard) markContentDelivered() {
	g.contentDelivered = true
}

func (g *turnCycleGuard) reset() {
	g.contentDelivered = false
	g.suppressActive = false
}

// applyTurnCycleGuard evaluates the guard for the given response and returns
// the response to forward downstream, or nil when the message should be
// fully suppressed.
//
// Suppression matrix (when the guard is armed and the message is model-
// content-bearing — i.e. Content!=nil with role=="model" and !isAudio):
//
//	Content+OT only       → nil (fully suppressed)
//	Content+OT+IT         → Content/OT stripped, IT preserved
//	Content+IT (no OT)    → Content stripped, IT preserved
//
// OT-stripping nuance: OutputTranscription is stripped only when the
// response is model-content-bearing AND the guard suppresses. An OT-only
// message (Content==nil) is a documented bypass case — isModelContent is
// false, the guard is not consulted, and OT is forwarded untouched. Audio
// model content also bypasses the guard via the !isAudio carve-out, so
// audio frames are never stripped here.
//
// InputTranscription (user input) is never stripped: it is not a model
// duplicate. The Interrupted branch bypasses the suppress check for the
// current message (it carries the truncated model output, which the
// caller must observe) and arms the guard so post-Interrupted duplicates
// that arrive before user activity are suppressed.
func applyTurnCycleGuard(resp *model.LLMResponse, guard *turnCycleGuard, isAudio bool) *model.LLMResponse {
	isModelContent := resp.Content != nil && resp.Content.Role == "model" && !isAudio

	// Interrupted: the truncated model output IS this message — deliver it
	// (bypass suppression). Arm the guard so any subsequent duplicates that
	// arrive before user activity are suppressed.
	if resp.Interrupted {
		if isModelContent {
			guard.markContentDelivered()
		}
		guard.onTurnComplete()
		return resp
	}

	suppressModelContent := false
	if isModelContent {
		suppressModelContent = guard.onModelContent()
	}

	// Arm only on a non-partial TurnComplete. Partial events fire before
	// the aggregate that carries the real content; arming on them would
	// strip the aggregate's content.
	if resp.TurnComplete && !resp.Partial {
		guard.onTurnComplete()
	}

	if !suppressModelContent {
		return resp
	}

	// Suppression: strip model output (Content + OutputTranscription).
	// InputTranscription (user input) is preserved since it is not a
	// model duplicate. With no IT remaining, the message is fully
	// suppressed (return nil).
	stripped := *resp
	stripped.Content = nil
	stripped.OutputTranscription = nil
	if stripped.InputTranscription == nil {
		return nil
	}
	return &stripped
}
