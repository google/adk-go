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
	"slices"
	"strings"

	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/toolauth"
	"google.golang.org/adk/v2/workflow"
)

// indexOpenInterrupts returns the interrupts raised in events that no later
// event of events answers, keyed by interrupt ID. Each maps to the authors
// that may own it, innermost first: the author of the event that raised it,
// then the authors of the calls whose IDs are run IDs of that event's branch,
// from its last segment to its first, as they were known when the event was
// raised. No author appears twice and none is empty.
//
// An event raises the interrupts whose IDs are in its LongRunningToolIDs, and
// those of its calls to adk_request_input or adk_request_credential (sessions
// stored before the long-running IDs were recorded carry only the call). A
// user event answers an interrupt when one of its function responses has the
// interrupt's ID, or when it has function responses and the interrupt's ID
// is a run ID of its branch.
//
// It ports adk-python's index_open_interrupts (events/_interrupts.py).
func indexOpenInterrupts(events []*session.Event) map[string][]string {
	index := map[string][]string{}
	callAuthorByID := map[string]string{}
	for _, ev := range events {
		runIDs := branchRunIDs(ev.Branch)
		if ev.Author != "" && ev.Author != "user" {
			for _, fc := range utils.FunctionCalls(ev.Content) {
				if fc.ID != "" {
					callAuthorByID[fc.ID] = ev.Author
				}
			}
		}
		if responses := utils.FunctionResponses(ev.Content); ev.Author == "user" && len(responses) > 0 {
			for _, fr := range responses {
				delete(index, fr.ID)
			}
			for _, id := range runIDs {
				delete(index, id)
			}
		}

		ids := interruptIDs(ev)
		if len(ids) == 0 {
			continue
		}
		var authors []string
		add := func(author string) {
			if author != "" && !slices.Contains(authors, author) {
				authors = append(authors, author)
			}
		}
		add(ev.Author)
		for _, id := range slices.Backward(runIDs) {
			add(callAuthorByID[id])
		}
		for _, id := range ids {
			index[id] = authors
		}
	}
	return index
}

// interruptIDs returns the IDs of the interrupts ev raises: its
// LongRunningToolIDs, and the IDs of its calls to adk_request_input or
// adk_request_credential (adk-python extract_event_interrupt_ids,
// events/_interrupts.py).
func interruptIDs(ev *session.Event) []string {
	ids := slices.Clone(ev.LongRunningToolIDs)
	for _, fc := range utils.FunctionCalls(ev.Content) {
		if fc.ID == "" {
			continue
		}
		if fc.Name == workflow.WorkflowInputFunctionCallName || fc.Name == toolauth.RequestCredentialFunctionCallName {
			ids = append(ids, fc.ID)
		}
	}
	return ids
}

// branchRunIDs returns the run IDs of branch in segment order. A branch is a
// dot-joined path of segments; a segment's run ID is what follows its last
// "@", and a segment with none, or with nothing after it, has no run ID
// (adk-python _BranchPath.ordered_run_ids, events/_branch_path.py).
func branchRunIDs(branch string) []string {
	if branch == "" {
		return nil
	}
	var ids []string
	for segment := range strings.SplitSeq(branch, ".") {
		if i := strings.LastIndexByte(segment, '@'); i >= 0 && i < len(segment)-1 {
			ids = append(ids, segment[i+1:])
		}
	}
	return ids
}
