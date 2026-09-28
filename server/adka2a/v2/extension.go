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

package adka2a

import (
	"maps"

	"github.com/a2aproject/a2a-go/v2/a2a"
)

// ADKExtensionURI identifies the ADK A2A integration extension.
//
// [Executor] writes it as a metadata key on every task and task status update
// it emits. The key tells a client that the payload follows ADK conventions:
// content is spread across task artifacts and the status message, with
// long-running function calls carried by the latter.
//
// A client that does not find the key may assume artifacts hold the whole
// response. An ADK Python client does exactly that, and consequently misses the
// long-running function call of an input-required task, synthesizing a
// mock_function_call_for_required_user_input in its place. See
// https://github.com/google/adk-go/issues/913.
//
// The key is metadata rather than an [a2asrv.Extensions.Activate] call because
// activation is reported through the X-A2A-Extensions response header, which an
// ADK Python client does not read. Declaring the extension in an
// [a2a.AgentCard] is not a substitute either: the card describes an agent
// before a request, and clients pick how to read a response from the response
// itself.
const ADKExtensionURI = "https://google.github.io/adk-docs/a2a/a2a-extension/"

// addADKExtensionMeta adds [ADKExtensionURI] to the metadata of events whose
// metadata reaches a2a.Task.Metadata, where clients look for it.
//
// Only tasks and task status updates qualify: a2a merges status update metadata
// into the task, while artifact update metadata is merged into the artifact.
func addADKExtensionMeta(event a2a.Event) {
	switch v := event.(type) {
	case *a2a.Task:
		if v != nil {
			v.Metadata = metaWithADKExtension(v.Metadata)
		}
	case *a2a.TaskStatusUpdateEvent:
		if v != nil {
			v.Metadata = metaWithADKExtension(v.Metadata)
		}
	}
}

// metaWithADKExtension returns a copy of meta holding [ADKExtensionURI].
//
// The copy keeps the key from becoming visible through a metadata map shared
// with other events, such as the per-invocation metadata every artifact update
// clones.
func metaWithADKExtension(meta map[string]any) map[string]any {
	result := make(map[string]any, len(meta)+1)
	maps.Copy(result, meta)
	result[ADKExtensionURI] = true
	return result
}

// withADKExtensionMeta decorates yield to add [ADKExtensionURI] to the metadata
// of every event passing through it. Decorating the sink rather than each event
// source keeps the guarantee in one place, and applies it after user callbacks
// have had a chance to replace event metadata.
func withADKExtensionMeta(yield func(a2a.Event, error) bool) func(a2a.Event, error) bool {
	return func(event a2a.Event, err error) bool {
		addADKExtensionMeta(event)
		return yield(event, err)
	}
}
