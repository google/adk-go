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

package agent

import (
	"errors"
	"time"

	"google.golang.org/genai"
)

// ErrLLMCallsLimitExceeded is returned when an invocation makes more model
// calls than RunConfig.MaxLLMCalls allows. Detect it with errors.Is.
var ErrLLMCallsLimitExceeded = errors.New("max number of llm calls exceeded")

// StreamingMode defines the streaming mode for agent execution.
type StreamingMode string

const (
	// StreamingModeNone indicates no streaming.
	StreamingModeNone StreamingMode = "none"
	// StreamingModeSSE enables server-sent events streaming, one-way, where
	// LLM response parts are streamed immediately as they are generated.
	StreamingModeSSE StreamingMode = "sse"
)

// GetSessionConfig controls which historical events the runner requests from
// [session.Service.Get] when loading a session for an invocation.
//
// Its fields mirror the event-window filter fields of [session.GetRequest].
// A zero GetSessionConfig (or a nil *GetSessionConfig on [RunConfig] or
// [LiveRunConfig]) applies no filter and loads the full session history.
// GetSessionConfig is a pure value type and is safe for concurrent read use.
type GetSessionConfig struct {
	// NumRecentEvents limits the loaded session to at most the NumRecentEvents
	// most recent events. If zero or negative, no count limit is applied.
	NumRecentEvents int
	// After limits the loaded session to events with Timestamp >= After.
	// If zero (After.IsZero()), no timestamp filter is applied.
	After time.Time
}

// RunConfig controls runtime behavior of an agent.
type RunConfig struct {
	// StreamingMode defines the streaming mode for an agent.
	StreamingMode StreamingMode
	// If true, ADK runner will save each part of the user input that is a blob
	// (e.g., images, files) as an artifact.
	SaveInputBlobsAsArtifacts bool
	// MaxLLMCalls bounds the total number of model calls one invocation may
	// make, across every agent it runs. Exceeding it ends the run with an error
	// wrapping ErrLLMCallsLimitExceeded.
	//
	// Zero, the value a caller gets from agent.RunConfig{}, means the default:
	// 500, or the value of the ADK_MAX_LLM_CALLS environment variable if it is
	// set to a valid integer. A value of ADK_MAX_LLM_CALLS that is not an
	// integer is ignored silently and the default of 500 applies, while 0 or a
	// negative value there disables the limit. A negative MaxLLMCalls, such as
	// -1, means no limit.
	//
	// The limit covers the agents the invocation runs, including those nested
	// through agenttool: each nested run gets its own budget of the same size.
	//
	// This differs from adk-python's RunConfig.max_llm_calls, where 0 means no
	// limit. Here 0 is the default, because it is the zero value of the field,
	// so code ported from Python that sets 0 to disable the limit must use -1.
	//
	// Only calls the agent flow makes to the model are counted. Calls made by
	// the compaction summarizer (see package session/compaction) after the
	// invocation are not.
	//
	// The limit exists because whether a run terminates otherwise depends
	// entirely on model behavior: a model that keeps requesting tool calls
	// appends an event per turn and re-sends a growing history, so token cost
	// grows quadratically with no exit.
	MaxLLMCalls int

	// CustomMetadata holds caller-supplied key-value metadata for this
	// invocation.
	//
	// The runner merges CustomMetadata into every event the invocation
	// persists and yields, including the initial user message event and a
	// BeforeRunCallback early-exit reply, before OnEventCallback plugins run
	// and before the event is appended to the session. When an event already
	// defines a key present in CustomMetadata, the event's own value wins.
	//
	// Keys reserved for ADK's own event bookkeeping — keys starting with
	// "__adk_internal_" or "adk.workflow.", and "adk:before_run_reply" — are
	// dropped from CustomMetadata before merging. Keys with those prefixes
	// that an event itself sets are preserved on that event.
	//
	// CustomMetadata is a pointer so that RunConfig stays comparable and
	// usable as a map key. Nil (the zero value) or a pointer to an empty map
	// leaves every event's CustomMetadata unchanged. The runner copies the
	// non-reserved entries when Run is called, before Run returns, and merges
	// that copy into events: the caller's map is never mutated by ADK, and
	// changes to it after Run is called do not reach the events. The copy is
	// shallow. Values stored in the map should be JSON-serializable for
	// persistent session backends and must not be mutated concurrently while
	// the invocation runs.
	//
	// A plugin that edits the event and returns nil keeps its edits; one that
	// returns an event, the same or a new one, gets the run's metadata merged
	// into it, its own keys winning. When a replacement event leaves
	// CustomMetadata nil, the original event's CustomMetadata is preserved and
	// its keys win over the run's metadata; to clear the original event's
	// public keys on a replacement, set CustomMetadata to an empty map
	// (map[string]any{}).
	CustomMetadata *map[string]any

	// GetSessionConfig controls which events are fetched when the runner
	// loads the session from the session service at the start of Run.
	//
	// When non-nil, the runner copies NumRecentEvents and After into the
	// session.GetRequest it sends to session.Service.Get. Nil or a zero
	// GetSessionConfig leaves both filters unset (0 and time.Time{}), loading
	// all session events. The runner snapshots GetSessionConfig when Run is
	// called, before Run returns, so mutating the struct after calling Run
	// does not affect the invocation.
	GetSessionConfig *GetSessionConfig

	// HTTPOptions holds per-invocation HTTP options (such as custom headers,
	// timeout, retry options, or extra body fields) merged into each model
	// request's GenerateContentConfig.HTTPOptions during the invocation.
	//
	// When the agent has no GenerateContentConfig.HTTPOptions, each request
	// receives a copy of HTTPOptions (including BaseURL and APIVersion). When
	// the agent already configures GenerateContentConfig.HTTPOptions,
	// HTTPOptions.Headers are merged into the agent's headers by canonical
	// HTTP header name (with HTTPOptions winning on conflict), and non-nil
	// Timeout, RetryOptions, and ExtraBody replace the agent's values;
	// configuration-time fields (BaseURL, APIVersion, BaseURLResourceScope,
	// ExtrasRequestProvider) on the agent's HTTPOptions are kept.
	//
	// Nil (the zero value) leaves each request's HTTPOptions as configured on
	// the agent. ADK copies HTTPOptions, its Headers and ExtraBody maps (the
	// ExtraBody map is copied and the values inside it are shared), its
	// RetryOptions struct, and its Timeout pointer before attaching them to a
	// request, so neither the caller's HTTPOptions nor the agent's is mutated
	// by request assembly or by BeforeModelCallback.
	//
	// Run does not copy HTTPOptions. ADK reads the struct HTTPOptions points
	// to, and its nested maps, each time it builds a model request, and it
	// builds them while the sequence returned by Run is iterated. A change
	// made through the pointer after Run returns therefore applies to every
	// request built after the change. Set HTTPOptions before calling Run, and
	// leave the struct and its maps unchanged until the iteration ends. An
	// agent run through tool/agenttool gets the same HTTPOptions.
	//
	// The model connectors in model/anthropicmodel and model/openaimodel
	// accept only the Timeout and Headers fields of a request's HTTPOptions:
	// they apply a positive Timeout as the request timeout and send none of
	// the Headers. For an agent whose model uses one of these connectors, ADK
	// drops the other fields of HTTPOptions before the merge, as adk-python's
	// connectors ignore HTTP options, so an invocation that sets RetryOptions
	// or ExtraBody for its Gemini agents does not fail on that agent. HTTP
	// options set on that agent's own GenerateContentConfig are not dropped:
	// the connector returns an error for a field it cannot send.
	HTTPOptions *genai.HTTPOptions

	// Labels holds per-invocation key-value labels (for example, for billing
	// or attribution) merged into each model request's
	// GenerateContentConfig.Labels during the invocation.
	//
	// When an agent also configures GenerateContentConfig.Labels, entries in
	// Labels are merged on top and win on key conflict. An explicitly set
	// "adk_agent_name" entry in Labels is preserved and not overwritten by
	// ADK's default agent-name attribution label.
	//
	// Labels is a pointer so that RunConfig stays comparable and usable as a
	// map key. Nil (the zero value) or a pointer to an empty map leaves each
	// request's Labels as configured on the agent. ADK copies Labels into a
	// fresh map per request, so neither the caller's map nor the agent's map
	// is mutated.
	//
	// Run does not copy Labels. ADK reads the map each time it builds a model
	// request, and it builds them while the sequence returned by Run is
	// iterated. A change to the map's entries after Run returns therefore
	// applies to every request built after the change. Set Labels before
	// calling Run, and leave the map unchanged until the iteration ends. An
	// agent run through tool/agenttool gets the same Labels.
	//
	// The model connectors in model/anthropicmodel and model/openaimodel have
	// no request labels to send. For an agent whose model uses one of these
	// connectors, ADK drops Labels, as adk-python's connectors ignore labels,
	// so an invocation that sets Labels for its Gemini agents does not fail on
	// that agent. Labels other than "adk_agent_name" set on that agent's own
	// GenerateContentConfig still make the connector return an error.
	Labels *map[string]string
}
