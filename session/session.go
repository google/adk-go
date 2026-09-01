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

// Package session provides types to manage user sessions and their states.
package session

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"math"
	"slices"
	"time"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

// Session represents a series of interactions between a user and agents.
//
// When a user starts interacting with your agent, session holds everything
// related to that one specific chat thread.
type Session interface {
	// ID returns the unique identifier of the session.
	ID() string
	// AppName returns name of the app.
	AppName() string
	// UserID returns the id of the user.
	UserID() string

	// State returns the state of the session.
	State() State
	// Events return the events of the session, e.g. user input, model response, function call/response, etc.
	Events() Events
	// LastUpdateTime returns the time of the last update.
	LastUpdateTime() time.Time
}

// State defines a standard interface for a key-value store.
// It provides basic methods for accessing, modifying, and iterating over
// key-value pairs.
type State interface {
	// Get retrieves the value associated with a given key.
	// It returns a ErrStateKeyNotExist error if the key does not exist.
	Get(string) (any, error)
	// Set assigns the given value to the given key, overwriting any
	// existing value. It returns an error if the underlying storage
	// operation fails.
	Set(string, any) error
	// All returns an iterator (iter.Seq2) that yields all key-value pairs
	// currently in the state. The order of iteration is not guaranteed.
	All() iter.Seq2[string, any]
}

// ReadonlyState defines a standard interface for a key-value store.
// It provides basic methods for accessing, and iterating over
// key-value pairs.
type ReadonlyState interface {
	// Get retrieves the value associated with a given key.
	// It returns a ErrStateKeyNotExist error if the key does not exist.
	Get(string) (any, error)
	// All returns an iterator (iter.Seq2) that yields all key-value pairs
	// currently in the state. The order of iteration is not guaranteed.
	All() iter.Seq2[string, any]
}

// Events define a standard interface for an [Event] list.
// It provides methods for iterating over the sequence and accessing
// individual events by their index.
type Events interface {
	// All returns an iterator (iter.Seq) that yields all events
	// in the sequence, preserving their order.
	All() iter.Seq[*Event]
	// Len returns the total number of events in the sequence.
	Len() int
	// At returns the event at the specified index i.
	At(i int) *Event
}

// Event represents an interaction in a conversation between agents and users.
// It is used to store the content of the conversation, as well as
// the actions taken by the agents like function calls, etc.
type Event struct {
	model.LLMResponse

	// Set by storage
	ID        string    `json:"id"`
	Timestamp time.Time `json:"timestamp"`

	// Set by agent.Context implementation.
	InvocationID string `json:"invocationId"`
	// The branch of the event.
	//
	// The format is like agent_1.agent_2.agent_3, where agent_1 is
	// the parent of agent_2, and agent_2 is the parent of agent_3.
	//
	// Branch is used when multiple sub-agent shouldn't see their peer agents'
	// conversation history.
	Branch string `json:"branch,omitempty"`
	// IsolationScope, when set, restricts which agent contexts include this
	// event in LLM prompt history: an event is visible only when
	// event.IsolationScope equals the agent's isolation scope (exact match;
	// empty sees only empty). Empty for non-scoped events.
	IsolationScope string `json:"isolationScope,omitempty"`
	// Author is the name of the event's author
	Author string `json:"author"`

	// The actions taken by the agent.
	Actions EventActions `json:"actions"`
	// Set of IDs of the long running function calls.
	// Agent client will know from this field about which function call is long running.
	// Only valid for function call event.
	LongRunningToolIDs []string `json:"longRunningToolIds,omitempty"`
	// Routing information for workflow execution
	Routes []string `json:"routes,omitempty"`
	// RequestedInput, when non-nil, signals that the workflow node
	// emitting this event is asking for human input and is about to
	// pause. The workflow scheduler observes this field at event
	// dispatch time and transitions the corresponding node to
	// NodeWaiting on the activation's completion. UI surfaces read
	// the same field to render the prompt.
	//
	// At most one event per node activation may carry this field.
	RequestedInput *RequestInput `json:"requestedInput,omitempty"`

	// Output is the generic data output from a workflow node.
	Output any `json:"output,omitempty"`

	// NodeInfo carries workflow-node metadata for events emitted
	// inside a workflow. Nil for non-workflow events.
	NodeInfo *NodeInfo `json:"nodeInfo,omitempty"`
}

// NodeInfo carries the per-event metadata identifying which node in
// a workflow activation emitted it.
type NodeInfo struct {
	// Path is the composite path of the emitting node within its
	// workflow activation. Empty for top-level static nodes;
	// "<parent_path>/<child_name>@<run_id>" for dynamic children.
	// The scheduler uses it to scope per-activation Output/Routes
	// invariants to the emitter, allowing dynamic nodes to forward
	// children's terminal events alongside their own.
	Path string `json:"path,omitempty"`

	// MessageAsOutput marks that this event's content IS the node's
	// output: when set and Event.Output is nil, readers derive the
	// node output from the event's model text. Mirrors adk-python's
	// node_info.message_as_output.
	MessageAsOutput bool `json:"messageAsOutput,omitempty"`

	// OutputFor lists the node paths this event's Output counts for: the
	// emitter plus any WithUseAsOutput delegating ancestors, so one event
	// stands in for a whole delegation chain rather than each level
	// re-emitting a duplicate. Mirrors adk-python's node_info.output_for.
	OutputFor []string `json:"outputFor,omitempty"`
}

// RequestInput describes a single human-in-the-loop prompt emitted
// by a workflow node. It travels on Event.RequestedInput from the
// node, through the scheduler, out to the UI surface; the matching
// response is routed back by InterruptID.
//
// JSON-marshallable: persisted in session.State across pause/resume
// turns. Payload is typed any and must be JSON-encodable; for
// binary data, stash the bytes via agent.Artifacts and put a URI
// string in Payload.
type RequestInput struct {
	// InterruptID correlates this request with the response that
	// resumes it; the reply is routed back by matching this ID.
	// Prefer a value that is unique per request: leave it empty and
	// the engine fills in a fresh UUID (the recommended default), or
	// build your own from a readable prefix plus a UUID
	// (e.g. "manager_approval-"+uuid).
	//
	// Avoid reusing one fixed literal across separate runs in the same
	// session. ADK clients — notably the Dev UI — track answered
	// requests by this ID and will not re-prompt for an ID already
	// resolved earlier in the session, so a later run that reuses it
	// shows no input box. Mirrors adk-python RequestInput.interrupt_id,
	// which defaults to a fresh UUID.
	InterruptID string `json:"interruptId"`

	// Message is the human-readable description of what is being
	// asked. Surfaced in UI as the prompt text. Optional.
	Message string `json:"message,omitempty"`

	// ResponseSchema, when non-nil, is the JSON schema the user's
	// response payload must conform to.
	ResponseSchema *jsonschema.Schema `json:"responseSchema,omitempty"`

	// Payload is optional context the UI may render alongside the
	// prompt (e.g. the document to approve, the proposed parameters).
	// Carried through opaquely; the engine does not interpret it.
	Payload any `json:"payload,omitempty"`
}

// IsFinalResponse returns whether the event is the final response of an agent.
//
// Note: when multiple agents participate in one invocation, there could be
// multiple events with IsFinalResponse() as True, for each participating agent.
func (e *Event) IsFinalResponse() bool {
	// A compaction event is bookkeeping rather than conversation: it carries a
	// record and no content of its own. It satisfies every clause below, so
	// without this a streaming client deciding what to show a user would surface
	// an empty final response every time compaction ran.
	if e.Actions.Compaction != nil {
		return false
	}
	if e.Actions.SkipSummarization || len(e.LongRunningToolIDs) > 0 {
		return true
	}

	return !hasFunctionCalls(&e.LLMResponse) && !hasFunctionResponses(&e.LLMResponse) && !e.LLMResponse.Partial && !hasTrailingCodeExecutionResult(&e.LLMResponse)
}

// NewEvent creates a new event defining now as the timestamp.
//
// The event ID and timestamp are obtained through the platform package, so a
// time or UUID provider installed on ctx (see [platform.WithTimeProvider] and
// [platform.WithUUIDProvider]) controls them. This lets callers such as
// workflow engines produce deterministic, replay-safe events.
func NewEvent(ctx context.Context, invocationID string) *Event {
	return &Event{
		ID:           platform.NewUUID(ctx),
		InvocationID: invocationID,
		Timestamp:    platform.Now(ctx),
		Actions:      EventActions{StateDelta: make(map[string]any), ArtifactDelta: make(map[string]int64)},
	}
}

// EventActions represent the actions attached to an event.
type EventActions struct {
	// Set by agent.Context implementation.
	StateDelta map[string]any `json:"stateDelta"`

	// Indicates that the event is updating an artifact. key is the filename,
	// value is the version.
	ArtifactDelta map[string]int64 `json:"artifactDelta"`

	RequestedToolConfirmations map[string]toolconfirmation.ToolConfirmation `json:"requestedToolConfirmations,omitempty"`

	// If true, it won't call model to summarize function response.
	// Only valid for function response event.
	SkipSummarization bool `json:"skipSummarization,omitempty"`
	// If set, the event transfers to the specified agent.
	TransferToAgent string `json:"transferToAgent,omitempty"`
	// The agent is escalating to a higher level agent.
	Escalate bool `json:"escalate,omitempty"`

	// Compaction, when non-nil, marks this event as a context-compaction
	// summary standing in for a contiguous range of earlier events.
	//
	// The framework writes this field and prompt assembly reads it. Setting it
	// from a tool handler or a callback has no effect: it is cleared wherever
	// caller-supplied actions are copied onto an event. A record is not a
	// request but an instruction to drop the events it names and show its own
	// content in their place, which is not a decision the code running inside a
	// turn gets to make about the conversation it is running in.
	Compaction *EventCompaction `json:"compaction,omitempty"`
}

// MarshalJSON omits StateDelta and ArtifactDelta when they are nil and writes
// an empty object when they are allocated but empty.
//
// The two cases have to stay distinguishable, and neither `omitempty` nor a
// plain tag manages it. A map tagged `omitempty` is dropped whenever its
// length is zero, which loses the difference: NewEvent allocates both maps and
// callers write into them without a nil check, so an allocated-but-empty map
// that decodes back as nil panics on the first write. Leaving the tag bare
// keeps the difference but encodes a nil map as `null`, which adk-python
// rejects -- both fields are non-Optional dicts there, so the whole
// EventActions fails to validate ("Input should be an object", type=dict_type)
// and takes the event with it.
//
// Omitting the key is the third option and the only one that satisfies both:
// adk-python fills an absent key from the field's default factory, and Go
// decodes it back to nil, so nil stays nil and empty stays empty in either
// runtime. The shadowing pointer fields below express that -- `omitempty` on a
// pointer keys off the pointer being nil, not the length of what it points to.
func (a EventActions) MarshalJSON() ([]byte, error) {
	// alias strips this method, so json does not recurse into it. The two
	// pointer fields sit at depth 0 and shadow the promoted ones at depth 1,
	// which is how encoding/json resolves a name collision; embedding rather
	// than restating the struct keeps fields added upstream from being
	// silently dropped here.
	type alias EventActions
	return json.Marshal(struct {
		alias
		StateDelta    *map[string]any   `json:"stateDelta,omitempty"`
		ArtifactDelta *map[string]int64 `json:"artifactDelta,omitempty"`
	}{
		alias:         alias(a),
		StateDelta:    nilOrRef(a.StateDelta),
		ArtifactDelta: nilOrRef(a.ArtifactDelta),
	})
}

// nilOrRef returns nil for a nil map and a pointer to m otherwise, turning
// "is this map allocated" into something `omitempty` can act on.
func nilOrRef[M ~map[K]V, K comparable, V any](m M) *M {
	if m == nil {
		return nil
	}
	return &m
}

// EventCompaction records that a contiguous range of session [Event]s has been
// replaced by a single piece of CompactedContent, typically a model-generated
// summary.
//
// An EventCompaction is attached to a new [Event] through
// [EventActions.Compaction]; the events it covers are left untouched in the
// session. When the next LLM prompt is built, the contents processor uses the
// range to skip the covered events and inserts CompactedContent in their place.
//
// Both bounds are inclusive, so an event whose timestamp ties EndTimestamp
// counts as covered. Producers must therefore keep EndTimestamp strictly below
// the timestamp of the oldest event they intend to leave un-compacted.
type EventCompaction struct {
	// StartTimestamp is the timestamp of the earliest covered event (inclusive).
	StartTimestamp time.Time `json:"startTimestamp"`
	// EndTimestamp is the timestamp of the latest covered event (inclusive).
	// It is never before StartTimestamp.
	EndTimestamp time.Time `json:"endTimestamp"`
	// CompactedContent is the content that replaces the covered events in the
	// prompt.
	CompactedContent *genai.Content `json:"compactedContent"`

	// ExcludedEvents are the events inside the range above that this summary
	// does NOT stand in for. Everything else in the range is covered.
	//
	// The range alone was not enough. Choosing a window filters events out of
	// the middle of its own span, by branch, by isolation scope and by what a
	// retained tail holds back, so an interval covering the ends also covered
	// the gaps. An event in a gap was dropped from every later prompt having
	// been summarized by nothing, and its content was simply lost.
	//
	// Recording the holes rather than the membership keeps this bounded. Holes
	// are rare, none at all in a single-agent conversation, so this is normally
	// empty where a membership list would carry one entry per event of the
	// conversation, for ever, recopied onto every rolling summary.
	//
	// It is keyed on the invocation and timestamp rather than the event ID
	// because event IDs do not survive every storage backend: the Vertex AI
	// service replaces them with a server resource name on read.
	//
	// Imprecision here is not symmetric. A key matching too much leaves an
	// extra event raw beside a summary of it, which is visible and recoverable.
	// A key that fails to match does not fall back to anything: coverage is the
	// range minus the exclusions, so the event it was protecting becomes
	// covered by a summary that never described it, and is dropped from every
	// later prompt. Producers must therefore err towards naming a hole too
	// broadly, and a backend that does not round-trip these timestamps exactly
	// will silently delete conversation.
	ExcludedEvents []EventRef `json:"excludedEvents,omitempty"`
}

// clone returns a deep copy, or nil for a nil receiver.
//
// A stored compaction decides which events every future prompt drops, so of all
// the fields on [EventActions] it is the one a producer must not be able to
// edit after the append. Sharing the pointer let a caller move EndTimestamp
// afterwards and silently change what history the agent sees, and tripped the
// race detector on the way.
func (c *EventCompaction) clone() *EventCompaction {
	if c == nil {
		return nil
	}
	out := *c
	out.ExcludedEvents = slices.Clone(c.ExcludedEvents)
	if c.CompactedContent != nil {
		content := *c.CompactedContent
		content.Parts = slices.Clone(c.CompactedContent.Parts)
		for i, p := range content.Parts {
			if p == nil {
				continue
			}
			part := *p
			content.Parts[i] = &part
		}
		out.CompactedContent = &content
	}
	return &out
}

// EventRef identifies a stored event by fields that survive a storage round
// trip, for a record that has to refer to an event it does not contain.
//
// Not the event ID, which one backend reassigns on read. The pair is not
// guaranteed unique: two events of one invocation can share a timestamp, and a
// reference then names both. Callers must therefore only use this where naming
// one event too many is the harmless direction.
type EventRef struct {
	InvocationID string    `json:"invocationId"`
	Timestamp    time.Time `json:"timestamp"`
}

// Prefixes for defining session's state scopes
const (
	// KeyPrefixApp is the prefix for app-level state keys.
	// They are shared across all users and sessions for that application.
	KeyPrefixApp string = "app:"
	// KeyPrefixTemp is the prefix for temporary state keys.
	// Such entries are specific to the current invocation (the entire process
	// from an agent receiving user input to generating the final output for
	// that input. Discarded after the invocation completes.
	KeyPrefixTemp string = "temp:"
	// KeyPrefixUser is the prefix for user-level state keys.
	// They are tied to the user_id, shared across all sessions for that user
	// (within the same app_name).
	KeyPrefixUser string = "user:"
)

// ErrStateKeyNotExist is the error thrown when key does not exist.
var ErrStateKeyNotExist = errors.New("state key does not exist")

func hasFunctionCalls(resp *model.LLMResponse) bool {
	if resp == nil || resp.Content == nil {
		return false
	}
	for _, part := range resp.Content.Parts {
		if part.FunctionCall != nil {
			return true
		}
	}
	return false
}

func hasFunctionResponses(resp *model.LLMResponse) bool {
	if resp == nil || resp.Content == nil {
		return false
	}
	for _, part := range resp.Content.Parts {
		if part.FunctionResponse != nil {
			return true
		}
	}
	return false
}

// Returns whether the event has a trailing code execution result.
func hasTrailingCodeExecutionResult(resp *model.LLMResponse) bool {
	if resp == nil || resp.Content == nil || len(resp.Content.Parts) == 0 {
		return false
	}
	lastPart := resp.Content.Parts[len(resp.Content.Parts)-1]
	return lastPart.CodeExecutionResult != nil
}

// Bounds on a decoded numeric event timestamp, in microseconds: years 1 and
// 9999. Wide enough for any real event time, narrow enough that a value in the
// wrong unit falls outside.
const (
	minTimestampMicros = -62135596800 * 1e6
	maxTimestampMicros = 253402300799 * 1e6
)

// UnmarshalJSON decodes an Event, accepting the timestamp either as an RFC
// 3339 string (this package's own encoding) or as a JSON number of epoch
// seconds. The numeric form is what adk-python emits, since its Event
// timestamp is a float; without this an Event serialized by another ADK
// runtime fails to decode part-way through, leaving the fields after the
// timestamp unset.
//
// Numeric timestamps are resolved to microseconds, matching the resolution of
// Python's datetime. float64 cannot represent epoch seconds to nanosecond
// precision, so a finer value would only encode rounding noise.
func (e *Event) UnmarshalJSON(b []byte) error {
	if err := e.unmarshalJSON(b); err != nil {
		return err
	}
	// adk-python's node_info field is not Optional, and it dumps with
	// exclude_none, so every event it writes carries a nodeInfo object even
	// for non-workflow events. Decoding that yields a non-nil pointer to an
	// empty NodeInfo, which contradicts the documented invariant above that a
	// nil NodeInfo means the event did not come from a workflow -- readers
	// test the pointer, not its contents. An all-zero NodeInfo carries no
	// information, so it is indistinguishable from absent.
	if ni := e.NodeInfo; ni != nil && ni.Path == "" && !ni.MessageAsOutput && len(ni.OutputFor) == 0 {
		e.NodeInfo = nil
	}
	return nil
}

func (e *Event) unmarshalJSON(b []byte) error {
	type alias Event // strips methods, so json does not recurse into this one

	// Fast path: a timestamp this package wrote decodes directly, so reading
	// back our own events does not pay for the retry below. An event sourced
	// from adk-python carries a numeric timestamp and always takes the retry,
	// so this is a fast path for one writer, not for every read.
	before := *e
	if err := json.Unmarshal(b, (*alias)(e)); err == nil {
		return nil
	}

	// Something did not decode. Retry with the timestamp held back, since that
	// is the field that legitimately differs between ADK runtimes. Restore the
	// original value first: the failed attempt may have partly overwritten it,
	// and decoding into an already-populated Event leaves absent fields alone.
	*e = before
	aux := struct {
		Timestamp json.RawMessage `json:"timestamp"`
		*alias
	}{alias: (*alias)(e)}
	if err := json.Unmarshal(b, &aux); err != nil {
		return err
	}
	if len(aux.Timestamp) == 0 || string(aux.Timestamp) == "null" {
		return nil
	}
	if aux.Timestamp[0] == '"' {
		if err := e.Timestamp.UnmarshalJSON(aux.Timestamp); err != nil {
			return fmt.Errorf("session: decoding event timestamp %s: %w", aux.Timestamp, err)
		}
		return nil
	}
	var secs float64
	if err := json.Unmarshal(aux.Timestamp, &secs); err != nil {
		return fmt.Errorf("session: decoding event timestamp %s: %w", aux.Timestamp, err)
	}
	micros := math.Round(secs * 1e6)
	// Reject anything outside the calendar. Two things land here: a float64 too
	// large for an int64, whose conversion is implementation-defined and
	// saturates silently while losing the sign; and a value in the wrong unit,
	// where epoch milliseconds read as seconds gives a decodable, wrong-looking
	// instant tens of thousands of years out. Both are better as errors than as
	// an event that sorts into the far future.
	if math.IsNaN(micros) || micros < minTimestampMicros || micros > maxTimestampMicros {
		return fmt.Errorf("session: event timestamp %s is not a plausible date", aux.Timestamp)
	}
	e.Timestamp = time.UnixMicro(int64(micros)).UTC()
	return nil
}
