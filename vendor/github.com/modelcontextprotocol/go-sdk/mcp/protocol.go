// Copyright 2025 The Go MCP SDK Authors. All rights reserved.
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

package mcp

// Protocol types for version 2025-06-18.
// To see the schema changes from the previous version, run:
//
//   prefix=https://raw.githubusercontent.com/modelcontextprotocol/modelcontextprotocol/refs/heads/main/schema
//   sdiff -l <(curl $prefix/2025-03-26/schema.ts) <(curl $prefix/2025/06-18/schema.ts)

import (
	"encoding/json"
	"fmt"

	"github.com/google/jsonschema-go/jsonschema"
)

// Optional annotations for the client. The client can use annotations to inform
// how objects are used or displayed
type Annotations struct {
	// Describes who the intended customer of this object or data is.
	//
	// It can include multiple entries to indicate content useful for multiple
	// audiences (e.g., []Role{"user", "assistant"}).
	Audience []Role `json:"audience,omitempty"`
	// The moment the resource was last modified, as an ISO 8601 formatted string.
	//
	// Should be an ISO 8601 formatted string (e.g., "2025-01-12T15:00:58Z").
	//
	// Examples: last activity timestamp in an open file, timestamp when the
	// resource was attached, etc.
	LastModified string `json:"lastModified,omitempty"`
	// Describes how important this data is for operating the server.
	//
	// A value of 1 means "most important," and indicates that the data is
	// effectively required, while 0 means "least important," and indicates that the
	// data is entirely optional.
	Priority float64 `json:"priority,omitempty"`
}

// CallToolParams is used by clients to call a tool.
type CallToolParams struct {
	// Meta is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta `json:"_meta,omitempty"`
	// Name is the name of the tool to call.
	Name string `json:"name"`
	// Arguments holds the tool arguments. It can hold any value that can be
	// marshaled to JSON.
	Arguments any `json:"arguments,omitempty"`
}

// CallToolParamsRaw is passed to tool handlers on the server. Its arguments
// are not yet unmarshaled (hence "raw"), so that the handlers can perform
// unmarshaling themselves.
type CallToolParamsRaw struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta `json:"_meta,omitempty"`
	// Name is the name of the tool being called.
	Name string `json:"name"`
	// Arguments is the raw arguments received over the wire from the client. It
	// is the responsibility of the tool handler to unmarshal and validate the
	// Arguments (see [AddTool]).
	Arguments json.RawMessage `json:"arguments,omitempty"`
}

// A CallToolResult is the server's response to a tool call.
//
// The [ToolHandler] and [ToolHandlerFor] handler functions return this result,
// though [ToolHandlerFor] populates much of it automatically as documented at
// each field.
type CallToolResult struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta `json:"_meta,omitempty"`

	// A list of content objects that represent the unstructured result of the tool
	// call.
	//
	// When using a [ToolHandlerFor] with structured output, if Content is unset
	// it will be populated with JSON text content corresponding to the
	// structured output value.
	Content []Content `json:"content"`

	// StructuredContent is an optional value that represents the structured
	// result of the tool call. It must marshal to a JSON object.
	//
	// When using a [ToolHandlerFor] with structured output, you should not
	// populate this field. It will be automatically populated with the typed Out
	// value.
	StructuredContent any `json:"structuredContent,omitempty"`

	// IsError reports whether the tool call ended in an error.
	//
	// If not set, this is assumed to be false (the call was successful).
	//
	// Any errors that originate from the tool should be reported inside the
	// Content field, with IsError set to true, not as an MCP protocol-level
	// error response. Otherwise, the LLM would not be able to see that an error
	// occurred and self-correct.
	//
	// However, any errors in finding the tool, an error indicating that the
	// server does not support tool calls, or any other exceptional conditions,
	// should be reported as an MCP error response.
	//
	// When using a [ToolHandlerFor], this field is automatically set when the
	// tool handler returns an error, and the error string is included as text in
	// the Content field.
	IsError bool `json:"isError,omitempty"`

	// The error passed to setError, if any.
	// It is not marshaled, and therefore it is only visible on the server.
	// Its only use is in server sending middleware, where it can be accessed
	// with getError.
	err error
}

// TODO(#64): consider exposing setError (and getError), by adding an error
// field on CallToolResult.
func (r *CallToolResult) setError(err error) {
	r.Content = []Content{&TextContent{Text: err.Error()}}
	r.IsError = true
	r.err = err
}

// getError returns the error set with setError, or nil if none.
// This function always returns nil on clients.
func (r *CallToolResult) getError() error {
	return r.err
}

func (*CallToolResult) isResult() {}

// UnmarshalJSON handles the unmarshalling of content into the Content
// interface.
func (x *CallToolResult) UnmarshalJSON(data []byte) error {
	type res CallToolResult // avoid recursion
	var wire struct {
		res
		Content []*wireContent `json:"content"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	var err error
	if wire.res.Content, err = contentsFromWire(wire.Content, nil); err != nil {
		return err
	}
	*x = CallToolResult(wire.res)
	return nil
}

func (x *CallToolParams) isParams()              {}
func (x *CallToolParams) GetProgressToken() any  { return getProgressToken(x) }
func (x *CallToolParams) SetProgressToken(t any) { setProgressToken(x, t) }

func (x *CallToolParamsRaw) isParams()              {}
func (x *CallToolParamsRaw) GetProgressToken() any  { return getProgressToken(x) }
func (x *CallToolParamsRaw) SetProgressToken(t any) { setProgressToken(x, t) }

type CancelledParams struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta `json:"_meta,omitempty"`
	// An optional string describing the reason for the cancellation. This may be
	// logged or presented to the user.
	Reason string `json:"reason,omitempty"`
	// The ID of the request to cancel.
	//
	// This must correspond to the ID of a request previously issued in the same
	// direction.
	RequestID any `json:"requestId"`
}

func (x *CancelledParams) isParams()              {}
func (x *CancelledParams) GetProgressToken() any  { return getProgressToken(x) }
func (x *CancelledParams) SetProgressToken(t any) { setProgressToken(x, t) }

// Capabilities a client may support. Known capabilities are defined here, in
// this schema, but this is not a closed set: any client can define its own,
// additional capabilities.
type ClientCapabilities struct {
	// Experimental, non-standard capabilities that the client supports.
	Experimental map[string]any `json:"experimental,omitempty"`
	// Present if the client supports listing roots.
	Roots struct {
		// Whether the client supports notifications for changes to the roots list.
		ListChanged bool `json:"listChanged,omitempty"`
	} `json:"roots,omitempty"`
	// Present if the client supports sampling from an LLM.
	Sampling *SamplingCapabilities `json:"sampling,omitempty"`
	// Present if the client supports elicitation from the server.
	Elicitation *ElicitationCapabilities `json:"elicitation,omitempty"`
}

type CompleteParamsArgument struct {
	// The name of the argument
	Name string `json:"name"`
	// The value of the argument to use for completion matching.
	Value string `json:"value"`
}

// CompleteContext represents additional, optional context for completions.
type CompleteContext struct {
	// Previously-resolved variables in a URI template or prompt.
	Arguments map[string]string `json:"arguments,omitempty"`
}

// CompleteReference represents a completion reference type (ref/prompt ref/resource).
// The Type field determines which other fields are relevant.
type CompleteReference struct {
	Type string `json:"type"`
	// Name is relevant when Type is "ref/prompt".
	Name string `json:"name,omitempty"`
	// URI is relevant when Type is "ref/resource".
	URI string `json:"uri,omitempty"`
}

func (r *CompleteReference) UnmarshalJSON(data []byte) error {
	type wireCompleteReference CompleteReference // for naive unmarshaling
	var r2 wireCompleteReference
	if err := json.Unmarshal(data, &r2); err != nil {
		return err
	}
	switch r2.Type {
	case "ref/prompt", "ref/resource":
		if r2.Type == "ref/prompt" && r2.URI != "" {
			return fmt.Errorf("reference of type %q must not have a URI set", r2.Type)
		}
		if r2.Type == "ref/resource" && r2.Name != "" {
			return fmt.Errorf("reference of type %q must not have a Name set", r2.Type)
		}
	default:
		return fmt.Errorf("unrecognized content type %q", r2.Type)
	}
	*r = CompleteReference(r2)
	return nil
}

func (r *CompleteReference) MarshalJSON() ([]byte, error) {
	// Validation for marshalling: ensure consistency before converting to JSON.
	switch r.Type {
	case "ref/prompt":
		if r.URI != "" {
			return nil, fmt.Errorf("reference of type %q must not have a URI set for marshalling", r.Type)
		}
	case "ref/resource":
		if r.Name != "" {
			return nil, fmt.Errorf("reference of type %q must not have a Name set for marshalling", r.Type)
		}
	default:
		return nil, fmt.Errorf("unrecognized reference type %q for marshalling", r.Type)
	}

	type wireReference CompleteReference
	return json.Marshal(wireReference(*r))
}

type CompleteParams struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta `json:"_meta,omitempty"`
	// The argument's information
	Argument CompleteParamsArgument `json:"argument"`
	Context  *CompleteContext       `json:"context,omitempty"`
	Ref      *CompleteReference     `json:"ref"`
}

func (*CompleteParams) isParams() {}

type CompletionResultDetails struct {
	HasMore bool     `json:"hasMore,omitempty"`
	Total   int      `json:"total,omitempty"`
	Values  []string `json:"values"`
}

// The server's response to a completion/complete request
type CompleteResult struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta       `json:"_meta,omitempty"`
	Completion CompletionResultDetails `json:"completion"`
}

func (*CompleteResult) isResult() {}

type CreateMessageParams struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta `json:"_meta,omitempty"`
	// A request to include context from one or more MCP servers (including the
	// caller), to be attached to the prompt. The client may ignore this request.
	IncludeContext string `json:"includeContext,omitempty"`
	// The maximum number of tokens to sample, as requested by the server. The
	// client may choose to sample fewer tokens than requested.
	MaxTokens int64              `json:"maxTokens"`
	Messages  []*SamplingMessage `json:"messages"`
	// Optional metadata to pass through to the LLM provider. The format of this
	// metadata is provider-specific.
	Metadata any `json:"metadata,omitempty"`
	// The server's preferences for which model to select. The client may ignore
	// these preferences.
	ModelPreferences *ModelPreferences `json:"modelPreferences,omitempty"`
	StopSequences    []string          `json:"stopSequences,omitempty"`
	// An optional system prompt the server wants to use for sampling. The client
	// may modify or omit this prompt.
	SystemPrompt string  `json:"systemPrompt,omitempty"`
	Temperature  float64 `json:"temperature,omitempty"`
}

func (x *CreateMessageParams) isParams()              {}
func (x *CreateMessageParams) GetProgressToken() any  { return getProgressToken(x) }
func (x *CreateMessageParams) SetProgressToken(t any) { setProgressToken(x, t) }

// The client's response to a sampling/create_message request from the server.
// The client should inform the user before returning the sampled message, to
// allow them to inspect the response (human in the loop) and decide whether to
// allow the server to see it.
type CreateMessageResult struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta    `json:"_meta,omitempty"`
	Content Content `json:"content"`
	// The name of the model that generated the message.
	Model string `json:"model"`
	Role  Role   `json:"role"`
	// The reason why sampling stopped, if known.
	StopReason string `json:"stopReason,omitempty"`
}

func (*CreateMessageResult) isResult() {}
func (r *CreateMessageResult) UnmarshalJSON(data []byte) error {
	type result CreateMessageResult // avoid recursion
	var wire struct {
		result
		Content *wireContent `json:"content"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	var err error
	if wire.result.Content, err = contentFromWire(wire.Content, map[string]bool{"text": true, "image": true, "audio": true}); err != nil {
		return err
	}
	*r = CreateMessageResult(wire.result)
	return nil
}

type GetPromptParams struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta `json:"_meta,omitempty"`
	// Arguments to use for templating the prompt.
	Arguments map[string]string `json:"arguments,omitempty"`
	// The name of the prompt or prompt template.
	Name string `json:"name"`
}

func (x *GetPromptParams) isParams()              {}
func (x *GetPromptParams) GetProgressToken() any  { return getProgressToken(x) }
func (x *GetPromptParams) SetProgressToken(t any) { setProgressToken(x, t) }

// The server's response to a prompts/get request from the client.
type GetPromptResult struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta `json:"_meta,omitempty"`
	// An optional description for the prompt.
	Description string           `json:"description,omitempty"`
	Messages    []*PromptMessage `json:"messages"`
}

func (*GetPromptResult) isResult() {}

type InitializeParams struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta         `json:"_meta,omitempty"`
	Capabilities *ClientCapabilities `json:"capabilities"`
	ClientInfo   *Implementation     `json:"clientInfo"`
	// The latest version of the Model Context Protocol that the client supports.
	// The client may decide to support older versions as well.
	ProtocolVersion string `json:"protocolVersion"`
}

func (x *InitializeParams) isParams()              {}
func (x *InitializeParams) GetProgressToken() any  { return getProgressToken(x) }
func (x *InitializeParams) SetProgressToken(t any) { setProgressToken(x, t) }

// After receiving an initialize request from the client, the server sends this
// response.
type InitializeResult struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta         `json:"_meta,omitempty"`
	Capabilities *ServerCapabilities `json:"capabilities"`
	// Instructions describing how to use the server and its features.
	//
	// This can be used by clients to improve the LLM's understanding of available
	// tools, resources, etc. It can be thought of like a "hint" to the model. For
	// example, this information may be added to the system prompt.
	Instructions string `json:"instructions,omitempty"`
	// The version of the Model Context Protocol that the server wants to use. This
	// may not match the version that the client requested. If the client cannot
	// support this version, it must disconnect.
	ProtocolVersion string          `json:"protocolVersion"`
	ServerInfo      *Implementation `json:"serverInfo"`
}

func (*InitializeResult) isResult() {}

type InitializedParams struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta `json:"_meta,omitempty"`
}

func (x *InitializedParams) isParams()              {}
func (x *InitializedParams) GetProgressToken() any  { return getProgressToken(x) }
func (x *InitializedParams) SetProgressToken(t any) { setProgressToken(x, t) }

type ListPromptsParams struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta `json:"_meta,omitempty"`
	// An opaque token representing the current pagination position. If provided,
	// the server should return results starting after this cursor.
	Cursor string `json:"cursor,omitempty"`
}

func (x *ListPromptsParams) isParams()              {}
func (x *ListPromptsParams) GetProgressToken() any  { return getProgressToken(x) }
func (x *ListPromptsParams) SetProgressToken(t any) { setProgressToken(x, t) }
func (x *ListPromptsParams) cursorPtr() *string     { return &x.Cursor }

// The server's response to a prompts/list request from the client.
type ListPromptsResult struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta `json:"_meta,omitempty"`
	// An opaque token representing the pagination position after the last returned
	// result. If present, there may be more results available.
	NextCursor string    `json:"nextCursor,omitempty"`
	Prompts    []*Prompt `json:"prompts"`
}

func (x *ListPromptsResult) isResult()              {}
func (x *ListPromptsResult) nextCursorPtr() *string { return &x.NextCursor }

type ListResourceTemplatesParams struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta `json:"_meta,omitempty"`
	// An opaque token representing the current pagination position. If provided,
	// the server should return results starting after this cursor.
	Cursor string `json:"cursor,omitempty"`
}

func (x *ListResourceTemplatesParams) isParams()              {}
func (x *ListResourceTemplatesParams) GetProgressToken() any  { return getProgressToken(x) }
func (x *ListResourceTemplatesParams) SetProgressToken(t any) { setProgressToken(x, t) }
func (x *ListResourceTemplatesParams) cursorPtr() *string     { return &x.Cursor }

// The server's response to a resources/templates/list request from the client.
type ListResourceTemplatesResult struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta `json:"_meta,omitempty"`
	// An opaque token representing the pagination position after the last returned
	// result. If present, there may be more results available.
	NextCursor        string              `json:"nextCursor,omitempty"`
	ResourceTemplates []*ResourceTemplate `json:"resourceTemplates"`
}

func (x *ListResourceTemplatesResult) isResult()              {}
func (x *ListResourceTemplatesResult) nextCursorPtr() *string { return &x.NextCursor }

type ListResourcesParams struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta `json:"_meta,omitempty"`
	// An opaque token representing the current pagination position. If provided,
	// the server should return results starting after this cursor.
	Cursor string `json:"cursor,omitempty"`
}

func (x *ListResourcesParams) isParams()              {}
func (x *ListResourcesParams) GetProgressToken() any  { return getProgressToken(x) }
func (x *ListResourcesParams) SetProgressToken(t any) { setProgressToken(x, t) }
func (x *ListResourcesParams) cursorPtr() *string     { return &x.Cursor }

// The server's response to a resources/list request from the client.
type ListResourcesResult struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta `json:"_meta,omitempty"`
	// An opaque token representing the pagination position after the last returned
	// result. If present, there may be more results available.
	NextCursor string      `json:"nextCursor,omitempty"`
	Resources  []*Resource `json:"resources"`
}

func (x *ListResourcesResult) isResult()              {}
func (x *ListResourcesResult) nextCursorPtr() *string { return &x.NextCursor }

type ListRootsParams struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta `json:"_meta,omitempty"`
}

func (x *ListRootsParams) isParams()              {}
func (x *ListRootsParams) GetProgressToken() any  { return getProgressToken(x) }
func (x *ListRootsParams) SetProgressToken(t any) { setProgressToken(x, t) }

// The client's response to a roots/list request from the server. This result
// contains an array of Root objects, each representing a root directory or file
// that the server can operate on.
type ListRootsResult struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta  `json:"_meta,omitempty"`
	Roots []*Root `json:"roots"`
}

func (*ListRootsResult) isResult() {}

type ListToolsParams struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta `json:"_meta,omitempty"`
	// An opaque token representing the current pagination position. If provided,
	// the server should return results starting after this cursor.
	Cursor string `json:"cursor,omitempty"`
}

func (x *ListToolsParams) isParams()              {}
func (x *ListToolsParams) GetProgressToken() any  { return getProgressToken(x) }
func (x *ListToolsParams) SetProgressToken(t any) { setProgressToken(x, t) }
func (x *ListToolsParams) cursorPtr() *string     { return &x.Cursor }

// The server's response to a tools/list request from the client.
type ListToolsResult struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta `json:"_meta,omitempty"`
	// An opaque token representing the pagination position after the last returned
	// result. If present, there may be more results available.
	NextCursor string  `json:"nextCursor,omitempty"`
	Tools      []*Tool `json:"tools"`
}

func (x *ListToolsResult) isResult()              {}
func (x *ListToolsResult) nextCursorPtr() *string { return &x.NextCursor }

// The severity of a log message.
//
// These map to syslog message severities, as specified in RFC-5424:
// https://datatracker.ietf.org/doc/html/rfc5424#section-6.2.1
type LoggingLevel string

type LoggingMessageParams struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta `json:"_meta,omitempty"`
	// The data to be logged, such as a string message or an object. Any JSON
	// serializable type is allowed here.
	Data any `json:"data"`
	// The severity of this log message.
	Level LoggingLevel `json:"level"`
	// An optional name of the logger issuing this message.
	Logger string `json:"logger,omitempty"`
}

func (x *LoggingMessageParams) isParams()              {}
func (x *LoggingMessageParams) GetProgressToken() any  { return getProgressToken(x) }
func (x *LoggingMessageParams) SetProgressToken(t any) { setProgressToken(x, t) }

// Hints to use for model selection.
//
// Keys not declared here are currently left unspecified by the spec and are up
// to the client to interpret.
type ModelHint struct {
	// A hint for a model name.
	//
	// The client should treat this as a substring of a model name; for example: -
	// `claude-3-5-sonnet` should match `claude-3-5-sonnet-20241022` - `sonnet`
	// should match `claude-3-5-sonnet-20241022`, `claude-3-sonnet-20240229`, etc. -
	// `claude` should match any Claude model
	//
	// The client may also map the string to a different provider's model name or a
	// different model family, as long as it fills a similar niche; for example: -
	// `gemini-1.5-flash` could match `claude-3-haiku-20240307`
	Name string `json:"name,omitempty"`
}

// The server's preferences for model selection, requested of the client during
// sampling.
//
// Because LLMs can vary along multiple dimensions, choosing the "best" model is
// rarely straightforward. Different models excel in different areas—some are
// faster but less capable, others are more capable but more expensive, and so
// on. This interface allows servers to express their priorities across multiple
// dimensions to help clients make an appropriate selection for their use case.
//
// These preferences are always advisory. The client may ignore them. It is also
// up to the client to decide how to interpret these preferences and how to
// balance them against other considerations.
type ModelPreferences struct {
	// How much to prioritize cost when selecting a model. A value of 0 means cost
	// is not important, while a value of 1 means cost is the most important factor.
	CostPriority float64 `json:"costPriority,omitempty"`
	// Optional hints to use for model selection.
	//
	// If multiple hints are specified, the client must evaluate them in order (such
	// that the first match is taken).
	//
	// The client should prioritize these hints over the numeric priorities, but may
	// still use the priorities to select from ambiguous matches.
	Hints []*ModelHint `json:"hints,omitempty"`
	// How much to prioritize intelligence and capabilities when selecting a model.
	// A value of 0 means intelligence is not important, while a value of 1 means
	// intelligence is the most important factor.
	IntelligencePriority float64 `json:"intelligencePriority,omitempty"`
	// How much to prioritize sampling speed (latency) when selecting a model. A
	// value of 0 means speed is not important, while a value of 1 means speed is
	// the most important factor.
	SpeedPriority float64 `json:"speedPriority,omitempty"`
}

type PingParams struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta `json:"_meta,omitempty"`
}

func (x *PingParams) isParams()              {}
func (x *PingParams) GetProgressToken() any  { return getProgressToken(x) }
func (x *PingParams) SetProgressToken(t any) { setProgressToken(x, t) }

type ProgressNotificationParams struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta `json:"_meta,omitempty"`
	// The progress token which was given in the initial request, used to associate
	// this notification with the request that is proceeding.
	ProgressToken any `json:"progressToken"`
	// An optional message describing the current progress.
	Message string `json:"message,omitempty"`
	// The progress thus far. This should increase every time progress is made, even
	// if the total is unknown.
	Progress float64 `json:"progress"`
	// Total number of items to process (or total progress required), if known.
	// Zero means unknown.
	Total float64 `json:"total,omitempty"`
}

func (*ProgressNotificationParams) isParams() {}

// A prompt or prompt template that the server offers.
type Prompt struct {
	// See [specification/2025-06-18/basic/index#general-fields] for notes on _meta
	// usage.
	Meta `json:"_meta,omitempty"`
	// A list of arguments to use for templating the prompt.
	Arguments []*PromptArgument `json:"arguments,omitempty"`
	// An optional description of what this prompt provides
	Description string `json:"description,omitempty"`
	// Intended for programmatic or logical use, but used as a display name in past
	// specs or fallback (if title isn't present).
	Name string `json:"name"`
	// Intended for UI and end-user contexts — optimized to be human-readable and
	// easily understood, even by those unfamiliar with domain-specific terminology.
	Title string `json:"title,omitempty"`
}

// Describes an argument that a prompt can accept.
type PromptArgument struct {
	// Intended for programmatic or logical use, but used as a display name in past
	// specs or fallback (if title isn't present).
	Name string `json:"name"`
	// Intended for UI and end-user contexts — optimized to be human-readable and
	// easily understood, even by those unfamiliar with domain-specific terminology.
	Title string `json:"title,omitempty"`
	// A human-readable description of the argument.
	Description string `json:"description,omitempty"`
	// Whether this argument must be provided.
	Required bool `json:"required,omitempty"`
}

type PromptListChangedParams struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta `json:"_meta,omitempty"`
}

func (x *PromptListChangedParams) isParams()              {}
func (x *PromptListChangedParams) GetProgressToken() any  { return getProgressToken(x) }
func (x *PromptListChangedParams) SetProgressToken(t any) { setProgressToken(x, t) }

// Describes a message returned as part of a prompt.
//
// This is similar to SamplingMessage, but also supports the embedding of
// resources from the MCP server.
type PromptMessage struct {
	Content Content `json:"content"`
	Role    Role    `json:"role"`
}

// UnmarshalJSON handles the unmarshalling of content into the Content
// interface.
func (m *PromptMessage) UnmarshalJSON(data []byte) error {
	type msg PromptMessage // avoid recursion
	var wire struct {
		msg
		Content *wireContent `json:"content"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	var err error
	if wire.msg.Content, err = contentFromWire(wire.Content, nil); err != nil {
		return err
	}
	*m = PromptMessage(wire.msg)
	return nil
}

type ReadResourceParams struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta `json:"_meta,omitempty"`
	// The URI of the resource to read. The URI can use any protocol; it is up to
	// the server how to interpret it.
	URI string `json:"uri"`
}

func (x *ReadResourceParams) isParams()              {}
func (x *ReadResourceParams) GetProgressToken() any  { return getProgressToken(x) }
func (x *ReadResourceParams) SetProgressToken(t any) { setProgressToken(x, t) }

// The server's response to a resources/read request from the client.
type ReadResourceResult struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta     `json:"_meta,omitempty"`
	Contents []*ResourceContents `json:"contents"`
}

func (*ReadResourceResult) isResult() {}

// A known resource that the server is capable of reading.
type Resource struct {
	// See [specification/2025-06-18/basic/index#general-fields] for notes on _meta
	// usage.
	Meta `json:"_meta,omitempty"`
	// Optional annotations for the client.
	Annotations *Annotations `json:"annotations,omitempty"`
	// A description of what this resource represents.
	//
	// This can be used by clients to improve the LLM's understanding of available
	// resources. It can be thought of like a "hint" to the model.
	Description string `json:"description,omitempty"`
	// The MIME type of this resource, if known.
	MIMEType string `json:"mimeType,omitempty"`
	// Intended for programmatic or logical use, but used as a display name in past
	// specs or fallback (if title isn't present).
	Name string `json:"name"`
	// The size of the raw resource content, in bytes (i.e., before base64 encoding
	// or any tokenization), if known.
	//
	// This can be used by Hosts to display file sizes and estimate context window
	// usage.
	Size int64 `json:"size,omitempty"`
	// Intended for UI and end-user contexts — optimized to be human-readable and
	// easily understood, even by those unfamiliar with domain-specific terminology.
	//
	// If not provided, the name should be used for display (except for Tool, where
	// Annotations.Title should be given precedence over using name, if
	// present).
	Title string `json:"title,omitempty"`
	// The URI of this resource.
	URI string `json:"uri"`
}

type ResourceListChangedParams struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta `json:"_meta,omitempty"`
}

func (x *ResourceListChangedParams) isParams()              {}
func (x *ResourceListChangedParams) GetProgressToken() any  { return getProgressToken(x) }
func (x *ResourceListChangedParams) SetProgressToken(t any) { setProgressToken(x, t) }

// A template description for resources available on the server.
type ResourceTemplate struct {
	// See [specification/2025-06-18/basic/index#general-fields] for notes on _meta
	// usage.
	Meta `json:"_meta,omitempty"`
	// Optional annotations for the client.
	Annotations *Annotations `json:"annotations,omitempty"`
	// A description of what this template is for.
	//
	// This can be used by clients to improve the LLM's understanding of available
	// resources. It can be thought of like a "hint" to the model.
	Description string `json:"description,omitempty"`
	// The MIME type for all resources that match this template. This should only be
	// included if all resources matching this template have the same type.
	MIMEType string `json:"mimeType,omitempty"`
	// Intended for programmatic or logical use, but used as a display name in past
	// specs or fallback (if title isn't present).
	Name string `json:"name"`
	// Intended for UI and end-user contexts — optimized to be human-readable and
	// easily understood, even by those unfamiliar with domain-specific terminology.
	//
	// If not provided, the name should be used for display (except for Tool, where
	// Annotations.Title should be given precedence over using name, if
	// present).
	Title string `json:"title,omitempty"`
	// A URI template (according to RFC 6570) that can be used to construct resource
	// URIs.
	URITemplate string `json:"uriTemplate"`
}

// The sender or recipient of messages and data in a conversation.
type Role string

// Represents a root directory or file that the server can operate on.
type Root struct {
	// See [specification/2025-06-18/basic/index#general-fields] for notes on _meta
	// usage.
	Meta `json:"_meta,omitempty"`
	// An optional name for the root. This can be used to provide a human-readable
	// identifier for the root, which may be useful for display purposes or for
	// referencing the root in other parts of the application.
	Name string `json:"name,omitempty"`
	// The URI identifying the root. This *must* start with file:// for now. This
	// restriction may be relaxed in future versions of the protocol to allow other
	// URI schemes.
	URI string `json:"uri"`
}

type RootsListChangedParams struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta `json:"_meta,omitempty"`
}

func (x *RootsListChangedParams) isParams()              {}
func (x *RootsListChangedParams) GetProgressToken() any  { return getProgressToken(x) }
func (x *RootsListChangedParams) SetProgressToken(t any) { setProgressToken(x, t) }

// SamplingCapabilities describes the capabilities for sampling.
type SamplingCapabilities struct{}

// ElicitationCapabilities describes the capabilities for elicitation.
type ElicitationCapabilities struct{}

// Describes a message issued to or received from an LLM API.
type SamplingMessage struct {
	Content Content `json:"content"`
	Role    Role    `json:"role"`
}

// UnmarshalJSON handles the unmarshalling of content into the Content
// interface.
func (m *SamplingMessage) UnmarshalJSON(data []byte) error {
	type msg SamplingMessage // avoid recursion
	var wire struct {
		msg
		Content *wireContent `json:"content"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	var err error
	if wire.msg.Content, err = contentFromWire(wire.Content, map[string]bool{"text": true, "image": true, "audio": true}); err != nil {
		return err
	}
	*m = SamplingMessage(wire.msg)
	return nil
}

type SetLoggingLevelParams struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta `json:"_meta,omitempty"`
	// The level of logging that the client wants to receive from the server. The
	// server should send all logs at this level and higher (i.e., more severe) to
	// the client as notifications/message.
	Level LoggingLevel `json:"level"`
}

func (x *SetLoggingLevelParams) isParams()              {}
func (x *SetLoggingLevelParams) GetProgressToken() any  { return getProgressToken(x) }
func (x *SetLoggingLevelParams) SetProgressToken(t any) { setProgressToken(x, t) }

// Definition for a tool the client can call.
type Tool struct {
	// See [specification/2025-06-18/basic/index#general-fields] for notes on _meta
	// usage.
	Meta `json:"_meta,omitempty"`
	// Optional additional tool information.
	//
	// Display name precedence order is: title, annotations.title, then name.
	Annotations *ToolAnnotations `json:"annotations,omitempty"`
	// A human-readable description of the tool.
	//
	// This can be used by clients to improve the LLM's understanding of available
	// tools. It can be thought of like a "hint" to the model.
	Description string `json:"description,omitempty"`
	// A JSON Schema object defining the expected parameters for the tool.
	InputSchema *jsonschema.Schema `json:"inputSchema"`
	// Intended for programmatic or logical use, but used as a display name in past
	// specs or fallback (if title isn't present).
	Name string `json:"name"`
	// An optional JSON Schema object defining the structure of the tool's output
	// returned in the structuredContent field of a CallToolResult.
	OutputSchema *jsonschema.Schema `json:"outputSchema,omitempty"`
	// Intended for UI and end-user contexts — optimized to be human-readable and
	// easily understood, even by those unfamiliar with domain-specific terminology.
	// If not provided, Annotations.Title should be used for display if present,
	// otherwise Name.
	Title string `json:"title,omitempty"`
}

// Additional properties describing a Tool to clients.
//
// NOTE: all properties in ToolAnnotations are hints. They are not
// guaranteed to provide a faithful description of tool behavior (including
// descriptive properties like title).
//
// Clients should never make tool use decisions based on ToolAnnotations
// received from untrusted servers.
type ToolAnnotations struct {
	// If true, the tool may perform destructive updates to its environment. If
	// false, the tool performs only additive updates.
	//
	// (This property is meaningful only when ReadOnlyHint == false.)
	//
	// Default: true
	DestructiveHint *bool `json:"destructiveHint,omitempty"`
	// If true, calling the tool repeatedly with the same arguments will have no
	// additional effect on the its environment.
	//
	// (This property is meaningful only when ReadOnlyHint == false.)
	//
	// Default: false
	IdempotentHint bool `json:"idempotentHint,omitempty"`
	// If true, this tool may interact with an "open world" of external entities. If
	// false, the tool's domain of interaction is closed. For example, the world of
	// a web search tool is open, whereas that of a memory tool is not.
	//
	// Default: true
	OpenWorldHint *bool `json:"openWorldHint,omitempty"`
	// If true, the tool does not modify its environment.
	//
	// Default: false
	ReadOnlyHint bool `json:"readOnlyHint,omitempty"`
	// A human-readable title for the tool.
	Title string `json:"title,omitempty"`
}

type ToolListChangedParams struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta `json:"_meta,omitempty"`
}

func (x *ToolListChangedParams) isParams()              {}
func (x *ToolListChangedParams) GetProgressToken() any  { return getProgressToken(x) }
func (x *ToolListChangedParams) SetProgressToken(t any) { setProgressToken(x, t) }

// Sent from the client to request resources/updated notifications from the
// server whenever a particular resource changes.
type SubscribeParams struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta `json:"_meta,omitempty"`
	// The URI of the resource to subscribe to.
	URI string `json:"uri"`
}

func (*SubscribeParams) isParams() {}

// Sent from the client to request cancellation of resources/updated
// notifications from the server. This should follow a previous
// resources/subscribe request.
type UnsubscribeParams struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta `json:"_meta,omitempty"`
	// The URI of the resource to unsubscribe from.
	URI string `json:"uri"`
}

func (*UnsubscribeParams) isParams() {}

// A notification from the server to the client, informing it that a resource
// has changed and may need to be read again. This should only be sent if the
// client previously sent a resources/subscribe request.
type ResourceUpdatedNotificationParams struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta `json:"_meta,omitempty"`
	// The URI of the resource that has been updated. This might be a sub-resource of the one that the client actually subscribed to.
	URI string `json:"uri"`
}

func (*ResourceUpdatedNotificationParams) isParams() {}

// TODO(jba): add CompleteRequest and related types.

// A request from the server to elicit additional information from the user via the client.
type ElicitParams struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta `json:"_meta,omitempty"`
	// The message to present to the user.
	Message string `json:"message"`
	// A restricted subset of JSON Schema.
	// Only top-level properties are allowed, without nesting.
	RequestedSchema *jsonschema.Schema `json:"requestedSchema"`
}

func (x *ElicitParams) isParams() {}

func (x *ElicitParams) GetProgressToken() any  { return getProgressToken(x) }
func (x *ElicitParams) SetProgressToken(t any) { setProgressToken(x, t) }

// The client's response to an elicitation/create request from the server.
type ElicitResult struct {
	// This property is reserved by the protocol to allow clients and servers to
	// attach additional metadata to their responses.
	Meta `json:"_meta,omitempty"`
	// The user action in response to the elicitation.
	// - "accept": User submitted the form/confirmed the action
	// - "decline": User explicitly declined the action
	// - "cancel": User dismissed without making an explicit choice
	Action string `json:"action"`
	// The submitted form data, only present when action is "accept".
	// Contains values matching the requested schema.
	Content map[string]any `json:"content,omitempty"`
}

func (*ElicitResult) isResult() {}

// An Implementation describes the name and version of an MCP implementation, with an optional
// title for UI representation.
type Implementation struct {
	// Intended for programmatic or logical use, but used as a display name in past
	// specs or fallback (if title isn't present).
	Name string `json:"name"`
	// Intended for UI and end-user contexts — optimized to be human-readable and
	// easily understood, even by those unfamiliar with domain-specific terminology.
	Title   string `json:"title,omitempty"`
	Version string `json:"version"`
}

// Present if the server supports argument autocompletion suggestions.
type CompletionCapabilities struct{}

// Present if the server supports sending log messages to the client.
type LoggingCapabilities struct{}

// Present if the server offers any prompt templates.
type PromptCapabilities struct {
	// Whether this server supports notifications for changes to the prompt list.
	ListChanged bool `json:"listChanged,omitempty"`
}

// Present if the server offers any resources to read.
type ResourceCapabilities struct {
	// Whether this server supports notifications for changes to the resource list.
	ListChanged bool `json:"listChanged,omitempty"`
	// Whether this server supports subscribing to resource updates.
	Subscribe bool `json:"subscribe,omitempty"`
}

// Capabilities that a server may support. Known capabilities are defined here,
// in this schema, but this is not a closed set: any server can define its own,
// additional capabilities.
type ServerCapabilities struct {
	// Present if the server supports argument autocompletion suggestions.
	Completions *CompletionCapabilities `json:"completions,omitempty"`
	// Experimental, non-standard capabilities that the server supports.
	Experimental map[string]any `json:"experimental,omitempty"`
	// Present if the server supports sending log messages to the client.
	Logging *LoggingCapabilities `json:"logging,omitempty"`
	// Present if the server offers any prompt templates.
	Prompts *PromptCapabilities `json:"prompts,omitempty"`
	// Present if the server offers any resources to read.
	Resources *ResourceCapabilities `json:"resources,omitempty"`
	// Present if the server offers any tools to call.
	Tools *ToolCapabilities `json:"tools,omitempty"`
}

// Present if the server offers any tools to call.
type ToolCapabilities struct {
	// Whether this server supports notifications for changes to the tool list.
	ListChanged bool `json:"listChanged,omitempty"`
}

const (
	methodCallTool                  = "tools/call"
	notificationCancelled           = "notifications/cancelled"
	methodComplete                  = "completion/complete"
	methodCreateMessage             = "sampling/createMessage"
	methodElicit                    = "elicitation/create"
	methodGetPrompt                 = "prompts/get"
	methodInitialize                = "initialize"
	notificationInitialized         = "notifications/initialized"
	methodListPrompts               = "prompts/list"
	methodListResourceTemplates     = "resources/templates/list"
	methodListResources             = "resources/list"
	methodListRoots                 = "roots/list"
	methodListTools                 = "tools/list"
	notificationLoggingMessage      = "notifications/message"
	methodPing                      = "ping"
	notificationProgress            = "notifications/progress"
	notificationPromptListChanged   = "notifications/prompts/list_changed"
	methodReadResource              = "resources/read"
	notificationResourceListChanged = "notifications/resources/list_changed"
	notificationResourceUpdated     = "notifications/resources/updated"
	notificationRootsListChanged    = "notifications/roots/list_changed"
	methodSetLevel                  = "logging/setLevel"
	methodSubscribe                 = "resources/subscribe"
	notificationToolListChanged     = "notifications/tools/list_changed"
	methodUnsubscribe               = "resources/unsubscribe"
)
