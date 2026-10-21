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

package agentanalytics

import (
	"cmp"
	"context"
	"encoding/json"
	"maps"
	"reflect"
	"strings"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/toolconfirmation"
	"google.golang.org/adk/v2/workflow"
)

// Tool origin values recorded in TOOL_STARTING, TOOL_COMPLETED, and
// TOOL_ERROR payloads. The vocabulary matches adk-python so cross-language
// queries over tool_origin return consistent values.
const (
	toolOriginLocal         = "LOCAL"
	toolOriginMCP           = "MCP"
	toolOriginTransferAgent = "TRANSFER_AGENT"
	toolOriginTransferA2A   = "TRANSFER_A2A"
	toolOriginSubAgent      = "SUB_AGENT"
	toolOriginA2A           = "A2A"
	toolOriginToolbox       = "TOOLBOX"
	toolOriginSkill         = "SKILL"
	toolOriginUnknown       = "UNKNOWN"
)

const (
	hitlCredentialFunctionCallName = "adk_request_credential"
)

type hitlToolMapping struct {
	requestEvent   string
	completedEvent string
	pauseKind      string
}

var hitlTools = map[string]hitlToolMapping{
	hitlCredentialFunctionCallName: {
		requestEvent:   EventHITLCredentialRequest,
		completedEvent: EventHITLCredentialCompleted,
		pauseKind:      "hitl_credential",
	},
	toolconfirmation.FunctionCallName: {
		requestEvent:   EventHITLConfirmationRequest,
		completedEvent: EventHITLConfirmationCompleted,
		pauseKind:      "hitl_confirmation",
	},
	workflow.WorkflowInputFunctionCallName: {
		requestEvent:   EventHITLInputRequest,
		completedEvent: EventHITLInputCompleted,
		pauseKind:      "hitl_input",
	},
}

var modelResponseCodes = newSet(
	string(genai.FinishReasonUnspecified),
	string(genai.FinishReasonStop),
	string(genai.FinishReasonMaxTokens),
	string(genai.FinishReasonSafety),
	string(genai.FinishReasonRecitation),
	string(genai.FinishReasonLanguage),
	string(genai.FinishReasonOther),
	string(genai.FinishReasonBlocklist),
	string(genai.FinishReasonProhibitedContent),
	string(genai.FinishReasonSPII),
	string(genai.FinishReasonMalformedFunctionCall),
	string(genai.FinishReasonImageSafety),
	string(genai.FinishReasonUnexpectedToolCall),
	string(genai.FinishReasonImageProhibitedContent),
	string(genai.FinishReasonNoImage),
	string(genai.FinishReasonImageRecitation),
	string(genai.FinishReasonImageOther),
	string(genai.BlockedReasonUnspecified),
	string(genai.BlockedReasonSafety),
	string(genai.BlockedReasonOther),
	string(genai.BlockedReasonBlocklist),
	string(genai.BlockedReasonProhibitedContent),
	string(genai.BlockedReasonImageSafety),
	string(genai.BlockedReasonModelArmor),
	string(genai.BlockedReasonJailbreak),
)

func isModelResponseCode(code string) bool {
	_, ok := modelResponseCodes[strings.TrimSpace(code)]
	return ok
}

func stripGenericParams(name string) string {
	if idx := strings.IndexByte(name, '['); idx >= 0 {
		return name[:idx]
	}
	return name
}

var unwrapToolType = reflect.TypeFor[interface{ Unwrap() tool.Tool }]()

// isADKType reports whether t is declared in core ADK Go packages
// (google.golang.org/adk/v2), whose wrapper and agent struct layouts are
// owned by ADK rather than user code.
func isADKType(t reflect.Type) bool {
	if t == nil {
		return false
	}
	pkg := t.PkgPath()
	return pkg == adkModulePath || strings.HasPrefix(pkg, adkModulePath+"/")
}

// toolOriginOf classifies t by its type, as adk-python's _get_tool_origin
// does, after unwrapping confirmation and delegating wrappers. The name of a
// tool plays no part, so a function tool named transfer_to_agent is LOCAL.
// analyticsPlugin.resolveToolOrigin tells a transfer to a remote agent apart.
func toolOriginOf(t tool.Tool) string {
	if isNil(t) {
		return toolOriginUnknown
	}
	v := unwrapToolValue(reflect.ValueOf(t))
	if !v.IsValid() {
		return toolOriginUnknown
	}
	typ := v.Type()
	return classifyADKToolType(v, typ.PkgPath(), typ.Name())
}

// classifyADKToolType classifies a tool of the type named typeName, which
// includes its type arguments, declared in package pkg. The checks follow
// adk-python's order. ADK Go builds its skill tools with functiontool, from
// argument types of the internal skilltool package, so such a function tool
// is SKILL, as adk-python records its skill tools.
func classifyADKToolType(v reflect.Value, pkg, typeName string) string {
	baseName := stripGenericParams(typeName)
	switch {
	case pkg == adkModulePath+"/tool/mcptoolset",
		strings.HasPrefix(pkg, adkModulePath+"/tool/mcptoolset/"):
		return toolOriginMCP
	case baseName == "TransferToAgentTool" && (strings.HasPrefix(pkg, adkModulePath+"/internal/") || pkg == adkModulePath+"/tool"):
		return toolOriginTransferAgent
	case pkg == adkModulePath+"/tool/agenttool",
		strings.HasPrefix(pkg, adkModulePath+"/tool/agenttool/"),
		pkg == adkModulePath+"/internal/workflowinternal" && (baseName == "TaskAgentTool" || baseName == "SingleTurnTool"):
		return subAgentOrA2A(v)
	case pkg == adkModulePath+"/tool/functiontool" && (strings.Contains(typeName, adkModulePath+"/tool/skilltoolset/") || strings.Contains(typeName, "skilltool")):
		return toolOriginSkill
	case pkg == adkModulePath+"/tool/functiontool" && (strings.Contains(strings.ToLower(typeName), "toolbox") || strings.Contains(strings.ToLower(pkg), "toolbox")):
		return toolOriginToolbox
	case pkg == adkModulePath+"/tool/functiontool",
		pkg == adkModulePath+"/tool/loadmemorytool":
		return toolOriginLocal
	case strings.Contains(strings.ToLower(pkg), "toolbox") || strings.Contains(strings.ToLower(baseName), "toolbox"):
		return toolOriginToolbox
	case pkg == adkModulePath+"/tool/skilltoolset",
		strings.HasPrefix(pkg, adkModulePath+"/tool/skilltoolset/"),
		strings.Contains(strings.ToLower(pkg), "skill"):
		return toolOriginSkill
	default:
		return toolOriginUnknown
	}
}

func subAgentOrA2A(v reflect.Value) string {
	if v.Kind() == reflect.Struct && isADKType(v.Type()) {
		for _, fieldName := range []string{"agent", "Agent", "subAgent", "SubAgent", "targetAgent", "TargetAgent"} {
			af := orZero(func() reflect.Value { return v.FieldByName(fieldName) })
			if !af.IsValid() {
				continue
			}
			if (af.Kind() == reflect.Interface || af.Kind() == reflect.Pointer) && af.IsNil() {
				continue
			}
			if isRemoteA2AAgentValue(af) {
				return toolOriginA2A
			}
			return toolOriginSubAgent
		}
	}
	return toolOriginSubAgent
}

const maxUnwrapHops = 8

// unwrapToolValue unwraps confirmation and delegating tool wrappers through
// exported Unwrap() methods and, for core ADK wrapper types, unexported
// wrapper fields (such as tool.WithConfirmation's confirmationTool). Non-ADK
// struct fields are never read, so user wrappers that guard inner state with
// their own mutexes are only unwrapped if they implement Unwrap() tool.Tool.
// Every Unwrap() or inner-field transition consumes one hop from a bounded
// budget, and pointer cycles return an invalid reflect.Value (UNKNOWN).
func unwrapToolValue(v reflect.Value) (unwrapped reflect.Value) {
	defer func() {
		if r := recover(); r != nil {
			unwrapped = reflect.Value{}
		}
	}()
	visited := make(map[visitKey]bool)
hopLoop:
	for range maxUnwrapHops {
		fromPtr := false
		for v.IsValid() && (v.Kind() == reflect.Interface || v.Kind() == reflect.Pointer) {
			if v.IsNil() {
				return reflect.Value{}
			}
			if v.Kind() == reflect.Pointer {
				key := visitKey{ptr: v.Pointer(), typ: v.Type()}
				if visited[key] {
					return reflect.Value{}
				}
				visited[key] = true
			}
			if v.CanInterface() && v.Type().Implements(unwrapToolType) {
				if u, ok := v.Interface().(interface{ Unwrap() tool.Tool }); ok {
					if inner := orZero(u.Unwrap); !isNil(inner) {
						v = reflect.ValueOf(inner)
						continue hopLoop
					}
				}
			}
			if v.Kind() == reflect.Pointer {
				fromPtr = true
			}
			v = v.Elem()
		}
		if !v.IsValid() {
			return reflect.Value{}
		}
		if !fromPtr && v.CanInterface() && v.Type().Implements(unwrapToolType) {
			if u, ok := v.Interface().(interface{ Unwrap() tool.Tool }); ok {
				if inner := orZero(u.Unwrap); !isNil(inner) {
					v = reflect.ValueOf(inner)
					continue hopLoop
				}
			}
		}
		if v.Kind() != reflect.Struct || !isADKType(v.Type()) {
			return v
		}

		if !v.CanAddr() && v.CanInterface() {
			cp := reflect.New(v.Type()).Elem()
			cp.Set(v)
			v = cp
		}

		didUnwrap := false
		for _, fieldName := range []string{
			"runnableTool", "RunnableTool",
			"streamingRunnableTool", "StreamingRunnableTool",
			"innerTool", "InnerTool",
			"wrappedTool", "WrappedTool",
		} {
			f := orZero(func() reflect.Value { return v.FieldByName(fieldName) })
			if !f.IsValid() {
				continue
			}
			if (f.Kind() == reflect.Interface || f.Kind() == reflect.Pointer) && f.IsNil() {
				continue
			}
			if !f.CanInterface() && f.CanAddr() {
				f = reflect.NewAt(f.Type(), f.Addr().UnsafePointer()).Elem()
			}
			v = f
			didUnwrap = true
			break
		}
		if !didUnwrap {
			return v
		}
	}
	return reflect.Value{}
}

// isRemoteA2AAgentValue inspects an agent reflect.Value to detect remote A2A agents.
// In ADK Go, remoteagent.NewA2A returns an *agent.agent whose embedded
// agentinternal.State has AgentType == "RemoteAgent" and Config of type
// iremoteagent.RemoteAgentState.
func isRemoteA2AAgentValue(v reflect.Value) bool {
	return isRemoteA2AAgentValueDepth(v, 0)
}

func isRemoteA2AAgentValueDepth(v reflect.Value, depth int) bool {
	if depth > 6 {
		return false
	}
	for v.IsValid() && (v.Kind() == reflect.Interface || v.Kind() == reflect.Pointer) {
		if v.IsNil() {
			return false
		}
		v = v.Elem()
	}
	if !v.IsValid() || v.Kind() != reflect.Struct || !isADKType(v.Type()) {
		return false
	}

	if at := v.FieldByName("AgentType"); at.IsValid() && at.Kind() == reflect.String {
		if at.String() == "RemoteAgent" {
			return true
		}
	}
	for _, fieldName := range []string{"Config", "config", "State", "state"} {
		if cf := v.FieldByName(fieldName); cf.IsValid() {
			if isRemoteA2AAgentValueDepth(cf, depth+1) {
				return true
			}
		}
	}
	return false
}

func findAgentA2AByName(v reflect.Value, targetName string, visited map[uintptr]bool, depth int) (bool, bool) {
	if depth > 6 || !v.IsValid() {
		return false, false
	}
	for v.IsValid() && (v.Kind() == reflect.Interface || v.Kind() == reflect.Pointer) {
		if v.IsNil() {
			return false, false
		}
		if v.Kind() == reflect.Pointer {
			ptr := v.Pointer()
			if ptr != 0 {
				if visited[ptr] {
					return false, false
				}
				visited[ptr] = true
			}
		}
		if v.CanInterface() {
			if ag, ok := v.Interface().(agent.Agent); ok && ag != nil {
				if strings.TrimSpace(orZero(ag.Name)) == targetName {
					return true, isRemoteA2AAgentValue(v)
				}
				if found := orZero(func() agent.Agent { return ag.FindAgent(targetName) }); found != nil {
					return true, isRemoteA2AAgentValue(reflect.ValueOf(found))
				}
			}
		}
		v = v.Elem()
	}
	if !v.IsValid() {
		return false, false
	}

	switch v.Kind() {
	case reflect.Struct:
		if !isADKType(v.Type()) {
			return false, false
		}
		if !v.CanAddr() && v.CanInterface() {
			cp := reflect.New(v.Type()).Elem()
			cp.Set(v)
			v = cp
		}
		for _, fieldName := range []string{
			"supportedAgents", "SupportedAgents",
			"SubAgents", "subAgents", "Agents", "agents",
			"RootAgent", "rootAgent", "CurrentAgent", "currentAgent",
			"ParentAgent", "parentAgent", "Agent", "agent",
			"TargetAgent", "targetAgent",
		} {
			f := v.FieldByName(fieldName)
			if !f.IsValid() {
				continue
			}
			if !f.CanInterface() && f.CanAddr() {
				f = reflect.NewAt(f.Type(), f.Addr().UnsafePointer()).Elem()
			}
			if found, isA2A := findAgentA2AByName(f, targetName, visited, depth+1); found {
				return true, isA2A
			}
		}
	case reflect.Slice, reflect.Array:
		for i := range v.Len() {
			if found, isA2A := findAgentA2AByName(v.Index(i), targetName, visited, depth+1); found {
				return true, isA2A
			}
		}
	}
	return false, false
}

func nonNilMap(m map[string]any) map[string]any {
	if m == nil {
		return map[string]any{}
	}
	return m
}

// scrubTempStateDelta returns a copy of delta with any session.KeyPrefixTemp
// ("temp:") keys removed. ADK Go session services match "temp:" case-sensitively
// as invocation-local scratch state; temp-only deltas emit no durable STATE_DELTA row.
func scrubTempStateDelta(delta map[string]any) map[string]any {
	if len(delta) == 0 {
		return nil
	}
	var out map[string]any
	for k, v := range delta {
		if strings.HasPrefix(strings.TrimSpace(k), session.KeyPrefixTemp) {
			continue
		}
		if out == nil {
			out = make(map[string]any, len(delta))
		}
		out[k] = v
	}
	return out
}

// extractA2AMetadata returns all custom metadata entries whose key starts with
// "a2a:". A2A_INTERACTION rows mirror adk-python: they fire when "a2a:request"
// or "a2a:response" is present, store "a2a:response" as content, and include
// all "a2a:" keys (such as "a2a:task_id" and "a2a:context_id") in
// attributes.a2a_metadata.
func extractA2AMetadata(meta map[string]any) map[string]any {
	if len(meta) == 0 {
		return nil
	}
	var out map[string]any
	for k, v := range meta {
		lower := strings.ToLower(strings.TrimSpace(k))
		if !strings.HasPrefix(lower, "a2a:") {
			continue
		}
		if out == nil {
			out = make(map[string]any)
		}
		out[k] = v
	}
	return out
}

// extractAgentStateCheckpoint checks custom metadata for agent state checkpoint
// fields ("adk_agent_state" or "agent_state", and "end_of_agent" or
// "adk_end_of_agent"). Terminal checkpoints (end_of_agent == true) are retained
// even when agent_state is nil, matching adk-python.
func extractAgentStateCheckpoint(meta map[string]any) (any, bool, bool) {
	if len(meta) == 0 {
		return nil, false, false
	}
	endOfAgent, _ := meta["end_of_agent"].(bool)
	if !endOfAgent {
		endOfAgent, _ = meta["adk_end_of_agent"].(bool)
	}
	stateVal, ok := meta["adk_agent_state"]
	if !ok || isNil(stateVal) {
		if alt, altOK := meta["agent_state"]; altOK {
			stateVal = alt
		}
	}
	if isNil(stateVal) && !endOfAgent {
		return nil, false, false
	}
	return stateVal, endOfAgent, true
}

func epochSeconds(ts time.Time) any {
	if ts.IsZero() {
		return nil
	}
	return float64(ts.Unix()) + float64(ts.Nanosecond())/1e9
}

const maxCompactionUnwrapDepth = 16

// unwrapCompactionValue dereferences interface and pointer wrappers around a
// compaction value, stopping on nil, pointer cycles, or after
// maxCompactionUnwrapDepth steps so cyclic CustomMetadata["adk_compaction"]
// values cannot hang the callback.
func unwrapCompactionValue(v reflect.Value) (reflect.Value, bool) {
	var seen map[visitKey]bool
	for range maxCompactionUnwrapDepth {
		if !v.IsValid() {
			return reflect.Value{}, false
		}
		if v.Kind() != reflect.Interface && v.Kind() != reflect.Pointer {
			return v, true
		}
		if v.IsNil() {
			return reflect.Value{}, false
		}
		if v.Kind() == reflect.Pointer {
			if seen == nil {
				seen = make(map[visitKey]bool)
			}
			key := visitKey{ptr: v.Pointer(), typ: v.Type()}
			if seen[key] {
				return reflect.Value{}, false
			}
			seen[key] = true
		}
		v = v.Elem()
	}
	return reflect.Value{}, false
}

func hasActionsCompaction(actions any) bool {
	if isNil(actions) {
		return false
	}
	v, ok := unwrapCompactionValue(reflect.ValueOf(actions))
	if !ok || v.Kind() != reflect.Struct {
		return false
	}
	f := v.FieldByName("Compaction")
	if !f.IsValid() {
		return false
	}
	_, ok = unwrapCompactionValue(f)
	return ok
}

// extractEventCompaction extracts compaction payload from Actions.Compaction or
// CustomMetadata["adk_compaction"]. Actions.Compaction is read reflectively so
// this submodule builds cleanly against both tagged ADK v2.0.0 (before
// session.EventActions.Compaction was added) and main via go.work.
func extractEventCompaction(ev *session.Event, cleanMeta map[string]any, maxLen int, redact bool) (map[string]any, bool, bool) {
	if ev == nil {
		return nil, false, false
	}
	if payload, trunc, ok := extractActionsCompaction(ev.Actions, maxLen, redact); ok {
		return payload, trunc, true
	}
	if cleanMeta == nil {
		return nil, false, false
	}
	if raw, exists := cleanMeta["adk_compaction"]; exists && !isNil(raw) {
		if payload, trunc, ok := formatCompactionValue(reflect.ValueOf(raw), maxLen, redact); ok {
			return payload, trunc, true
		}
	}
	return nil, false, false
}

func extractActionsCompaction(actions any, maxLen int, redact bool) (map[string]any, bool, bool) {
	if isNil(actions) {
		return nil, false, false
	}
	v, ok := unwrapCompactionValue(reflect.ValueOf(actions))
	if !ok || v.Kind() != reflect.Struct {
		return nil, false, false
	}
	if f := v.FieldByName("Compaction"); f.IsValid() {
		return formatCompactionValue(f, maxLen, redact)
	}
	return nil, false, false
}

func scrubCompactionText(s string, maxLen int, redact bool) (string, bool) {
	return sanitizeAndTruncateText(s, maxLen, redact)
}

func scrubCompactionGenAIContent(c *genai.Content, maxLen int, redact bool) (string, bool) {
	summary, trunc := formatResponseContent(c, maxLen, redact)
	if !redact || summary == "" {
		return summary, trunc
	}
	scrubbed, sTrunc := scrubCompactionText(summary, maxLen, redact)
	return scrubbed, trunc || sTrunc
}

// formatCompactionValue formats a compaction struct or map as an
// EVENT_COMPACTION payload.
func formatCompactionValue(v reflect.Value, maxLen int, redact bool) (map[string]any, bool, bool) {
	v, ok := unwrapCompactionValue(v)
	if !ok {
		return nil, false, false
	}

	if v.Kind() == reflect.Struct && v.CanInterface() {
		startTS := compactionTimestamp(v.FieldByName("StartTimestamp"))
		endTS := compactionTimestamp(v.FieldByName("EndTimestamp"))
		var summary string
		var trunc bool
		if cf := v.FieldByName("CompactedContent"); cf.IsValid() && cf.CanInterface() {
			switch c := cf.Interface().(type) {
			case *genai.Content:
				summary, trunc = scrubCompactionGenAIContent(c, maxLen, redact)
			case string:
				summary, trunc = scrubCompactionText(c, maxLen, redact)
			}
		}
		if startTS == nil && endTS == nil && summary == "" {
			return nil, false, false
		}
		return map[string]any{
			"start_timestamp":   startTS,
			"end_timestamp":     endTS,
			"compacted_content": summary,
		}, trunc, true
	}

	if v.Kind() == reflect.Map && v.CanInterface() {
		if m, ok := v.Interface().(map[string]any); ok {
			out := maps.Clone(m)
			normalizeCompactionKeys(out)
			_, hasStart := out["start_timestamp"]
			_, hasEnd := out["end_timestamp"]
			_, hasContent := out["compacted_content"]
			if !hasStart && !hasEnd && !hasContent {
				return nil, false, false
			}
			startTS := compactionTimestampValue(out["start_timestamp"])
			endTS := compactionTimestampValue(out["end_timestamp"])
			var summary string
			var trunc bool
			switch c := out["compacted_content"].(type) {
			case *genai.Content:
				summary, trunc = scrubCompactionGenAIContent(c, maxLen, redact)
			case string:
				summary, trunc = scrubCompactionText(c, maxLen, redact)
			case map[string]any:
				if gc, ok := genaiContentFromMap(c); ok {
					summary, trunc = scrubCompactionGenAIContent(gc, maxLen, redact)
				}
			}
			return map[string]any{
				"start_timestamp":   startTS,
				"end_timestamp":     endTS,
				"compacted_content": summary,
			}, trunc, true
		}
	}
	return nil, false, false
}

func compactionTimestampValue(v any) any {
	if v == nil {
		return nil
	}
	if ts, ok := v.(time.Time); ok {
		return epochSeconds(ts)
	}
	rv := reflect.ValueOf(v)
	return compactionTimestamp(rv)
}

// compactionTimestamp returns a compaction struct's timestamp field in epoch
// seconds. A time.Time is converted; a number is taken to be epoch seconds
// already and kept as is. Zero values are unset and return nil.
func compactionTimestamp(f reflect.Value) any {
	if !f.IsValid() || !f.CanInterface() || f.IsZero() {
		return nil
	}
	if ts, ok := f.Interface().(time.Time); ok {
		return epochSeconds(ts)
	}
	if f.Type() == jsonNumberType {
		return f.Interface()
	}
	switch f.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
		reflect.Float32, reflect.Float64:
		return f.Interface()
	default:
		return nil
	}
}

// compactionKeyAliases lists, for each EVENT_COMPACTION payload key, the keys
// a compaction struct has once scrubbing rewrites it as a map: its Go field
// name and its session.EventCompaction json tag.
var compactionKeyAliases = [...]struct {
	key     string
	aliases [2]string
}{
	{"start_timestamp", [2]string{"StartTimestamp", "startTimestamp"}},
	{"end_timestamp", [2]string{"EndTimestamp", "endTimestamp"}},
	{"compacted_content", [2]string{"CompactedContent", "compactedContent"}},
}

// normalizeCompactionKeys renames alias keys in m to their payload keys. An
// existing payload key wins over its aliases; aliases are always removed.
func normalizeCompactionKeys(m map[string]any) {
	for _, a := range compactionKeyAliases {
		for _, alias := range a.aliases {
			v, ok := m[alias]
			if !ok {
				continue
			}
			delete(m, alias)
			if _, exists := m[a.key]; !exists {
				m[a.key] = v
			}
		}
	}
}

// genaiContentFromMap rebuilds a *genai.Content from the map that scrubbing
// produces when it rewrites one, so a scrubbed summary is formatted like an
// untouched one. ok is false if m does not decode as content with parts.
func genaiContentFromMap(m map[string]any) (*genai.Content, bool) {
	if _, ok := m["parts"]; !ok {
		return nil, false
	}
	b, err := safeCall(func() ([]byte, error) { return json.Marshal(m) })
	if err != nil {
		return nil, false
	}
	var c genai.Content
	if err := json.Unmarshal(b, &c); err != nil || len(c.Parts) == 0 {
		return nil, false
	}
	return &c, true
}

func hasFunctionCall(content *genai.Content) bool {
	if content == nil {
		return false
	}
	for _, part := range content.Parts {
		if part != nil && part.FunctionCall != nil {
			return true
		}
	}
	return false
}

// isLoggableFinalAgentResponse reports whether ev is a final agent turn that
// may emit an AGENT_RESPONSE row once visible text parts are filtered. It
// allows SkipSummarization events (such as agenttool's final display event,
// which pairs a FunctionResponse with visible answer text) while excluding
// partial chunks, user-authored events, compaction bookkeeping, long-running
// pause events, and FunctionCall steps. Response-only SkipSummarization
// acknowledgements carry no visible text parts and are skipped by
// filterVisibleTextParts.
func isLoggableFinalAgentResponse(ev *session.Event) bool {
	if ev == nil || ev.Partial || ev.Author == userAuthor || ev.Content == nil || len(ev.Content.Parts) == 0 {
		return false
	}
	if len(ev.LongRunningToolIDs) > 0 || hasFunctionCall(ev.Content) || hasActionsCompaction(ev.Actions) {
		return false
	}
	return isFinalResponse(ev) || ev.Actions.SkipSummarization
}

func isFinalResponse(ev *session.Event) bool {
	if ev == nil || ev.Content == nil {
		return false
	}
	hasNil := false
	for _, part := range ev.Content.Parts {
		if part == nil {
			hasNil = true
			break
		}
	}
	if !hasNil {
		return orZero(ev.IsFinalResponse)
	}
	var cleanParts []*genai.Part
	for _, part := range ev.Content.Parts {
		if part != nil {
			cleanParts = append(cleanParts, part)
		}
	}
	evCopy := *ev
	contentCopy := *ev.Content
	contentCopy.Parts = cleanParts
	evCopy.Content = &contentCopy
	return orZero(evCopy.IsFinalResponse)
}

func filterVisibleTextParts(parts []*genai.Part) []*genai.Part {
	if len(parts) == 0 {
		return nil
	}
	var out []*genai.Part
	for _, part := range parts {
		if part == nil || part.Thought || part.Text == "" {
			continue
		}
		out = append(out, part)
	}
	return out
}

// resolveToolOriginForCall determines the provenance of a tool in the context of a call.
func resolveToolOriginForCall(_ context.Context, t tool.Tool, _ map[string]any) (origin string) {
	defer func() {
		if r := recover(); r != nil {
			origin = toolOriginUnknown
		}
	}()
	if origin = toolOriginOf(t); origin != toolOriginUnknown {
		return origin
	}
	// Check tool name only after tool type classification, so a functiontool
	// named transfer_to_agent is classified as LOCAL.
	if strings.TrimSpace(toolName(t)) == "transfer_to_agent" {
		return toolOriginTransferAgent
	}
	return toolOriginUnknown
}

// resolveToolOrigin resolves tool provenance for a tool callback, also consulting
// the traceManager's cached agent hierarchy for transfer_to_agent calls.
func (p *analyticsPlugin) resolveToolOrigin(ctx context.Context, id identity, t tool.Tool, args map[string]any) (origin string) {
	defer func() {
		if r := recover(); r != nil {
			origin = toolOriginUnknown
		}
	}()
	origin = resolveToolOriginForCall(ctx, t, args)
	if origin == toolOriginTransferAgent {
		if targetName, _ := args["agent_name"].(string); strings.TrimSpace(targetName) != "" {
			targetName = strings.TrimSpace(targetName)
			v := unwrapToolValue(reflect.ValueOf(t))
			if v.IsValid() {
				visited := make(map[uintptr]bool)
				if found, isA2A := findAgentA2AByName(v, targetName, visited, 0); found && isA2A {
					return toolOriginTransferA2A
				}
			}
			if p != nil && p.traces != nil {
				targetID := id
				targetID.agentName = targetName
				if targetAgent := p.traces.agentNamed(ctx, targetID); targetAgent != nil {
					if isRemoteA2AAgentValue(reflect.ValueOf(targetAgent)) {
						return toolOriginTransferA2A
					}
				}
			}
		}
	}
	return origin
}

// resumedToolOrigin returns the origin of the tool called name whose paused
// call a function response in a user message answers. The response does not
// carry the tool, so it is looked up by name among the tools configured on
// the agent that the runner runs and its sub-agents, which ADK sets to the
// agent that made the call when it resumes one, and then in the rest of the
// agent tree. Tools that toolsets provide are not listed, as listing them
// can do I/O, so they resolve to [toolOriginUnknown], as do names shared by
// tools of different origins.
func (p *analyticsPlugin) resumedToolOrigin(ctx agent.InvocationContext, name string) string {
	if name == "" {
		return toolOriginUnknown
	}
	origin := orZero(func() string {
		runner := runnerAgentOf(ctx)
		if origin, found := configuredToolOrigin(ctx, runner, name); found {
			return origin
		}
		if p.traces == nil {
			return ""
		}
		// Look in the tree of the highest ancestor of runner seen so far.
		if root := p.traces.rememberAgentTree(runner); root != runner {
			origin, _ := configuredToolOrigin(ctx, root, name)
			return origin
		}
		return ""
	})
	return cmp.Or(origin, toolOriginUnknown)
}

// configuredToolOrigin returns the origin of the tools called name that are
// configured on the LLM agents of the tree of root, and whether there are
// any. Tools of different origins yield [toolOriginUnknown].
func configuredToolOrigin(ctx context.Context, root agent.Agent, name string) (origin string, found bool) {
	visited := map[agent.Agent]bool{}
	var walk func(a agent.Agent)
	walk = func(a agent.Agent) {
		if isNil(a) || visited[a] {
			return
		}
		visited[a] = true
		if cfg, ok := orZero(func() any { return agentConfig(a) }).(llmagent.Config); ok {
			for _, t := range cfg.Tools {
				if isNil(t) || toolName(t) != name {
					continue
				}
				o := resolveToolOriginForCall(ctx, t, nil)
				if found && o != origin {
					o = toolOriginUnknown
				}
				origin, found = o, true
			}
		}
		for _, sub := range orZero(a.SubAgents) {
			walk(sub)
		}
	}
	walk(root)
	return origin, found
}

// emitUserMessageSignalEvents emits HITL completion or resumed TOOL_COMPLETED rows
// when an incoming user message carries FunctionResponse parts.
func (p *analyticsPlugin) emitUserMessageSignalEvents(ctx agent.InvocationContext, id identity, msg *genai.Content, span rowSpan) {
	if msg == nil {
		return
	}
	spanFn := func() rowSpan { return span }
	for _, part := range msg.Parts {
		if part == nil || part.FunctionResponse == nil {
			continue
		}
		fr := part.FunctionResponse
		if hitl, ok := hitlTools[fr.Name]; ok {
			p.logSessionSpanEvent(ctx, id, hitl.completedEvent, spanFn, func(d *eventData) {
				d.author = userAuthor
				d.functionCallID = fr.ID
				d.content = map[string]any{
					"tool_id": fr.ID,
					"tool":    fr.Name,
					"result":  nonNilMap(fr.Response),
				}
			})
			continue
		}
		p.logSessionSpanEvent(ctx, id, EventToolCompleted, spanFn, func(d *eventData) {
			d.author = userAuthor
			d.functionCallID = fr.ID
			d.adkExtras = map[string]any{"pause_kind": "tool"}
			d.content = map[string]any{
				"tool":        fr.Name,
				"result":      nonNilMap(fr.Response),
				"tool_origin": p.resumedToolOrigin(ctx, fr.Name),
			}
		})
	}
}

// scrubA2ACustomMetadata filters and redacts custom metadata map.
// Only keys allowed by allowlist or starting with "a2a:" are retained,
// and their values are scrubbed using standard key-based redaction (redactJSONValue).
func scrubA2ACustomMetadata(meta map[string]any, allowlist customMetadataFilter, redact bool) (map[string]any, bool) {
	if len(meta) == 0 {
		return nil, false
	}
	out := make(map[string]any)
	var trunc bool
	for k, v := range meta {
		if !allowlist.allows(k) && !strings.HasPrefix(k, "a2a:") {
			continue
		}
		if redact {
			if isSensitiveKey(k) || isSensitiveKey(strings.TrimPrefix(k, "a2a:")) {
				out[k] = redactedValue
				continue
			}
			budget := &nodeBudget{remaining: maxSanitizeNodes}
			cleanVal, _, lost := redactJSONValue(v, 0, budget)
			out[k] = cleanVal
			trunc = trunc || lost
		} else {
			out[k] = v
		}
	}
	if len(out) == 0 {
		return nil, false
	}
	return out, trunc
}

// emitSessionEventSignals evaluates all orthogonal session.Event signals without
// short-circuiting so a single event can emit multiple domain rows.
//
// Partial streamed events (ev.Partial == true) are ignored at entry. ADK Go's
// streamAggregator and finalizeModelResponseEvent attach LongRunningToolIDs and
// the step's stateDelta to every streamed partial chunk as well as to the final
// aggregated event, and runner.Run persists only non-partial events. Unlike
// adk-python (which currently lacks a partial guard on pause/state rows),
// returning early here prevents duplicate STATE_DELTA and TOOL_PAUSED rows on
// SSE runs and avoids phantom pause rows when a stream aborts before commit.
func (p *analyticsPlugin) emitSessionEventSignals(ctx agent.InvocationContext, ev *session.Event) {
	if ev == nil || ev.Partial {
		return
	}

	id := identityOf(ctx)
	if ev.NodeInfo != nil && ev.NodeInfo.Path != "" {
		id.path = ev.NodeInfo.Path
	}
	span := p.traces.eventSpan(ctx, id, ev.Author, ev.Branch)
	cleanCustomMeta, a2aTrunc := scrubA2ACustomMetadata(ev.CustomMetadata, p.customMetadata, p.config.redactSensitiveKeys())
	applyEventBase := func(d *eventData) {
		d.sourceEvent = ev
		d.author = ev.Author
		d.customMetadata = cleanCustomMeta
		d.customMetadataScrubbed = true
		d.customMetadataTruncated = a2aTrunc
	}

	// 1. STATE_DELTA (scrubbing temp: keys, writing delta to attributes.state_delta only).
	if cleanDelta := scrubTempStateDelta(ev.Actions.StateDelta); len(cleanDelta) > 0 {
		p.logSessionEvent(ctx, id, EventStateDelta, span, func(d *eventData) {
			applyEventBase(d)
			d.attributes = map[string]any{"state_delta": maps.Clone(cleanDelta)}
		})
	}

	// 2. AGENT_TRANSFER.
	if target := strings.TrimSpace(ev.Actions.TransferToAgent); target != "" {
		p.logSessionEvent(ctx, id, EventAgentTransfer, span, func(d *eventData) {
			applyEventBase(d)
			d.content = map[string]any{
				"from_agent": ev.Author,
				"to_agent":   target,
			}
		})
	}

	// 3. EVENT_COMPACTION.
	if p.shouldLogSessionEvent(EventEventCompaction) {
		if compPayload, compTrunc, ok := extractEventCompaction(ev, cleanCustomMeta, p.config.maxContentLength(), p.config.redactSensitiveKeys()); ok {
			p.logSessionEvent(ctx, id, EventEventCompaction, span, func(d *eventData) {
				applyEventBase(d)
				d.content = compPayload
				d.contentTruncated = compTrunc || a2aTrunc
			})
		}
	}

	// 4. AGENT_STATE_CHECKPOINT.
	if stateVal, endOfAgent, ok := extractAgentStateCheckpoint(cleanCustomMeta); ok {
		p.logSessionEvent(ctx, id, EventAgentStateCheckpoint, span, func(d *eventData) {
			applyEventBase(d)
			d.content = map[string]any{
				"agent_state":  stateVal,
				"end_of_agent": endOfAgent,
			}
			d.contentTruncated = a2aTrunc
		})
	}

	// 5. HITL Requests, TOOL_PAUSED, and HITL Completions.
	p.emitHITLAndPauseSignals(ctx, id, ev, span, applyEventBase)

	// 6. A2A_INTERACTION, NODE_ERROR, NODE_OUTPUT, AGENT_RESPONSE.
	if a2aMeta := extractA2AMetadata(cleanCustomMeta); len(a2aMeta) > 0 {
		a2aReq := a2aMeta["a2a:request"]
		a2aResp := a2aMeta["a2a:response"]
		if !isNil(a2aReq) || !isNil(a2aResp) {
			p.logSessionEvent(ctx, id, EventA2AInteraction, span, func(d *eventData) {
				applyEventBase(d)
				d.content = a2aResp
				d.contentTruncated = a2aTrunc
				d.attributes = map[string]any{"a2a_metadata": a2aMeta}
			})
		}
	}

	if ni := ev.NodeInfo; ni != nil && ni.Path != "" {
		if ev.ErrorCode != "" && !isModelResponseCode(ev.ErrorCode) {
			p.logSessionEvent(ctx, id, EventNodeError, span, func(d *eventData) {
				applyEventBase(d)
				d.failed = true
				d.errMsg = cmp.Or(ev.ErrorMessage, ev.ErrorCode)
				d.content = map[string]any{
					"node_path":  ni.Path,
					"error_code": ev.ErrorCode,
				}
			})
		}
		if !isNil(ev.Output) && !ni.MessageAsOutput {
			p.logSessionEvent(ctx, id, EventNodeOutput, span, func(d *eventData) {
				applyEventBase(d)
				d.content = ev.Output
			})
		}
	}

	if isLoggableFinalAgentResponse(ev) {
		visibleParts := filterVisibleTextParts(ev.Content.Parts)
		if len(visibleParts) > 0 {
			p.logSessionEvent(ctx, id, EventAgentResponse, span, func(d *eventData) {
				applyEventBase(d)
				visibleContent := &genai.Content{
					Role:  ev.Content.Role,
					Parts: visibleParts,
				}
				summary, summaryTrunc := formatResponseContent(visibleContent, p.config.maxContentLength(), p.config.redactSensitiveKeys())
				var parts []map[string]any
				var partsTrunc bool
				if !p.denied(columnContentParts) && p.config.logMultiModalContent() && p.config.ContentFormatter == nil {
					parts, partsTrunc = formatContentParts(visibleContent, p.config.maxContentLength(), p.config.redactSensitiveKeys())
				}
				attrs := map[string]any{
					"event_author": ev.Author,
				}
				putNonEmpty(attrs, "source_event_id", ev.ID)
				putNonEmpty(attrs, "source_event_author", ev.Author)
				putNonEmpty(attrs, "source_event_branch", ev.Branch)

				d.content = map[string]any{"response": summary}
				d.parts = parts
				d.partsTruncated = partsTrunc
				d.contentTruncated = summaryTrunc
				d.attributes = attrs
			})
		}
	}
}

func (p *analyticsPlugin) emitHITLAndPauseSignals(ctx agent.InvocationContext, id identity, ev *session.Event, span rowSpan, applyEventBase func(*eventData)) {
	var calls []*genai.FunctionCall
	if ev.Content != nil {
		for _, part := range ev.Content.Parts {
			if part != nil && part.FunctionCall != nil {
				calls = append(calls, part.FunctionCall)
			}
		}
	}

	emittedCallIdx := make(map[int]bool, len(calls))
	emittedInput := false

	for _, toolID := range ev.LongRunningToolIDs {
		matchIdx := -1
		for idx, fc := range calls {
			if fc.ID == toolID {
				matchIdx = idx
				break
			}
		}
		if matchIdx == -1 && len(calls) == 1 && !emittedCallIdx[0] && calls[0].ID == "" {
			matchIdx = 0
		}

		if matchIdx >= 0 {
			emittedCallIdx[matchIdx] = true
			fc := calls[matchIdx]
			resolvedID := cmp.Or(toolID, fc.ID)
			pauseKind := "tool"
			args, argsTrunc := nonNilMap(fc.Args), false
			if hitl, ok := hitlTools[fc.Name]; ok {
				pauseKind = hitl.pauseKind
				if fc.Name == workflow.WorkflowInputFunctionCallName {
					emittedInput = true
				}
				p.logSessionEvent(ctx, id, hitl.requestEvent, span, func(d *eventData) {
					applyEventBase(d)
					d.functionCallID = resolvedID
					d.contentTruncated = argsTrunc
					d.content = map[string]any{
						"tool_id": resolvedID,
						"tool":    fc.Name,
						"args":    args,
					}
				})
			}

			p.logSessionEvent(ctx, id, EventToolPaused, span, func(d *eventData) {
				applyEventBase(d)
				d.functionCallID = resolvedID
				d.contentTruncated = argsTrunc
				d.adkExtras = map[string]any{"pause_kind": pauseKind}
				d.content = map[string]any{
					"tool_id": resolvedID,
					"tool":    fc.Name,
					"args":    args,
				}
			})
			continue
		}

		// LongRunningToolID with no matching FunctionCall part.
		p.logSessionEvent(ctx, id, EventToolPaused, span, func(d *eventData) {
			applyEventBase(d)
			d.functionCallID = toolID
			d.adkExtras = map[string]any{"pause_kind": "tool"}
			d.content = map[string]any{
				"tool_id": toolID,
				"tool":    nil,
				"args":    nil,
			}
		})
	}

	// Emit HITL requests for any HITL FunctionCall parts not already matched via LongRunningToolIDs.
	for idx, fc := range calls {
		if emittedCallIdx[idx] {
			continue
		}
		hitl, ok := hitlTools[fc.Name]
		if !ok {
			continue
		}
		if fc.Name == workflow.WorkflowInputFunctionCallName {
			emittedInput = true
		}
		fcCopy := fc
		p.logSessionEvent(ctx, id, hitl.requestEvent, span, func(d *eventData) {
			args, argsTrunc := nonNilMap(fcCopy.Args), false
			applyEventBase(d)
			d.functionCallID = fcCopy.ID
			d.contentTruncated = argsTrunc
			d.content = map[string]any{
				"tool_id": fcCopy.ID,
				"tool":    fcCopy.Name,
				"args":    args,
			}
		})
	}

	// Emit HITL_INPUT_REQUEST from ev.RequestedInput if not already emitted.
	if ev.RequestedInput != nil && !emittedInput {
		p.logSessionEvent(ctx, id, EventHITLInputRequest, span, func(d *eventData) {
			applyEventBase(d)
			d.content = map[string]any{
				"tool":            workflow.WorkflowInputFunctionCallName,
				"args":            ev.RequestedInput,
				"requested_input": ev.RequestedInput,
			}
		})
	}

	// Emit HITL_*_COMPLETED for FunctionResponse parts matching HITL tools.
	if ev.Content != nil {
		for _, part := range ev.Content.Parts {
			if part == nil || part.FunctionResponse == nil {
				continue
			}
			fr := part.FunctionResponse
			hitl, ok := hitlTools[fr.Name]
			if !ok {
				continue
			}
			p.logSessionEvent(ctx, id, hitl.completedEvent, span, func(d *eventData) {
				result, resTrunc := nonNilMap(fr.Response), false
				applyEventBase(d)
				d.functionCallID = fr.ID
				d.contentTruncated = resTrunc
				d.content = map[string]any{
					"tool_id": fr.ID,
					"tool":    fr.Name,
					"result":  result,
				}
			})
		}
	}
}
