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

// Tool origin constants recorded in TOOL_STARTING, TOOL_COMPLETED, and
// TOOL_ERROR payloads. The vocabulary matches adk-python so cross-language
// queries over tool_origin return consistent values.
const (
	ToolOriginLocal         = "LOCAL"
	ToolOriginMCP           = "MCP"
	ToolOriginTransferAgent = "TRANSFER_AGENT"
	ToolOriginTransferA2A   = "TRANSFER_A2A"
	ToolOriginSubAgent      = "SUB_AGENT"
	ToolOriginA2A           = "A2A"
	ToolOriginUnknown       = "UNKNOWN"
)

const (
	transferToAgentToolName        = "transfer_to_agent"
	hitlCredentialFunctionCallName = "adk_request_credential"
)

// ToolOriginProvider can be implemented by custom tool implementations to
// explicitly report their provenance in BigQuery Agent Analytics tool events.
// Implementations should return one of the ToolOrigin* constants (for example
// [ToolOriginLocal], [ToolOriginMCP], [ToolOriginSubAgent], [ToolOriginA2A], or
// [ToolOriginUnknown]). Provenance is preserved when the tool is wrapped by
// [tool.WithConfirmation] or by a wrapper that implements `Unwrap() tool.Tool`.
type ToolOriginProvider interface {
	ToolOrigin() string
}

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

// resolveToolOriginForCall determines the provenance of a tool in the context of a call.
func resolveToolOriginForCall(_ context.Context, t tool.Tool, args map[string]any) string {
	if isNil(t) {
		return ToolOriginUnknown
	}
	if p, ok := t.(ToolOriginProvider); ok {
		if origin := strings.TrimSpace(orZero(p.ToolOrigin)); origin != "" {
			return origin
		}
	}

	v, providerOrigin := unwrapToolValue(reflect.ValueOf(t))
	if providerOrigin != "" {
		return providerOrigin
	}
	if !v.IsValid() {
		return ToolOriginUnknown
	}

	if strings.TrimSpace(toolName(t)) == transferToAgentToolName {
		return transferOrigin(v, args)
	}

	typ := v.Type()
	return classifyADKToolType(v, typ.PkgPath(), stripGenericParams(typ.Name()), args)
}

func classifyADKToolType(v reflect.Value, pkg, baseName string, args map[string]any) string {
	switch {
	case pkg == adkModulePath+"/tool/functiontool",
		pkg == adkModulePath+"/tool/loadmemorytool":
		return ToolOriginLocal
	case pkg == adkModulePath+"/tool/mcptoolset",
		strings.HasPrefix(pkg, adkModulePath+"/tool/mcptoolset/"):
		return ToolOriginMCP
	case pkg == adkModulePath+"/tool/agenttool",
		strings.HasPrefix(pkg, adkModulePath+"/tool/agenttool/"),
		pkg == adkModulePath+"/internal/workflowinternal" && (baseName == "TaskAgentTool" || baseName == "SingleTurnTool"):
		return subAgentOrA2A(v)
	case baseName == "TransferToAgentTool" && strings.HasPrefix(pkg, adkModulePath+"/internal/"):
		return transferOrigin(v, args)
	default:
		return ToolOriginUnknown
	}
}

func subAgentOrA2A(v reflect.Value) string {
	if v.Kind() == reflect.Struct {
		for _, fieldName := range []string{"agent", "Agent", "subAgent", "SubAgent", "targetAgent", "TargetAgent"} {
			af := v.FieldByName(fieldName)
			if !af.IsValid() {
				continue
			}
			if (af.Kind() == reflect.Interface || af.Kind() == reflect.Pointer) && af.IsNil() {
				continue
			}
			if isRemoteA2AAgentValue(af) {
				return ToolOriginA2A
			}
			return ToolOriginSubAgent
		}
	}
	return ToolOriginSubAgent
}

func transferOrigin(v reflect.Value, args map[string]any) string {
	if targetName, _ := args["agent_name"].(string); strings.TrimSpace(targetName) != "" {
		visited := make(map[uintptr]bool)
		if found, isA2A := findAgentA2AByName(v, strings.TrimSpace(targetName), visited, 0); found && isA2A {
			return ToolOriginTransferA2A
		}
	}
	return ToolOriginTransferAgent
}

const maxUnwrapHops = 8

// unwrapToolValue unwraps confirmation and delegating tool wrappers, preserving
// explicit ToolOriginProvider provenance through both exported Unwrap() methods
// and unexported wrapper fields (such as tool.WithConfirmation's confirmationTool).
// Every Unwrap() or inner-field transition consumes one hop from a bounded
// budget, and pointer cycles return an invalid reflect.Value (ToolOriginUnknown).
func unwrapToolValue(v reflect.Value) (reflect.Value, string) {
	visited := make(map[visitKey]bool)
hopLoop:
	for hops := range maxUnwrapHops {
		for v.IsValid() && (v.Kind() == reflect.Interface || v.Kind() == reflect.Pointer) {
			if v.IsNil() {
				return reflect.Value{}, ""
			}
			if v.Kind() == reflect.Pointer {
				key := visitKey{ptr: v.Pointer(), typ: v.Type()}
				if visited[key] {
					return reflect.Value{}, ""
				}
				visited[key] = true
			}
			if v.CanInterface() {
				if hops > 0 {
					if p, ok := v.Interface().(ToolOriginProvider); ok {
						if origin := strings.TrimSpace(orZero(p.ToolOrigin)); origin != "" {
							return v, origin
						}
					}
				}
				if u, ok := v.Interface().(interface{ Unwrap() tool.Tool }); ok {
					if inner := orZero(u.Unwrap); !isNil(inner) {
						v = reflect.ValueOf(inner)
						continue hopLoop
					}
				}
			}
			v = v.Elem()
		}
		if !v.IsValid() {
			return reflect.Value{}, ""
		}
		if v.CanInterface() {
			if hops > 0 {
				if p, ok := v.Interface().(ToolOriginProvider); ok {
					if origin := strings.TrimSpace(orZero(p.ToolOrigin)); origin != "" {
						return v, origin
					}
				}
			}
			if u, ok := v.Interface().(interface{ Unwrap() tool.Tool }); ok {
				if inner := orZero(u.Unwrap); !isNil(inner) {
					v = reflect.ValueOf(inner)
					continue hopLoop
				}
			}
		}
		if v.Kind() != reflect.Struct {
			return v, ""
		}

		if !v.CanAddr() && v.CanInterface() {
			cp := reflect.New(v.Type()).Elem()
			cp.Set(v)
			v = cp
		}

		unwrapped := false
		for _, fieldName := range []string{
			"runnableTool", "RunnableTool",
			"streamingRunnableTool", "StreamingRunnableTool",
			"innerTool", "InnerTool",
			"wrappedTool", "WrappedTool",
		} {
			f := v.FieldByName(fieldName)
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
			unwrapped = true
			break
		}
		if !unwrapped {
			return v, ""
		}
	}
	return reflect.Value{}, ""
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
	if !v.IsValid() {
		return false
	}

	pkgLower := strings.ToLower(v.Type().PkgPath())
	typeLower := strings.ToLower(v.Type().Name())
	if strings.Contains(pkgLower, "remoteagent") ||
		strings.Contains(pkgLower, "/a2a") ||
		strings.Contains(typeLower, "a2a") ||
		strings.Contains(typeLower, "remoteagent") {
		return true
	}

	if v.Kind() != reflect.Struct {
		return false
	}

	if at := v.FieldByName("AgentType"); at.IsValid() && at.Kind() == reflect.String {
		if strings.EqualFold(strings.TrimSpace(at.String()), "RemoteAgent") {
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
	for _, fieldName := range []string{"isA2A", "IsA2A", "isRemote", "IsRemote"} {
		if bf := v.FieldByName(fieldName); bf.IsValid() && bf.Kind() == reflect.Bool && bf.Bool() {
			return true
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
		if name := extractStructAgentName(v); name == targetName {
			return true, isRemoteA2AAgentValue(v)
		}
		for _, fieldName := range []string{
			"supportedAgents", "SupportedAgents",
			"SubAgents", "subAgents", "Agents", "agents",
			"RootAgent", "rootAgent", "CurrentAgent", "currentAgent",
			"ParentAgent", "parentAgent", "Agent", "agent",
			"TargetAgent", "targetAgent", "State", "state",
		} {
			f := v.FieldByName(fieldName)
			if !f.IsValid() {
				continue
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
	case reflect.Map:
		if kt := v.Type().Key(); kt.Kind() == reflect.String {
			if mv := v.MapIndex(reflect.ValueOf(targetName).Convert(kt)); mv.IsValid() {
				return true, isRemoteA2AAgentValue(mv)
			}
		}
		iter := v.MapRange()
		for iter.Next() {
			if found, isA2A := findAgentA2AByName(iter.Value(), targetName, visited, depth+1); found {
				return true, isA2A
			}
		}
	}
	return false, false
}

func extractStructAgentName(v reflect.Value) string {
	for _, fieldName := range []string{"Name", "name", "AgentName", "agentName"} {
		f := v.FieldByName(fieldName)
		if f.IsValid() && f.Kind() == reflect.String {
			if s := strings.TrimSpace(f.String()); s != "" {
				return s
			}
		}
	}
	if sf := v.FieldByName("State"); sf.IsValid() {
		for sf.Kind() == reflect.Interface || sf.Kind() == reflect.Pointer {
			if sf.IsNil() {
				break
			}
			sf = sf.Elem()
		}
		if sf.IsValid() && sf.Kind() == reflect.Struct {
			for _, fieldName := range []string{"Name", "name"} {
				f := sf.FieldByName(fieldName)
				if f.IsValid() && f.Kind() == reflect.String {
					if s := strings.TrimSpace(f.String()); s != "" {
						return s
					}
				}
			}
		}
	}
	return ""
}

func nonNilMap(m map[string]any) map[string]any {
	if m == nil {
		return map[string]any{}
	}
	return m
}

// sanitizeCredentialMap converts and redacts HITL credential payloads using
// the shared bounded sanitizer, which handles typed maps, slices, structs,
// camelCase/snake_case keys, and cyclic references safely.
func (p *analyticsPlugin) sanitizeCredentialMap(m map[string]any) (any, bool) {
	return sanitizeValue(nonNilMap(m), p.config.maxContentLength(), p.config.redactSensitiveKeys())
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

// extractEventCompaction extracts compaction payload from Actions.Compaction or
// CustomMetadata["adk_compaction"]. Actions.Compaction is read reflectively so
// this submodule builds cleanly against both tagged ADK v2.0.0 (before
// session.EventActions.Compaction was added) and main via go.work.
func extractEventCompaction(ev *session.Event, maxLen int, redact bool) (map[string]any, bool, bool) {
	if ev == nil {
		return nil, false, false
	}
	if payload, trunc, ok := extractActionsCompaction(ev.Actions, maxLen, redact); ok {
		return payload, trunc, true
	}
	if raw, exists := ev.CustomMetadata["adk_compaction"]; exists && !isNil(raw) {
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
	v := reflect.ValueOf(actions)
	for v.IsValid() && (v.Kind() == reflect.Interface || v.Kind() == reflect.Pointer) {
		if v.IsNil() {
			return nil, false, false
		}
		v = v.Elem()
	}
	if !v.IsValid() || v.Kind() != reflect.Struct {
		return nil, false, false
	}
	if f := v.FieldByName("Compaction"); f.IsValid() {
		return formatCompactionValue(f, maxLen, redact)
	}
	return nil, false, false
}

func formatCompactionValue(v reflect.Value, maxLen int, redact bool) (map[string]any, bool, bool) {
	for v.IsValid() && (v.Kind() == reflect.Interface || v.Kind() == reflect.Pointer) {
		if v.IsNil() {
			return nil, false, false
		}
		v = v.Elem()
	}
	if !v.IsValid() {
		return nil, false, false
	}

	if v.Kind() == reflect.Struct && v.CanInterface() {
		var startTS, endTS any
		hasField := false
		if sf := v.FieldByName("StartTimestamp"); sf.IsValid() && sf.CanInterface() {
			hasField = true
			if ts, ok := sf.Interface().(time.Time); ok {
				startTS = epochSeconds(ts)
			}
		}
		if ef := v.FieldByName("EndTimestamp"); ef.IsValid() && ef.CanInterface() {
			hasField = true
			if ts, ok := ef.Interface().(time.Time); ok {
				endTS = epochSeconds(ts)
			}
		}
		var summary string
		var trunc bool
		if cf := v.FieldByName("CompactedContent"); cf.IsValid() && cf.CanInterface() {
			hasField = true
			switch c := cf.Interface().(type) {
			case *genai.Content:
				summary, trunc = formatResponseContent(c, maxLen, redact)
			case string:
				summary, trunc = sanitizeAndTruncateText(c, maxLen, redact)
			}
		}
		if !hasField || (startTS == nil && endTS == nil && summary == "") {
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
			_, hasStart := m["start_timestamp"]
			_, hasEnd := m["end_timestamp"]
			_, hasContent := m["compacted_content"]
			if !hasStart && !hasEnd && !hasContent {
				return nil, false, false
			}
			out := maps.Clone(m)
			for _, key := range []string{"start_timestamp", "end_timestamp"} {
				if ts, ok := out[key].(time.Time); ok {
					out[key] = epochSeconds(ts)
				}
			}
			var trunc bool
			switch c := out["compacted_content"].(type) {
			case *genai.Content:
				out["compacted_content"], trunc = formatResponseContent(c, maxLen, redact)
			case string:
				out["compacted_content"], trunc = sanitizeAndTruncateText(c, maxLen, redact)
			}
			return out, trunc, true
		}
	}
	return nil, false, false
}

func hasFunctionCallOrResponse(content *genai.Content) bool {
	if content == nil {
		return false
	}
	for _, part := range content.Parts {
		if part == nil {
			continue
		}
		if part.FunctionCall != nil || part.FunctionResponse != nil {
			return true
		}
	}
	return false
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

// resolveToolOrigin resolves tool provenance for a tool callback, also consulting
// the traceManager's cached agent hierarchy for transfer_to_agent calls.
func (p *analyticsPlugin) resolveToolOrigin(ctx context.Context, id identity, t tool.Tool, args map[string]any) string {
	origin := resolveToolOriginForCall(ctx, t, args)
	if origin == ToolOriginTransferAgent && p != nil && p.traces != nil {
		if targetName, _ := args["agent_name"].(string); strings.TrimSpace(targetName) != "" {
			targetID := id
			targetID.agentName = strings.TrimSpace(targetName)
			if targetAgent := p.traces.agentNamed(ctx, targetID); targetAgent != nil {
				if isRemoteA2AAgentValue(reflect.ValueOf(targetAgent)) {
					return ToolOriginTransferA2A
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
// can do I/O, so they resolve to [ToolOriginUnknown], as do names shared by
// tools of different origins.
func (p *analyticsPlugin) resumedToolOrigin(ctx agent.InvocationContext, name string) string {
	if name == "" {
		return ToolOriginUnknown
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
	return cmp.Or(origin, ToolOriginUnknown)
}

// configuredToolOrigin returns the origin of the tools called name that are
// configured on the LLM agents of the tree of root, and whether there are
// any. Tools of different origins yield [ToolOriginUnknown].
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
					o = ToolOriginUnknown
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
			p.logEvent(ctx, id, hitl.completedEvent, spanFn, func(d *eventData) {
				d.author = userAuthor
				d.functionCallID = fr.ID
				var result any = nonNilMap(fr.Response)
				if fr.Name == hitlCredentialFunctionCallName {
					result, d.contentTruncated = p.sanitizeCredentialMap(fr.Response)
				}
				d.content = map[string]any{
					"tool_id": fr.ID,
					"tool":    fr.Name,
					"result":  result,
				}
			})
			continue
		}
		p.logEvent(ctx, id, EventToolCompleted, spanFn, func(d *eventData) {
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

// emitSessionEventSignals evaluates all orthogonal session.Event signals without
// short-circuiting so a single event can emit multiple domain rows.
func (p *analyticsPlugin) emitSessionEventSignals(ctx agent.InvocationContext, ev *session.Event) {
	if ev == nil {
		return
	}

	id := identityOf(ctx)
	if ev.NodeInfo != nil && ev.NodeInfo.Path != "" {
		id.path = ev.NodeInfo.Path
	}
	span := p.traces.eventSpan(ctx, id, ev.Author, ev.Branch)
	applyEventBase := func(d *eventData) {
		d.sourceEvent = ev
		d.author = ev.Author
		d.customMetadata = ev.CustomMetadata
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
	if compPayload, compTrunc, ok := extractEventCompaction(ev, p.config.maxContentLength(), p.config.redactSensitiveKeys()); ok {
		p.logSessionEvent(ctx, id, EventEventCompaction, span, func(d *eventData) {
			applyEventBase(d)
			d.content = compPayload
			d.contentTruncated = compTrunc
		})
	}

	// 4. AGENT_STATE_CHECKPOINT.
	if stateVal, endOfAgent, ok := extractAgentStateCheckpoint(ev.CustomMetadata); ok {
		p.logSessionEvent(ctx, id, EventAgentStateCheckpoint, span, func(d *eventData) {
			applyEventBase(d)
			d.content = map[string]any{
				"agent_state":  stateVal,
				"end_of_agent": endOfAgent,
			}
		})
	}

	// 5. HITL Requests, TOOL_PAUSED, and HITL Completions.
	p.emitHITLAndPauseSignals(ctx, id, ev, span, applyEventBase)

	// 6. Non-partial signals: A2A_INTERACTION, NODE_ERROR, NODE_OUTPUT, AGENT_RESPONSE.
	if ev.Partial {
		return
	}

	if a2aMeta := extractA2AMetadata(ev.CustomMetadata); len(a2aMeta) > 0 {
		a2aReq := a2aMeta["a2a:request"]
		a2aResp := a2aMeta["a2a:response"]
		if !isNil(a2aReq) || !isNil(a2aResp) {
			p.logSessionEvent(ctx, id, EventA2AInteraction, span, func(d *eventData) {
				applyEventBase(d)
				d.content = a2aResp
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

	if ev.Author != userAuthor &&
		ev.Content != nil &&
		len(ev.Content.Parts) > 0 &&
		!ev.Actions.SkipSummarization &&
		len(ev.LongRunningToolIDs) == 0 &&
		!hasFunctionCallOrResponse(ev.Content) &&
		isFinalResponse(ev) {
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
			if hitl, ok := hitlTools[fc.Name]; ok {
				pauseKind = hitl.pauseKind
				if fc.Name == workflow.WorkflowInputFunctionCallName {
					emittedInput = true
				}
				p.logSessionEvent(ctx, id, hitl.requestEvent, span, func(d *eventData) {
					applyEventBase(d)
					d.functionCallID = resolvedID
					var args any = nonNilMap(fc.Args)
					if fc.Name == hitlCredentialFunctionCallName {
						args, d.contentTruncated = p.sanitizeCredentialMap(fc.Args)
					}
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
				d.adkExtras = map[string]any{"pause_kind": pauseKind}
				var args any = nonNilMap(fc.Args)
				if fc.Name == hitlCredentialFunctionCallName {
					args, d.contentTruncated = p.sanitizeCredentialMap(fc.Args)
				}
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
			applyEventBase(d)
			d.functionCallID = fcCopy.ID
			var args any = nonNilMap(fcCopy.Args)
			if fcCopy.Name == hitlCredentialFunctionCallName {
				args, d.contentTruncated = p.sanitizeCredentialMap(fcCopy.Args)
			}
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
				applyEventBase(d)
				d.functionCallID = fr.ID
				var result any = nonNilMap(fr.Response)
				if fr.Name == hitlCredentialFunctionCallName {
					result, d.contentTruncated = p.sanitizeCredentialMap(fr.Response)
				}
				d.content = map[string]any{
					"tool_id": fr.ID,
					"tool":    fr.Name,
					"result":  result,
				}
			})
		}
	}
}
