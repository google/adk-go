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
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"log"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/agenttool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/loadmemorytool"
	"google.golang.org/adk/v2/tool/toolconfirmation"
	"google.golang.org/adk/v2/workflow"
)

func TestSessionEvents_EmitRichLifecycleTypes(t *testing.T) {
	t.Run("STATE_DELTA scrubs temp: keys case-sensitively and writes attributes.state_delta with NULL content", func(t *testing.T) {
		p, requests, mCtx := setupTestPlugin(t)
		var errs callbackErrors

		// Temp-only delta ("temp:" prefix) must be suppressed completely.
		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			ID:     "ev-temp-only",
			Author: "test-agent",
			Actions: session.EventActions{
				StateDelta: map[string]any{
					"temp:scratch": "ephemeral",
					"temp:token":   "secret",
				},
			},
		}))

		// Mixed delta must strip lowercase "temp:" keys and keep durable keys
		// (including uppercase "TEMP:" keys, matching ADK Go's case-sensitive KeyPrefixTemp).
		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			ID:     "ev-state-1",
			Author: "test-agent",
			Actions: session.EventActions{
				StateDelta: map[string]any{
					"temp:scratch": "ephemeral",
					"TEMP:durable": "kept",
					"counter":      42,
					"mode":         "active",
				},
			},
		}))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
		if diff := cmp.Diff([]string{EventStateDelta, EventInvocationCompleted}, eventTypes(rows)); diff != "" {
			t.Fatalf("event types mismatch (-want +got):\n%s", diff)
		}
		row := onlyRowOfType(t, rows, EventStateDelta)
		if gotContent, hasContent := row["content"]; hasContent {
			t.Errorf("STATE_DELTA content = %v, want absent/NULL", gotContent)
		}
		attrs := jsonObjectColumn(t, row, "attributes")
		// "TEMP:durable" is retained by scrubTempStateDelta (case-sensitive "temp:")
		// and its value is redacted by smartTruncate's sensitive-key check.
		checkJSONAt(t, attrs, `{"TEMP:durable":"[REDACTED]","counter":42,"mode":"active"}`, "state_delta")
		checkJSONAt(t, attrs, `"ev-state-1"`, "adk", "source_event_id")
	})

	t.Run("AGENT_TRANSFER", func(t *testing.T) {
		p, requests, mCtx := setupTestPlugin(t)
		var errs callbackErrors
		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			ID:      "ev-transfer-1",
			Author:  "router-agent",
			Actions: session.EventActions{TransferToAgent: "specialist-agent"},
		}))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
		row := onlyRowOfType(t, rows, EventAgentTransfer)
		checkJSON(t, "AGENT_TRANSFER content", jsonObjectColumn(t, row, "content"),
			`{"from_agent":"router-agent","to_agent":"specialist-agent"}`)
	})

	t.Run("EVENT_COMPACTION fractional epoch timestamps, formatResponseContent, and Actions.Compaction extraction", func(t *testing.T) {
		p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
			c.MaxContentLen = 40
			c.CustomMetadataAllowlist = []string{"adk_compaction"}
		})
		startTS := time.Unix(1700000000, 500_000_000).UTC()
		endTS := time.Unix(1700000300, 250_000_000).UTC()

		type compactionRecord struct {
			StartTimestamp   time.Time
			EndTimestamp     time.Time
			CompactedContent *genai.Content
		}

		var errs callbackErrors

		// 1. Application metadata under "compaction" (or non-compaction maps/structs) must NOT emit EVENT_COMPACTION.
		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			ID:     "ev-app-meta-ignored",
			Author: "test-agent",
			LLMResponse: model.LLMResponse{
				CustomMetadata: map[string]any{
					"compaction":     map[string]any{"policy": "aggressive", "max_turns": 20},
					"adk_compaction": map[string]any{"unrelated": "ignored"},
				},
			},
		}))
		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			ID:     "ev-app-meta-struct-ignored",
			Author: "test-agent",
			LLMResponse: model.LLMResponse{
				CustomMetadata: map[string]any{
					"adk_compaction": struct{ Unrelated string }{Unrelated: "noop"},
				},
			},
		}))
		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			ID:     "ev-app-meta-ptr-struct-ignored",
			Author: "test-agent",
			LLMResponse: model.LLMResponse{
				CustomMetadata: map[string]any{
					"adk_compaction": &struct{ Unrelated string }{Unrelated: "noop"},
				},
			},
		}))

		// 2. Real session.EventActions.Compaction (when present on session.EventActions) via OnEventCallback.
		evActionsCompact := &session.Event{
			ID:     "ev-actions-compact",
			Author: "test-agent",
		}
		hasTypedActionsCompaction := false
		if cf := reflect.ValueOf(&evActionsCompact.Actions).Elem().FieldByName("Compaction"); cf.IsValid() && cf.CanSet() {
			hasTypedActionsCompaction = true
			comp := reflect.New(cf.Type().Elem())
			comp.Elem().FieldByName("StartTimestamp").Set(reflect.ValueOf(startTS))
			comp.Elem().FieldByName("EndTimestamp").Set(reflect.ValueOf(endTS))
			comp.Elem().FieldByName("CompactedContent").Set(reflect.ValueOf(textContent("model", "short summary")))
			cf.Set(comp)
			errs.add(p.OnEventCallback()(mCtx, evActionsCompact))
		}

		// 3. Fallback via CustomMetadata["adk_compaction"] with truncation.
		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			ID:     "ev-compact-1",
			Author: "test-agent",
			LLMResponse: model.LLMResponse{
				CustomMetadata: map[string]any{
					"adk_compaction": &compactionRecord{
						StartTimestamp:   startTS,
						EndTimestamp:     endTS,
						CompactedContent: textContent("model", "this compacted summary is longer than 40 chars"),
					},
				},
			},
		}))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
		compRows := rowsOfType(rows, EventEventCompaction)
		wantCount := 1
		if hasTypedActionsCompaction {
			wantCount = 2
		}
		if len(compRows) != wantCount {
			t.Fatalf("got %d EVENT_COMPACTION rows, want %d", len(compRows), wantCount)
		}

		if hasTypedActionsCompaction {
			checkJSON(t, "Actions.Compaction EVENT_COMPACTION content", jsonObjectColumn(t, compRows[0], "content"),
				`{"start_timestamp":1700000000.5,"end_timestamp":1700000300.25,"compacted_content":"text: 'short summary'"}`)
			if got := compRows[0]["is_truncated"]; got != false {
				t.Errorf("Actions.Compaction is_truncated = %v, want false", got)
			}
		}

		metaRow := compRows[len(compRows)-1]
		checkJSON(t, "adk_compaction EVENT_COMPACTION content", jsonObjectColumn(t, metaRow, "content"),
			`{"start_timestamp":1700000000.5,"end_timestamp":1700000300.25,"compacted_content":"text: 'this compacted summary is longer ...[TRUNCATED]"}`)
		if got := metaRow["is_truncated"]; got != true {
			t.Errorf("is_truncated = %v, want true", got)
		}

		// Direct unit test for extractActionsCompaction on an Actions struct with Compaction.
		actionsWithCompaction := struct {
			Compaction *compactionRecord
		}{
			Compaction: &compactionRecord{
				StartTimestamp:   startTS,
				EndTimestamp:     endTS,
				CompactedContent: textContent("model", "short summary"),
			},
		}
		payload, trunc, ok := extractActionsCompaction(actionsWithCompaction, 100, true)
		if !ok || trunc {
			t.Fatalf("extractActionsCompaction() = (%v, %v, %v), want (payload, false, true)", payload, trunc, ok)
		}
		if got := payload["compacted_content"]; got != "text: 'short summary'" {
			t.Errorf("extractActionsCompaction compacted_content = %q, want %q", got, "text: 'short summary'")
		}
	})

	t.Run("AGENT_STATE_CHECKPOINT keys and terminal-only checkpoints", func(t *testing.T) {
		tests := []struct {
			name        string
			meta        map[string]any
			wantEmitted bool
			wantJSON    string
		}{
			{
				name: "adk_agent_state with end_of_agent",
				meta: map[string]any{
					"adk_agent_state": map[string]any{"step": 2, "cursor": "abc"},
					"end_of_agent":    true,
				},
				wantEmitted: true,
				wantJSON:    `{"agent_state":{"step":2,"cursor":"abc"},"end_of_agent":true}`,
			},
			{
				name: "alternate agent_state key",
				meta: map[string]any{
					"agent_state":  map[string]any{"step": 3},
					"end_of_agent": false,
				},
				wantEmitted: true,
				wantJSON:    `{"agent_state":{"step":3},"end_of_agent":false}`,
			},
			{
				name: "terminal checkpoint with nil state payload",
				meta: map[string]any{
					"agent_state":  nil,
					"end_of_agent": true,
				},
				wantEmitted: true,
				wantJSON:    `{"agent_state":null,"end_of_agent":true}`,
			},
			{
				name: "terminal checkpoint with adk_end_of_agent only",
				meta: map[string]any{
					"adk_end_of_agent": true,
				},
				wantEmitted: true,
				wantJSON:    `{"agent_state":null,"end_of_agent":true}`,
			},
			{
				name: "nil state and non-terminal does not emit",
				meta: map[string]any{
					"agent_state":  nil,
					"end_of_agent": false,
				},
				wantEmitted: false,
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
					c.CustomMetadataAllowlist = []string{"agent_state", "adk_agent_state", "adk_end_of_agent", "end_of_agent"}
				})
				var errs callbackErrors
				errs.add(p.OnEventCallback()(mCtx, &session.Event{
					ID:     "ev-ckpt",
					Author: "workflow-agent",
					LLMResponse: model.LLMResponse{
						CustomMetadata: tt.meta,
					},
				}))
				errs.check(t)
				p.AfterRunCallback()(mCtx)

				rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
				ckptRows := rowsOfType(rows, EventAgentStateCheckpoint)
				if !tt.wantEmitted {
					if len(ckptRows) != 0 {
						t.Fatalf("got %d AGENT_STATE_CHECKPOINT rows, want 0", len(ckptRows))
					}
					return
				}
				row := onlyRowOfType(t, rows, EventAgentStateCheckpoint)
				checkJSON(t, "AGENT_STATE_CHECKPOINT content", jsonObjectColumn(t, row, "content"), tt.wantJSON)
			})
		}
	})

	t.Run("HITL requests, credential scrubbing (camelCase+snake_case), TOOL_PAUSED, and HITL completions", func(t *testing.T) {
		p, requests, mCtx := setupTestPlugin(t)
		var errs callbackErrors

		// 1. Session event with LongRunningToolIDs matching HITL credential (with camelCase & snake_case secrets),
		// non-HITL tool, and orphan ID.
		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			ID:                 "ev-hitl-1",
			Author:             "test-agent",
			LongRunningToolIDs: []string{"call-cred", "call-pause", "call-orphan"},
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{FunctionCall: &genai.FunctionCall{
							ID:   "call-cred",
							Name: hitlCredentialFunctionCallName,
							Args: map[string]any{
								"auth_scheme":    "oauth2",
								"access_token":   "snake-secret",
								"auth_code":      "snake-auth-code-secret",
								"code":           "raw-oauth-code-secret",
								"code_verifier":  "snake-verifier-secret",
								"api_key_string": "snake-api-key-string-secret",
								"api_key_secret": "snake-api-key-secret-val",
								"redirect_uri":   "https://app.example.com/cb?code=uri-callback-code-secret&state=pub",
								"authConfig": map[string]any{
									"exchangedAuthCredential": map[string]any{
										"oauth2": map[string]any{
											"accessToken":  "camel-access-secret",
											"refreshToken": "camel-refresh-secret",
											"clientSecret": "camel-client-secret",
											"authCode":     "camel-auth-code-secret",
											"codeVerifier": "camel-verifier-secret",
											"apiKeyString": "camel-api-key-string-secret",
											"apiKeySecret": "camel-api-key-secret-val",
											"clientId":     "public-client-id",
										},
									},
								},
							},
						}},
						{FunctionCall: &genai.FunctionCall{
							ID:   "call-pause",
							Name: "deploy_cluster",
							Args: map[string]any{"region": "us-central1", "code": "E_OK"},
						}},
					},
				},
			},
		}))

		// 2a. Runtime confirmation flow step 1: tool-response event carrying Actions.RequestedToolConfirmations
		// (plus RequestedInput). RequestedToolConfirmations alone must NOT emit a duplicate HITL_CONFIRMATION_REQUEST.
		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			ID:     "ev-hitl-2a",
			Author: "test-agent",
			Actions: session.EventActions{
				RequestedToolConfirmations: map[string]toolconfirmation.ToolConfirmation{
					"orig-call-1": {Hint: "Approve deletion?"},
				},
			},
			RequestedInput: &session.RequestInput{
				InterruptID: "int-1",
				Message:     "Which environment?",
			},
		}))

		// 2b. Runtime confirmation flow step 2: synthesized adk_request_confirmation FunctionCall event
		// with its call ID in LongRunningToolIDs. Emits one HITL_CONFIRMATION_REQUEST and one TOOL_PAUSED.
		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			ID:                 "ev-hitl-2b",
			Author:             "test-agent",
			LongRunningToolIDs: []string{"call-confirm-1"},
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{FunctionCall: &genai.FunctionCall{
							ID:   "call-confirm-1",
							Name: toolconfirmation.FunctionCallName,
							Args: map[string]any{"hint": "Approve deletion?"},
						}},
					},
				},
			},
		}))

		// 3. Session event with HITL FunctionResponse completion containing camelCase & snake_case secrets.
		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			ID:     "ev-hitl-3",
			Author: "test-agent",
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Role: "user",
					Parts: []*genai.Part{
						{FunctionResponse: &genai.FunctionResponse{
							ID:   "call-cred",
							Name: hitlCredentialFunctionCallName,
							Response: map[string]any{
								"status":       "granted",
								"accessToken":  "granted-camel-secret",
								"refreshToken": "granted-refresh-secret",
								"api_key":      "granted-snake-secret",
								"auth_code":    "granted-auth-code-secret",
								"code":         "granted-raw-code-secret",
								"codeVerifier": "granted-verifier-secret",
								"apiKeyString": "granted-api-key-string-secret",
								"apiKeySecret": "granted-api-key-secret-val",
							},
						}},
					},
				},
			},
		}))

		// 4. User message carrying HITL confirmation response and resumed tool response.
		errs.add(p.OnUserMessageCallback()(mCtx, &genai.Content{
			Role: "user",
			Parts: []*genai.Part{
				{FunctionResponse: &genai.FunctionResponse{
					ID:       "call-confirm-1",
					Name:     toolconfirmation.FunctionCallName,
					Response: map[string]any{"confirmed": true},
				}},
				{FunctionResponse: &genai.FunctionResponse{
					ID:       "call-pause",
					Name:     "deploy_cluster",
					Response: map[string]any{"cluster_id": "c-123", "code": "E_OK"},
				}},
			},
		}))

		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
		wantTypes := []string{
			EventHITLCredentialRequest,
			EventToolPaused, // hitl_credential
			EventToolPaused, // deploy_cluster ("tool")
			EventToolPaused, // call-orphan ("tool")
			EventHITLInputRequest,
			EventHITLConfirmationRequest,
			EventToolPaused, // hitl_confirmation
			EventHITLCredentialCompleted,
			EventUserMessageReceived,
			EventHITLConfirmationCompleted,
			EventToolCompleted,
			EventInvocationCompleted,
		}
		if diff := cmp.Diff(wantTypes, eventTypes(rows)); diff != "" {
			t.Fatalf("event types mismatch (-want +got):\n%s", diff)
		}

		wantScrubbedArgs := `{"tool_id":"call-cred","tool":"adk_request_credential","args":{"auth_scheme":"oauth2","access_token":"[REDACTED]","auth_code":"[REDACTED]","code":"raw-oauth-code-secret","code_verifier":"[REDACTED]","api_key_string":"[REDACTED]","api_key_secret":"[REDACTED]","redirect_uri":"https://app.example.com/cb?code=uri-callback-code-secret&state=pub","authConfig":{"exchangedAuthCredential":{"oauth2":{"accessToken":"[REDACTED]","refreshToken":"[REDACTED]","clientSecret":"[REDACTED]","authCode":"[REDACTED]","codeVerifier":"[REDACTED]","apiKeyString":"[REDACTED]","apiKeySecret":"[REDACTED]","clientId":"public-client-id"}}}}}`

		credReq := onlyRowOfType(t, rows, EventHITLCredentialRequest)
		checkJSON(t, "HITL_CREDENTIAL_REQUEST content", jsonObjectColumn(t, credReq, "content"), wantScrubbedArgs)
		checkJSONAt(t, jsonObjectColumn(t, credReq, "attributes"), `"call-cred"`, "adk", "function_call_id")

		confirmReq := onlyRowOfType(t, rows, EventHITLConfirmationRequest)
		checkJSON(t, "HITL_CONFIRMATION_REQUEST content", jsonObjectColumn(t, confirmReq, "content"),
			`{"tool_id":"call-confirm-1","tool":"adk_request_confirmation","args":{"hint":"Approve deletion?"}}`)

		pausedRows := rowsOfType(rows, EventToolPaused)
		if len(pausedRows) != 4 {
			t.Fatalf("got %d TOOL_PAUSED rows, want 4", len(pausedRows))
		}
		checkJSON(t, "TOOL_PAUSED hitl_credential content", jsonObjectColumn(t, pausedRows[0], "content"), wantScrubbedArgs)
		checkJSONAt(t, jsonObjectColumn(t, pausedRows[0], "attributes"), `"hitl_credential"`, "adk", "pause_kind")
		checkJSONAt(t, jsonObjectColumn(t, pausedRows[0], "attributes"), `"call-cred"`, "adk", "function_call_id")

		checkJSON(t, "TOOL_PAUSED matched tool content", jsonObjectColumn(t, pausedRows[1], "content"),
			`{"tool_id":"call-pause","tool":"deploy_cluster","args":{"region":"us-central1","code":"E_OK"}}`)
		checkJSONAt(t, jsonObjectColumn(t, pausedRows[1], "attributes"), `"tool"`, "adk", "pause_kind")
		checkJSONAt(t, jsonObjectColumn(t, pausedRows[1], "attributes"), `"call-pause"`, "adk", "function_call_id")

		checkJSON(t, "TOOL_PAUSED orphan content", jsonObjectColumn(t, pausedRows[2], "content"),
			`{"tool_id":"call-orphan","tool":null,"args":null}`)
		checkJSONAt(t, jsonObjectColumn(t, pausedRows[2], "attributes"), `"tool"`, "adk", "pause_kind")

		checkJSON(t, "TOOL_PAUSED hitl_confirmation content", jsonObjectColumn(t, pausedRows[3], "content"),
			`{"tool_id":"call-confirm-1","tool":"adk_request_confirmation","args":{"hint":"Approve deletion?"}}`)
		checkJSONAt(t, jsonObjectColumn(t, pausedRows[3], "attributes"), `"hitl_confirmation"`, "adk", "pause_kind")
		checkJSONAt(t, jsonObjectColumn(t, pausedRows[3], "attributes"), `"call-confirm-1"`, "adk", "function_call_id")

		credCompleted := onlyRowOfType(t, rows, EventHITLCredentialCompleted)
		checkJSON(t, "HITL_CREDENTIAL_REQUEST_COMPLETED content", jsonObjectColumn(t, credCompleted, "content"),
			`{"tool_id":"call-cred","tool":"adk_request_credential","result":{"status":"granted","accessToken":"[REDACTED]","refreshToken":"[REDACTED]","api_key":"[REDACTED]","auth_code":"[REDACTED]","code":"granted-raw-code-secret","codeVerifier":"[REDACTED]","apiKeyString":"[REDACTED]","apiKeySecret":"[REDACTED]"}}`)

		confirmCompleted := onlyRowOfType(t, rows, EventHITLConfirmationCompleted)
		checkJSON(t, "HITL_CONFIRMATION_COMPLETED content", jsonObjectColumn(t, confirmCompleted, "content"),
			`{"tool_id":"call-confirm-1","tool":"adk_request_confirmation","result":{"confirmed":true}}`)

		resumedTool := onlyRowOfType(t, rows, EventToolCompleted)
		checkJSON(t, "resumed TOOL_COMPLETED content", jsonObjectColumn(t, resumedTool, "content"),
			`{"tool":"deploy_cluster","result":{"cluster_id":"c-123","code":"E_OK"},"tool_origin":"UNKNOWN"}`)
		checkJSONAt(t, jsonObjectColumn(t, resumedTool, "attributes"), `"tool"`, "adk", "pause_kind")
		checkJSONAt(t, jsonObjectColumn(t, resumedTool, "attributes"), `"call-pause"`, "adk", "function_call_id")
	})

	t.Run("HITL credential scrubbing redacts typed maps, struct slices, and cyclic payloads while propagating is_truncated", func(t *testing.T) {
		p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
			c.MaxContentLen = 20
		})
		var errs callbackErrors

		type credTokenStruct struct {
			AccessToken  string `json:"accessToken"`
			RefreshToken string `json:"refreshToken"`
			ClientID     string `json:"clientId"`
		}

		cyclicArgs := map[string]any{
			"accessToken": "cycle-args-secret",
			"note":        strings.Repeat("a", 50),
		}
		cyclicArgs["oauth2"] = cyclicArgs

		cyclicResp := map[string]any{
			"refreshToken": "cycle-resp-secret",
			"detail":       strings.Repeat("b", 50),
		}
		cyclicResp["self"] = cyclicResp

		// 1. Long-running credential request with typed map, typed struct slice, and cyclic map.
		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			ID:                 "ev-cred-typed-1",
			Author:             "test-agent",
			LongRunningToolIDs: []string{"call-cred-typed"},
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{FunctionCall: &genai.FunctionCall{
							ID:   "call-cred-typed",
							Name: hitlCredentialFunctionCallName,
							Args: map[string]any{
								"typedMap": map[string]string{
									"accessToken": "typed-map-secret",
									"clientId":    "public-id",
								},
								"tokens": []credTokenStruct{
									{
										AccessToken:  "struct-access-secret",
										RefreshToken: "struct-refresh-secret",
										ClientID:     "pub",
									},
								},
								"cyclic": cyclicArgs,
							},
						}},
					},
				},
			},
		}))

		// 2. Non-long-running credential request and session-event completion with cyclic/typed payloads.
		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			ID:     "ev-cred-typed-2",
			Author: "test-agent",
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{FunctionCall: &genai.FunctionCall{
							ID:   "call-cred-unmatched",
							Name: hitlCredentialFunctionCallName,
							Args: map[string]any{
								"cyclic": cyclicArgs,
							},
						}},
						{FunctionResponse: &genai.FunctionResponse{
							ID:   "call-cred-unmatched",
							Name: hitlCredentialFunctionCallName,
							Response: map[string]any{
								"tokens": []credTokenStruct{
									{AccessToken: "resp-struct-secret", ClientID: "pub-resp"},
								},
								"cyclic": cyclicResp,
							},
						}},
					},
				},
			},
		}))

		// 3. User-message credential completion with camelCase accessToken, cyclic, and typed payloads.
		errs.add(p.OnUserMessageCallback()(mCtx, &genai.Content{
			Role: "user",
			Parts: []*genai.Part{
				{FunctionResponse: &genai.FunctionResponse{
					ID:   "call-cred-typed",
					Name: hitlCredentialFunctionCallName,
					Response: map[string]any{
						"accessToken": "user-msg-access-token-secret",
						"typedMap":    map[string]string{"apiKey": "user-msg-api-key-secret"},
						"cyclic":      cyclicResp,
					},
				}},
			},
		}))

		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
		wantTypes := []string{
			EventHITLCredentialRequest,
			EventToolPaused,
			EventHITLCredentialRequest,
			EventHITLCredentialCompleted,
			EventUserMessageReceived,
			EventHITLCredentialCompleted,
			EventInvocationCompleted,
		}
		if diff := cmp.Diff(wantTypes, eventTypes(rows)); diff != "" {
			t.Fatalf("event types mismatch (-want +got):\n%s", diff)
		}

		secrets := []string{
			"typed-map-secret",
			"struct-access-secret",
			"struct-refresh-secret",
			"cycle-args-secret",
			"cycle-resp-secret",
			"resp-struct-secret",
			"user-msg-access-token-secret",
			"user-msg-api-key-secret",
		}
		for _, r := range rows {
			evType, _ := r["event_type"].(string)
			if evType == EventInvocationCompleted {
				continue
			}
			rawRowBytes, err := json.Marshal(r)
			if err != nil {
				t.Fatalf("json.Marshal(row %s): %v", evType, err)
			}
			rawRow := string(rawRowBytes)
			for _, secret := range secrets {
				if strings.Contains(rawRow, secret) {
					t.Errorf("%s leaked secret %q in row: %s", evType, secret, rawRow)
				}
			}
			if !strings.Contains(rawRow, "[CIRCULAR_REFERENCE]") {
				t.Errorf("%s row = %s, want [CIRCULAR_REFERENCE] placeholder", evType, rawRow)
			}
			if got := r["is_truncated"]; got != true {
				t.Errorf("%s is_truncated = %v, want true", evType, got)
			}
		}

		reqRows := rowsOfType(rows, EventHITLCredentialRequest)
		reqContent := jsonObjectColumn(t, reqRows[0], "content")
		checkJSONAt(t, reqContent, `{"accessToken":"[REDACTED]","clientId":"public-id"}`, "args", "typedMap")
		checkJSONAt(t, reqContent, `[{"accessToken":"[REDACTED]","refreshToken":"[REDACTED]","clientId":"pub"}]`, "args", "tokens")
		checkJSONAt(t, reqContent, `"[CIRCULAR_REFERENCE]"`, "args", "cyclic", "oauth2")

		completedRows := rowsOfType(rows, EventHITLCredentialCompleted)
		userMsgCompletedContent := jsonObjectColumn(t, completedRows[1], "content")
		checkJSONAt(t, userMsgCompletedContent, `"[REDACTED]"`, "result", "accessToken")
		checkJSONAt(t, userMsgCompletedContent, `{"apiKey":"[REDACTED]"}`, "result", "typedMap")
	})

	t.Run("HITL adk_request_input LongRunningToolIDs, single ID-less call fallback, and non-long-running HITL calls", func(t *testing.T) {
		p, requests, mCtx := setupTestPlugin(t)
		var errs callbackErrors

		// 1. adk_request_input with empty FunctionCall.ID matched via single-call fallback in LongRunningToolIDs,
		// plus ev.RequestedInput on the same event (must NOT double-emit HITL_INPUT_REQUEST).
		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			ID:                 "ev-input-lr",
			Author:             "workflow-agent",
			LongRunningToolIDs: []string{"resolved-input-1"},
			RequestedInput: &session.RequestInput{
				InterruptID: "int-dup",
				Message:     "Enter approval code",
			},
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{FunctionCall: &genai.FunctionCall{
							ID:   "",
							Name: workflow.WorkflowInputFunctionCallName,
							Args: map[string]any{"prompt": "Enter approval code"},
						}},
					},
				},
			},
		}))

		// 2. HITL FunctionCall without LongRunningToolIDs emits HITL_CONFIRMATION_REQUEST without TOOL_PAUSED.
		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			ID:     "ev-hitl-no-lr",
			Author: "workflow-agent",
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{FunctionCall: &genai.FunctionCall{
							ID:   "call-confirm-no-lr",
							Name: toolconfirmation.FunctionCallName,
							Args: map[string]any{"hint": "Continue?"},
						}},
					},
				},
			},
		}))

		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
		wantTypes := []string{
			EventHITLInputRequest,
			EventToolPaused,
			EventHITLConfirmationRequest,
			EventInvocationCompleted,
		}
		if diff := cmp.Diff(wantTypes, eventTypes(rows)); diff != "" {
			t.Fatalf("event types mismatch (-want +got):\n%s", diff)
		}

		inputReq := onlyRowOfType(t, rows, EventHITLInputRequest)
		checkJSON(t, "HITL_INPUT_REQUEST content", jsonObjectColumn(t, inputReq, "content"),
			`{"tool_id":"resolved-input-1","tool":"adk_request_input","args":{"prompt":"Enter approval code"}}`)
		paused := onlyRowOfType(t, rows, EventToolPaused)
		checkJSONAt(t, jsonObjectColumn(t, paused, "attributes"), `"hitl_input"`, "adk", "pause_kind")
		checkJSONAt(t, jsonObjectColumn(t, paused, "attributes"), `"resolved-input-1"`, "adk", "function_call_id")
	})

	t.Run("A2A_INTERACTION, NODE_OUTPUT, NODE_ERROR, and multi-signal orthogonality", func(t *testing.T) {
		p, requests, mCtx := setupTestPlugin(t)
		var errs callbackErrors

		// 1. Correlation-only A2A metadata (without a2a:request or a2a:response) must NOT emit A2A_INTERACTION,
		// and MessageAsOutput=true must NOT emit NODE_OUTPUT.
		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			ID:       "ev-corr-only",
			Author:   "remote-worker",
			NodeInfo: &session.NodeInfo{Path: "wf@0/msg@0", MessageAsOutput: true},
			Output:   textContent("model", "message output"),
			LLMResponse: model.LLMResponse{
				Content: textContent("model", "message output"),
				CustomMetadata: map[string]any{
					"a2a:task_id":    "task-corr-only",
					"a2a:context_id": "ctx-corr-only",
				},
			},
		}))

		// 1b. NodeInfo with empty Path must NOT emit NODE_ERROR or NODE_OUTPUT.
		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			ID:       "ev-empty-node-path",
			Author:   "remote-worker",
			NodeInfo: &session.NodeInfo{Path: ""},
			Output:   map[string]any{"ignored": true},
			LLMResponse: model.LLMResponse{
				ErrorCode:    "DEADLINE_EXCEEDED",
				ErrorMessage: "node with empty path",
			},
		}))

		// 2. Single event with multiple orthogonal signals: StateDelta, TransferToAgent,
		// A2A interaction payload (with non-nil ev.Content, verifying A2A content uses protocol response),
		// NodeInfo output + node failure on the same event, and final AGENT_RESPONSE text.
		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			ID:       "ev-multi-1",
			Author:   "remote-worker",
			Branch:   "root.remote-worker",
			NodeInfo: &session.NodeInfo{Path: "wf@0/step@1"},
			Output:   map[string]any{"node_result": "partial_done"},
			Actions: session.EventActions{
				StateDelta:      map[string]any{"progress": "100%"},
				TransferToAgent: "reviewer-agent",
			},
			LLMResponse: model.LLMResponse{
				Content:      textContent("model", "all tasks completed"),
				ErrorCode:    "DEADLINE_EXCEEDED",
				ErrorMessage: "node failed with Authorization: Bearer secret-node-token",
				CustomMetadata: map[string]any{
					"a2a:task_id":    "task-99",
					"a2a:context_id": "ctx-42",
					"a2a:response":   map[string]any{"protocol_status": "completed"},
					"a2a:custom":     "colon-val",
					"a2a_underscore": "ignored",
					"non_a2a_key":    "ignored",
				},
			},
		}))

		// 3. LLM termination/blocking code (e.g. SAFETY) on a workflow node must NOT emit NODE_ERROR,
		// and AGENT_RESPONSE must remain status="OK".
		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			ID:       "ev-node-safety",
			Author:   "remote-worker",
			NodeInfo: &session.NodeInfo{Path: "wf@0/step@2"},
			LLMResponse: model.LLMResponse{
				Content:      textContent("model", "safety blocked response"),
				ErrorCode:    string(genai.FinishReasonSafety),
				ErrorMessage: "Blocked by safety filter",
			},
		}))

		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
		wantTypes := []string{
			EventAgentResponse, // from ev-corr-only (no A2A_INTERACTION, no NODE_OUTPUT)
			EventStateDelta,
			EventAgentTransfer,
			EventA2AInteraction,
			EventNodeError,
			EventNodeOutput,
			EventAgentResponse, // from ev-multi-1
			EventAgentResponse, // from ev-node-safety (no NODE_ERROR)
			EventInvocationCompleted,
		}
		if diff := cmp.Diff(wantTypes, eventTypes(rows)); diff != "" {
			t.Fatalf("event types mismatch (-want +got):\n%s", diff)
		}

		a2aRow := onlyRowOfType(t, rows, EventA2AInteraction)
		checkJSONAt(t, jsonObjectColumn(t, a2aRow, "attributes"),
			`{"a2a:task_id":"task-99","a2a:context_id":"ctx-42","a2a:response":{"protocol_status":"completed"},"a2a:custom":"colon-val"}`, "a2a_metadata")
		checkJSON(t, "A2A protocol response content", jsonObjectColumn(t, a2aRow, "content"),
			`{"protocol_status":"completed"}`)

		nodeOut := onlyRowOfType(t, rows, EventNodeOutput)
		checkJSON(t, "NODE_OUTPUT content", jsonObjectColumn(t, nodeOut, "content"),
			`{"node_result":"partial_done"}`)

		nodeErr := onlyRowOfType(t, rows, EventNodeError)
		if got := nodeErr["status"]; got != "ERROR" {
			t.Errorf("NODE_ERROR status = %v, want ERROR", got)
		}
		wantRedactedErr := "node failed with Authorization: [REDACTED]"
		if got := nodeErr["error_message"]; got != wantRedactedErr {
			t.Errorf("NODE_ERROR error_message = %v, want %q", got, wantRedactedErr)
		}
		// NODE_ERROR content must NOT duplicate raw error_message or leak Bearer tokens.
		checkJSON(t, "NODE_ERROR content", jsonObjectColumn(t, nodeErr, "content"),
			`{"node_path":"wf@0/step@1","error_code":"DEADLINE_EXCEEDED"}`)
		if contentStr, _ := nodeErr["content"].(string); strings.Contains(contentStr, "secret-node-token") {
			t.Errorf("NODE_ERROR content leaked bearer token: %s", contentStr)
		}

		// Verify all AGENT_RESPONSE rows have status="OK", even when ev.ErrorCode was set.
		for _, respRow := range rowsOfType(rows, EventAgentResponse) {
			if got := respRow["status"]; got != "OK" {
				t.Errorf("AGENT_RESPONSE status = %v, want OK", got)
			}
		}
	})

	t.Run("streaming partial events do not emit A2A_INTERACTION, NODE_OUTPUT, or NODE_ERROR", func(t *testing.T) {
		p, requests, mCtx := setupTestPlugin(t)
		var errs callbackErrors

		// Partial event carrying NodeInfo, Output, ErrorCode/ErrorMessage, and a2a:* metadata
		// must not emit A2A_INTERACTION, NODE_OUTPUT, or NODE_ERROR.
		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			ID:       "ev-partial-signals",
			Author:   "remote-worker",
			NodeInfo: &session.NodeInfo{Path: "wf@0/step@1"},
			Output:   map[string]any{"chunk": "data"},
			LLMResponse: model.LLMResponse{
				Partial:      true,
				ErrorCode:    "DEADLINE_EXCEEDED",
				ErrorMessage: "stream interrupted",
				CustomMetadata: map[string]any{
					"a2a:task_id":  "task-stream-1",
					"a2a:response": map[string]any{"chunk": 1},
				},
			},
		}))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
		if diff := cmp.Diff([]string{EventInvocationCompleted}, eventTypes(rows)); diff != "" {
			t.Fatalf("event types mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestSessionEvents_LegacyEventFiltering(t *testing.T) {
	allNonSessionTypes := []string{
		EventUserMessageReceived,
		EventToolStarting,
		EventToolCompleted,
		EventInvocationCompleted,
	}
	tests := []struct {
		name      string
		denylist  []string
		allowlist []string
		wantTypes []string
	}{
		{
			name:     "legacy EVENT in EventDenylist suppresses session-event rows across all emitters",
			denylist: []string{"EVENT"},
			wantTypes: []string{
				EventUserMessageReceived,
				EventToolStarting,
				EventToolCompleted,
				EventAgentResponse,
				EventInvocationCompleted,
			},
		},
		{
			name:      "legacy EVENT in EventAllowlist permits session-event rows across all emitters",
			allowlist: append([]string{"EVENT", EventAgentResponse}, allNonSessionTypes...),
			wantTypes: []string{
				EventUserMessageReceived,
				EventHITLCredentialCompleted,
				EventToolCompleted,
				EventStateDelta,
				EventAgentResponse,
				EventHITLConfirmationCompleted,
				EventToolStarting,
				EventToolCompleted,
				EventAgentResponse,
				EventInvocationCompleted,
			},
		},
		{
			name:      "per-type denylist takes precedence over legacy EVENT allowlist",
			allowlist: append([]string{"EVENT"}, allNonSessionTypes...),
			denylist:  []string{EventAgentResponse, EventHITLCredentialCompleted},
			wantTypes: []string{
				EventUserMessageReceived,
				EventToolCompleted,
				EventStateDelta,
				EventHITLConfirmationCompleted,
				EventToolStarting,
				EventToolCompleted,
				EventInvocationCompleted,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
				c.EventDenylist = tt.denylist
				c.EventAllowlist = tt.allowlist
				c.FinalResponseToolNames = []string{"final_answer"}
			})
			var errs callbackErrors
			// 1. User message carrying both a HITL credential completion and a resumed tool completion.
			errs.add(p.OnUserMessageCallback()(mCtx, &genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{
						FunctionResponse: &genai.FunctionResponse{
							ID:       "call-cred-1",
							Name:     hitlCredentialFunctionCallName,
							Response: map[string]any{"ok": true},
						},
					},
					{
						FunctionResponse: &genai.FunctionResponse{
							ID:       "call-resumed-1",
							Name:     "fetch_data",
							Response: map[string]any{"rows": 3},
						},
					},
				},
			}))
			// 2a. Final session event carrying StateDelta and visible text.
			errs.add(p.OnEventCallback()(mCtx, &session.Event{
				ID:     "ev-filter-1",
				Author: "test-agent",
				Actions: session.EventActions{
					StateDelta: map[string]any{"k": "v"},
				},
				LLMResponse: model.LLMResponse{
					Content: textContent("model", "hello"),
				},
			}))
			// 2b. Session event carrying HITL confirmation completion.
			errs.add(p.OnEventCallback()(mCtx, &session.Event{
				ID:     "ev-filter-2",
				Author: "test-agent",
				LLMResponse: model.LLMResponse{
					Content: &genai.Content{
						Role: "user",
						Parts: []*genai.Part{
							{
								FunctionResponse: &genai.FunctionResponse{
									ID:       "call-confirm-1",
									Name:     toolconfirmation.FunctionCallName,
									Response: map[string]any{"confirmed": true},
								},
							},
						},
					},
				},
			}))
			// 3. FinalResponseToolNames tool call emitting TOOL_STARTING, TOOL_COMPLETED, and AGENT_RESPONSE.
			finalCtx := withFunctionCall(mCtx, "call-final-1")
			finalTool := &fakeTool{name: "final_answer"}
			errs.add(p.BeforeToolCallback()(finalCtx, finalTool, map[string]any{"q": "hi"}))
			errs.add(p.AfterToolCallback()(finalCtx, finalTool, map[string]any{"q": "hi"}, map[string]any{"answer": "done"}, nil))
			errs.check(t)
			p.AfterRunCallback()(mCtx)

			rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
			if diff := cmp.Diff(tt.wantTypes, eventTypes(rows)); diff != "" {
				t.Fatalf("event types mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestAgentResponse_ContentFormatterAndNilParts(t *testing.T) {
	t.Run("ContentFormatter scrubbing does not leak raw text into content_parts", func(t *testing.T) {
		p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
			c.ContentFormatter = func(payload any, eventType string) (any, error) {
				if eventType == EventAgentResponse {
					return map[string]any{
						"response": "[REDACTED]",
					}, nil
				}
				return payload, nil
			}
		})
		var errs callbackErrors
		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			ID:          "ev-fmt-1",
			Author:      "test-agent",
			LLMResponse: model.LLMResponse{Content: textContent("model", "top-secret-value")},
		}))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
		row := onlyRowOfType(t, rows, EventAgentResponse)
		checkJSON(t, "formatted AGENT_RESPONSE content", jsonObjectColumn(t, row, "content"),
			`{"response":"[REDACTED]"}`)
		if parts, ok := row["content_parts"]; ok && len(parts.([]map[string]any)) > 0 {
			for _, part := range parts.([]map[string]any) {
				if text, _ := part["text"].(string); strings.Contains(text, "top-secret-value") {
					t.Errorf("content_parts leaked unformatted text: %v", parts)
				}
			}
		}
	})

	t.Run("nil entries at beginning and end of Parts slice do not panic", func(t *testing.T) {
		p, requests, mCtx := setupTestPlugin(t)
		var errs callbackErrors

		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			ID:     "ev-nil-start",
			Author: "test-agent",
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Role:  "model",
					Parts: []*genai.Part{nil, {Text: "leading nil"}},
				},
			},
		}))
		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			ID:     "ev-nil-end",
			Author: "test-agent",
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Role:  "model",
					Parts: []*genai.Part{{Text: "trailing nil"}, nil},
				},
			},
		}))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
		respRows := rowsOfType(rows, EventAgentResponse)
		if len(respRows) != 2 {
			t.Fatalf("got %d AGENT_RESPONSE rows, want 2", len(respRows))
		}
		checkJSON(t, "leading nil content", jsonObjectColumn(t, respRows[0], "content"),
			`{"response":"text: 'leading nil'"}`)
		checkJSON(t, "trailing nil content", jsonObjectColumn(t, respRows[1], "content"),
			`{"response":"text: 'trailing nil'"}`)
	})
}

func TestAgentResponse_SkipsThoughtOnlyAndPartialEvents(t *testing.T) {
	p, requests, mCtx := setupTestPlugin(t)
	var errs callbackErrors

	// 1. Partial event -> skipped.
	errs.add(p.OnEventCallback()(mCtx, &session.Event{
		ID:     "ev-partial",
		Author: "test-agent",
		LLMResponse: model.LLMResponse{
			Partial: true,
			Content: textContent("model", "streaming chunk"),
		},
	}))

	// 2. User-authored event -> skipped.
	errs.add(p.OnEventCallback()(mCtx, &session.Event{
		ID:          "ev-user",
		Author:      "user",
		LLMResponse: model.LLMResponse{Content: textContent("user", "user text")},
	}))

	// 3. Thought-only event -> skipped.
	errs.add(p.OnEventCallback()(mCtx, &session.Event{
		ID:     "ev-thought-only",
		Author: "test-agent",
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{
				Role:  "model",
				Parts: []*genai.Part{{Text: "internal chain of thought", Thought: true}},
			},
		},
	}))

	// 4. Event with FunctionCall -> skipped.
	errs.add(p.OnEventCallback()(mCtx, &session.Event{
		ID:     "ev-tool-call",
		Author: "test-agent",
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					{Text: "calling tool"},
					{FunctionCall: &genai.FunctionCall{Name: "lookup"}},
				},
			},
		},
	}))

	// 5. Response-only SkipSummarization acknowledgement -> skipped (no visible text).
	errs.add(p.OnEventCallback()(mCtx, &session.Event{
		ID:      "ev-skip-summary",
		Author:  "test-agent",
		Actions: session.EventActions{SkipSummarization: true},
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					{
						FunctionResponse: &genai.FunctionResponse{
							ID:       "call-ack-1",
							Name:     "lookup",
							Response: map[string]any{"ok": true},
						},
					},
				},
			},
		},
	}))

	// 5b. SkipSummarization event with visible text (such as agenttool display) -> emits AGENT_RESPONSE.
	errs.add(p.OnEventCallback()(mCtx, &session.Event{
		ID:      "ev-skip-summary-visible",
		Author:  "test-agent",
		Actions: session.EventActions{SkipSummarization: true},
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					{
						FunctionResponse: &genai.FunctionResponse{
							ID:       "call-agent-1",
							Name:     "sub_agent",
							Response: map[string]any{"result": "sub-agent answer"},
						},
					},
					{Text: "sub-agent answer"},
				},
			},
		},
	}))

	// 6. Final response mixing Thought and visible Text -> emits only visible Text.
	errs.add(p.OnEventCallback()(mCtx, &session.Event{
		ID:     "ev-final-mixed",
		Author: "test-agent",
		Branch: "root.test-agent",
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					{Text: "hidden reasoning", Thought: true},
					{Text: "visible final answer"},
				},
			},
		},
	}))

	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	if diff := cmp.Diff([]string{EventAgentResponse, EventAgentResponse, EventInvocationCompleted}, eventTypes(rows)); diff != "" {
		t.Fatalf("event types mismatch (-want +got):\n%s", diff)
	}
	respRows := rowsOfType(rows, EventAgentResponse)
	checkJSON(t, "SkipSummarization AGENT_RESPONSE content", jsonObjectColumn(t, respRows[0], "content"),
		`{"response":"text: 'sub-agent answer'"}`)
	row := respRows[1]
	checkJSON(t, "AGENT_RESPONSE content", jsonObjectColumn(t, row, "content"),
		`{"response":"text: 'visible final answer'"}`)
	gotParts, _ := row["content_parts"].([]map[string]any)
	wantParts := []map[string]any{textPart("visible final answer", 0)}
	if diff := cmp.Diff(wantParts, gotParts); diff != "" {
		t.Errorf("content_parts mismatch (-want +got):\n%s", diff)
	}
}

func TestADKEnvelope_ExtrasCannotOverwriteCoreKeys(t *testing.T) {
	ap := &analyticsPlugin{config: applyDefaults(DefaultConfig()), traces: newTraceManager()}
	id := identity{
		agentName:    "agent-name",
		sessionID:    "sess-id",
		invocationID: "inv-id",
		userID:       "user-id",
		appName:      "test-app",
		branch:       "root.agent-name",
	}
	d := &eventData{
		author:         "event-author",
		functionCallID: "call-core",
		adkExtras: map[string]any{
			"schema_version":   "999",
			"event_id":         "spoofed-id",
			"app_name":         "spoofed-app",
			"invocation_id":    "spoofed-inv",
			"author":           "spoofed-author",
			"function_call_id": "spoofed-call",
			"pause_kind":       "hitl_confirmation",
		},
	}
	adk := ap.adkEnvelope("real-event-id", id, "agent-name", d)
	if got := adk["schema_version"]; got != adkEnvelopeSchemaVersion {
		t.Errorf("adk[schema_version] = %v, want %q", got, adkEnvelopeSchemaVersion)
	}
	if got := adk["event_id"]; got != "real-event-id" {
		t.Errorf("adk[event_id] = %v, want %q", got, "real-event-id")
	}
	if got := adk["app_name"]; got != "test-app" {
		t.Errorf("adk[app_name] = %v, want %q", got, "test-app")
	}
	if got := adk["invocation_id"]; got != "inv-id" {
		t.Errorf("adk[invocation_id] = %v, want %q", got, "inv-id")
	}
	if got := adk["author"]; got != "event-author" {
		t.Errorf("adk[author] = %v, want %q", got, "event-author")
	}
	if got := adk["function_call_id"]; got != "call-core" {
		t.Errorf("adk[function_call_id] = %v, want %q", got, "call-core")
	}
	if got := adk["pause_kind"]; got != "hitl_confirmation" {
		t.Errorf("adk[pause_kind] = %v, want %q", got, "hitl_confirmation")
	}
}

type singleToolset struct {
	name string
	t    tool.Tool
}

func (s singleToolset) Name() string { return s.name }

func (s singleToolset) Tools(_ agent.ReadonlyContext) ([]tool.Tool, error) {
	return []tool.Tool{s.t}, nil
}

type unwrapMethodTool struct {
	fakeTool
	inner tool.Tool
}

func (t *unwrapMethodTool) Unwrap() tool.Tool {
	return t.inner
}

type fakeFinishTaskTool struct {
	fakeTool
}

type confirmationWrapperTool struct {
	fakeTool
	runnableTool tool.Tool
}

type exportedWrapperTool struct {
	fakeTool
	RunnableTool tool.Tool
}

type transferContextWithAgent struct {
	*mockInvocationContext
	rootAgent agent.Agent
}

func (c *transferContextWithAgent) Agent() agent.Agent {
	return c.rootAgent
}

// MCPToolCallArgs and SubAgentRequest intentionally contain "MCP" and "Agent"
// in their type names to verify that generic functiontool type parameters do
// not trigger false-positive MCP or SUB_AGENT classification.
type MCPToolCallArgs struct {
	Query string `json:"query"`
}

type SubAgentRequest struct {
	Target string `json:"target"`
}

func TestResolveToolOrigin(t *testing.T) {
	localAgent, err := agent.New(agent.Config{
		Name:        "local_helper",
		Description: "Local helper agent",
	})
	if err != nil {
		t.Fatalf("agent.New(local_helper) error: %v", err)
	}

	remoteA2AAgent, err := agent.New(agent.Config{
		Name:        "remote_specialist",
		Description: "Remote A2A specialist",
	})
	if err != nil {
		t.Fatalf("agent.New(remote_specialist) error: %v", err)
	}
	// Match the exact runtime state produced by remoteagent.NewA2A:
	// *agent.agent with embedded agentinternal.State{AgentType: "RemoteAgent"}.
	reflect.ValueOf(remoteA2AAgent).Elem().FieldByName("AgentType").SetString("RemoteAgent")

	rootWithSubAgents, err := agent.New(agent.Config{
		Name:        "root_router",
		Description: "Root router",
		SubAgents:   []agent.Agent{localAgent, remoteA2AAgent},
	})
	if err != nil {
		t.Fatalf("agent.New(root_router) error: %v", err)
	}

	p, _, baseCtx := setupTestPlugin(t)
	var realTransferTool tool.Tool
	transferRouter, err := llmagent.New(llmagent.Config{
		Name:        "transfer_router",
		Description: "Router",
		Model:       &scriptedLLM{},
		SubAgents:   []agent.Agent{localAgent, remoteA2AAgent},
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{
			func(_ agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
				realTransferTool, _ = req.Tools["transfer_to_agent"].(tool.Tool)
				return &model.LLMResponse{
					Content:      textContent("model", "done"),
					TurnComplete: true,
				}, nil
			},
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New(transfer_router) error: %v", err)
	}
	for _, runErr := range newTestRunner(t, transferRouter, p).Run(t.Context(), "u1", "s1", textContent("user", "route"), agent.RunConfig{}) {
		if runErr != nil {
			t.Fatalf("newTestRunner(transferRouter).Run() error: %v", runErr)
		}
	}
	if realTransferTool == nil {
		t.Fatal("realTransferTool = nil, want ADK transfer_to_agent tool")
	}

	fnTool, err := functiontool.New(functiontool.Config{
		Name:        "fn_lookup",
		Description: "Function tool",
	}, func(_ agent.Context, _ struct{}) (struct{}, error) {
		return struct{}{}, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New error: %v", err)
	}

	fnToolWithMCPAndAgentGenericArgs, err := functiontool.New(functiontool.Config{
		Name:        "fn_generic_names",
		Description: "Function tool with MCP and Agent in generic type args",
	}, func(_ agent.Context, _ MCPToolCallArgs) (SubAgentRequest, error) {
		return SubAgentRequest{}, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New(MCPToolCallArgs) error: %v", err)
	}

	confirmedFnTool, err := functiontool.New(functiontool.Config{
		Name:                "fn_confirmed",
		Description:         "Confirmed function tool",
		RequireConfirmation: true,
	}, func(_ agent.Context, _ struct{}) (struct{}, error) {
		return struct{}{}, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New(RequireConfirmation) error: %v", err)
	}

	// Wrap fnTool using the real tool.WithConfirmation toolset wrapper.
	confirmedLocalTools, err := tool.WithConfirmation(
		singleToolset{
			name: "local_ts",
			t:    fnTool,
		},
		true,
		nil,
	).Tools(nil)
	if err != nil || len(confirmedLocalTools) != 1 {
		t.Fatalf("tool.WithConfirmation.Tools() = %v, %v; want 1 tool", confirmedLocalTools, err)
	}
	realConfirmedLocalTool := confirmedLocalTools[0]

	localAgentTool := agenttool.New(localAgent, nil)
	remoteA2AAgentTool := agenttool.New(remoteA2AAgent, nil)

	confirmedA2ATools, err := tool.WithConfirmation(
		singleToolset{
			name: "a2a_ts",
			t:    remoteA2AAgentTool,
		},
		true,
		nil,
	).Tools(nil)
	if err != nil || len(confirmedA2ATools) != 1 {
		t.Fatalf("tool.WithConfirmation(a2a_ts).Tools() = %v, %v; want 1 tool", confirmedA2ATools, err)
	}
	realConfirmedA2ATool := confirmedA2ATools[0]

	transferTool := &fakeTool{name: "transfer_to_agent", description: "Transfer to another agent"}

	ap := &analyticsPlugin{config: applyDefaults(DefaultConfig()), traces: newTraceManager()}
	ctxWithHierarchy := &transferContextWithAgent{
		mockInvocationContext: baseCtx,
		rootAgent:             rootWithSubAgents,
	}
	// Register rootWithSubAgents in ap.traces via invocationSpan so ap.traces.agentNamed resolves sub-agents.
	ap.traces.invocationSpan(ctxWithHierarchy, identityOf(ctxWithHierarchy), runnerAgentOf(ctxWithHierarchy))
	realToolCtx := agent.NewToolContext(ctxWithHierarchy, "call-tx-1", nil, nil)
	// ctxWithoutTraces shares no invocation ID with ctxWithHierarchy, so ap.traces
	// has no cached agent hierarchy for it. Only findAgentA2AByName (which inspects
	// the ADK tool struct's fields directly) can classify the transfer target.
	ctxWithoutTraces := &mockInvocationContext{
		ctx:          context.Background(),
		agentName:    "test-agent",
		sessionID:    "sess-id-no-traces",
		invocationID: "inv-id-no-traces",
		userID:       "user-id",
	}

	tests := []struct {
		name string
		ctx  agent.InvocationContext
		tool tool.Tool
		args map[string]any
		want string
	}{
		{
			name: "nil tool returns UNKNOWN",
			tool: nil,
			want: toolOriginUnknown,
		},
		{
			name: "custom tool without classification returns UNKNOWN",
			tool: &fakeTool{name: "custom_unclassified"},
			want: toolOriginUnknown,
		},
		{
			name: "user exported wrapper without Unwrap is not read via reflection",
			tool: &exportedWrapperTool{
				fakeTool:     fakeTool{name: "wrap_custom"},
				RunnableTool: fnTool,
			},
			want: toolOriginUnknown,
		},
		{
			name: "real tool.WithConfirmation wrapper preserves inner origin",
			tool: realConfirmedLocalTool,
			want: toolOriginLocal,
		},
		{
			name: "real tool.WithConfirmation wrapper around A2A agenttool unwraps to A2A",
			tool: realConfirmedA2ATool,
			want: toolOriginA2A,
		},
		{
			name: "user unexported confirmationWrapperTool without Unwrap is not read via reflection",
			tool: &confirmationWrapperTool{
				fakeTool:     fakeTool{name: "confirm_val"},
				runnableTool: fnTool,
			},
			want: toolOriginUnknown,
		},
		{
			name: "Unwrap() method wrapper preserves inner origin",
			tool: &unwrapMethodTool{
				fakeTool: fakeTool{name: "unwrap_custom"},
				inner:    fnTool,
			},
			want: toolOriginLocal,
		},
		{
			name: "value-receiver Unwrap() method wrapper preserves inner origin",
			tool: valueUnwrapTool{
				name:  "val_unwrap",
				inner: remoteA2AAgentTool,
			},
			want: toolOriginA2A,
		},
		{
			name: "functiontool is LOCAL",
			tool: fnTool,
			want: toolOriginLocal,
		},
		{
			name: "functiontool with MCP/Agent in generic type parameters is still LOCAL",
			tool: fnToolWithMCPAndAgentGenericArgs,
			want: toolOriginLocal,
		},
		{
			name: "functiontool with RequireConfirmation unwraps confirmationTool to LOCAL",
			tool: confirmedFnTool,
			want: toolOriginLocal,
		},
		{
			name: "custom FinishTaskTool is UNKNOWN",
			tool: &fakeFinishTaskTool{fakeTool: fakeTool{name: "finish_task"}},
			want: toolOriginUnknown,
		},
		{
			name: "agenttool wrapping local agent is SUB_AGENT",
			tool: localAgentTool,
			want: toolOriginSubAgent,
		},
		{
			name: "agenttool wrapping remoteagent.NewA2A is A2A",
			tool: remoteA2AAgentTool,
			want: toolOriginA2A,
		},
		{
			name: "transfer_to_agent without args is TRANSFER_AGENT",
			ctx:  ctxWithHierarchy,
			tool: transferTool,
			want: toolOriginTransferAgent,
		},
		{
			name: "transfer_to_agent targeting local sub-agent via traceManager is TRANSFER_AGENT",
			ctx:  ctxWithHierarchy,
			tool: transferTool,
			args: map[string]any{"agent_name": "local_helper"},
			want: toolOriginTransferAgent,
		},
		{
			name: "transfer_to_agent targeting remote A2A sub-agent via traceManager is TRANSFER_A2A",
			ctx:  ctxWithHierarchy,
			tool: transferTool,
			args: map[string]any{"agent_name": "remote_specialist"},
			want: toolOriginTransferA2A,
		},
		{
			name: "ADK TransferToAgentTool with supportedAgents and real NewToolContext targeting local agent is TRANSFER_AGENT",
			ctx:  realToolCtx,
			tool: realTransferTool,
			args: map[string]any{"agent_name": "local_helper"},
			want: toolOriginTransferAgent,
		},
		{
			name: "ADK TransferToAgentTool with supportedAgents and real NewToolContext targeting remote A2A agent is TRANSFER_A2A",
			ctx:  realToolCtx,
			tool: realTransferTool,
			args: map[string]any{"agent_name": "remote_specialist"},
			want: toolOriginTransferA2A,
		},
		{
			name: "transfer_to_agent without supportedAgents under real NewToolContext resolves via traceManager to TRANSFER_A2A without log spam",
			ctx:  realToolCtx,
			tool: transferTool,
			args: map[string]any{"agent_name": "remote_specialist"},
			want: toolOriginTransferA2A,
		},
		{
			// When p.traces has NOT cached the target agent (e.g., BeforeRunCallback
			// was not called), the ADK tool struct's supportedAgents slice is the
			// sole source for A2A classification via findAgentA2AByName.
			name: "ADK TransferToAgentTool with supportedAgents resolves A2A via tool struct when traces is empty",
			ctx:  ctxWithoutTraces,
			tool: realTransferTool,
			args: map[string]any{"agent_name": "remote_specialist"},
			want: toolOriginTransferA2A,
		},
		{
			name: "loadmemorytool.New() is LOCAL",
			tool: loadmemorytool.New(),
			want: toolOriginLocal,
		},
		{
			name: "non-addressable user struct value wrapper without Unwrap is not read via reflection",
			tool: valueWrapperTool{
				name:         "val_wrap",
				runnableTool: remoteA2AAgentTool,
			},
			want: toolOriginUnknown,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var logBuf bytes.Buffer
			origOut := log.Writer()
			log.SetOutput(&logBuf)
			defer log.SetOutput(origOut)

			callCtx := tt.ctx
			if callCtx == nil {
				callCtx = baseCtx
			}
			got := ap.resolveToolOrigin(callCtx, identityOf(callCtx), tt.tool, tt.args)
			if got != tt.want {
				t.Errorf("resolveToolOrigin() = %q, want %q", got, tt.want)
			}
			if strings.Contains(logBuf.String(), "Agent() is not supported") {
				t.Errorf("unexpected log output during tool origin resolution: %q", logBuf.String())
			}
		})
	}

	t.Run("classifyADKToolType exact package and baseName matching", func(t *testing.T) {
		localHolder := reflect.ValueOf(localAgentTool).Elem()
		remoteHolder := reflect.ValueOf(remoteA2AAgentTool).Elem()
		txVal := reflect.ValueOf(realTransferTool).Elem()

		pkgTests := []struct {
			name     string
			v        reflect.Value
			pkg      string
			baseName string
			want     string
		}{
			{
				name:     "loadmemorytool package is LOCAL",
				v:        localHolder,
				pkg:      adkModulePath + "/tool/loadmemorytool",
				baseName: "loadMemoryTool",
				want:     toolOriginLocal,
			},
			{
				name:     "mcptoolset package is MCP",
				v:        localHolder,
				pkg:      adkModulePath + "/tool/mcptoolset",
				baseName: "mcpTool",
				want:     toolOriginMCP,
			},
			{
				name:     "workflowinternal TaskAgentTool wrapping local agent is SUB_AGENT",
				v:        localHolder,
				pkg:      adkModulePath + "/internal/workflowinternal",
				baseName: "TaskAgentTool",
				want:     toolOriginSubAgent,
			},
			{
				name:     "workflowinternal SingleTurnTool wrapping remote agent is A2A",
				v:        remoteHolder,
				pkg:      adkModulePath + "/internal/workflowinternal",
				baseName: "SingleTurnTool",
				want:     toolOriginA2A,
			},
			{
				name:     "workflowinternal FinishTaskTool is UNKNOWN",
				v:        localHolder,
				pkg:      adkModulePath + "/internal/workflowinternal",
				baseName: "FinishTaskTool",
				want:     toolOriginUnknown,
			},
			{
				name:     "internal TransferToAgentTool is TRANSFER_AGENT",
				v:        txVal,
				pkg:      adkModulePath + "/internal/llminternal",
				baseName: "TransferToAgentTool",
				want:     toolOriginTransferAgent,
			},
			{
				name:     "toolbox in package is TOOLBOX",
				v:        localHolder,
				pkg:      "github.com/googleapis/genai-toolbox/toolbox",
				baseName: "ToolboxTool",
				want:     toolOriginToolbox,
			},
			{
				name:     "skilltoolset package is SKILL",
				v:        localHolder,
				pkg:      adkModulePath + "/tool/skilltoolset",
				baseName: "SkillTool",
				want:     toolOriginSkill,
			},
		}
		for _, tc := range pkgTests {
			t.Run(tc.name, func(t *testing.T) {
				if got := classifyADKToolType(tc.v, tc.pkg, tc.baseName); got != tc.want {
					t.Errorf("classifyADKToolType(%q, %q) = %q, want %q", tc.pkg, tc.baseName, got, tc.want)
				}
			})
		}
	})

	t.Run("unwrapToolValue cycle detection and 8-hop limit on pointer Unwrap", func(t *testing.T) {
		selfRef := &selfRefUnwrapTool{fakeTool: fakeTool{name: "self_ref"}}
		if got := ap.resolveToolOrigin(baseCtx, identityOf(baseCtx), selfRef, nil); got != toolOriginUnknown {
			t.Errorf("resolveToolOrigin(selfRef) = %q, want %q", got, toolOriginUnknown)
		}
		if selfRef.calls > 8 {
			t.Errorf("selfRef.calls = %d, want <= 8", selfRef.calls)
		}

		mutualCalls := 0
		w1 := &mutualUnwrapTool{fakeTool: fakeTool{name: "mutual_1"}, calls: &mutualCalls}
		w2 := &mutualUnwrapTool{fakeTool: fakeTool{name: "mutual_2"}, peer: w1, calls: &mutualCalls}
		w1.peer = w2
		if got := ap.resolveToolOrigin(baseCtx, identityOf(baseCtx), w1, nil); got != toolOriginUnknown {
			t.Errorf("resolveToolOrigin(mutual) = %q, want %q", got, toolOriginUnknown)
		}
		if mutualCalls > 8 {
			t.Errorf("mutualCalls = %d, want <= 8", mutualCalls)
		}

		allocCalls := 0
		allocRef := &allocUnwrapTool{fakeTool: fakeTool{name: "alloc_ref"}, calls: &allocCalls}
		if got := ap.resolveToolOrigin(baseCtx, identityOf(baseCtx), allocRef, nil); got != toolOriginUnknown {
			t.Errorf("resolveToolOrigin(allocRef) = %q, want %q", got, toolOriginUnknown)
		}
		if allocCalls != 8 {
			t.Errorf("allocCalls = %d, want 8", allocCalls)
		}
	})
}

type swapTool struct {
	mu              sync.RWMutex
	name            string
	innerTool       tool.Tool
	supportedAgents []agent.Agent
}

func (s *swapTool) Name() string {
	return s.name
}

func (s *swapTool) Description() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.innerTool != nil {
		return s.innerTool.Description()
	}
	return "swap tool"
}

func (s *swapTool) IsLongRunning() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.innerTool != nil {
		return s.innerTool.IsLongRunning()
	}
	return false
}

func (s *swapTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{Name: s.name, Description: "swap tool"}
}

func (s *swapTool) Run(_ agent.Context, _ any) (map[string]any, error) {
	return map[string]any{"ok": true}, nil
}

type swapUnwrapTool struct {
	mu        sync.RWMutex
	name      string
	innerTool tool.Tool
}

func (s *swapUnwrapTool) Name() string {
	return s.name
}

func (s *swapUnwrapTool) Description() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.innerTool != nil {
		return s.innerTool.Description()
	}
	return "swap unwrap tool"
}

func (s *swapUnwrapTool) IsLongRunning() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.innerTool != nil {
		return s.innerTool.IsLongRunning()
	}
	return false
}

func (s *swapUnwrapTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{Name: s.name, Description: "swap unwrap tool"}
}

func (s *swapUnwrapTool) Run(_ agent.Context, _ any) (map[string]any, error) {
	return map[string]any{"ok": true}, nil
}

func (s *swapUnwrapTool) Unwrap() tool.Tool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.innerTool
}

func TestUnwrapToolValue_ConcurrentUserWrapperNoRace(t *testing.T) {
	fnTool, err := functiontool.New(functiontool.Config{
		Name:        "fn_swap",
		Description: "Function tool",
	}, func(_ agent.Context, _ struct{}) (struct{}, error) {
		return struct{}{}, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New error: %v", err)
	}

	localAgent, err := agent.New(agent.Config{
		Name:        "local_helper",
		Description: "Local helper agent",
	})
	if err != nil {
		t.Fatalf("agent.New(local_helper) error: %v", err)
	}
	remoteA2AAgent, err := agent.New(agent.Config{
		Name:        "remote_specialist",
		Description: "Remote A2A specialist",
	})
	if err != nil {
		t.Fatalf("agent.New(remote_specialist) error: %v", err)
	}
	reflect.ValueOf(remoteA2AAgent).Elem().FieldByName("AgentType").SetString("RemoteAgent")
	remoteA2AAgentTool := agenttool.New(remoteA2AAgent, nil)

	t.Run("user wrapper without Unwrap is not read via struct reflection", func(t *testing.T) {
		p, requests, mCtx := setupTestPlugin(t)
		normalSwap := &swapTool{name: "user_swap_tool", innerTool: fnTool}
		transferSwap := &swapTool{
			name:            "transfer_to_agent",
			innerTool:       fnTool,
			supportedAgents: []agent.Agent{localAgent, remoteA2AAgent},
		}
		confirmedTools, err := tool.WithConfirmation(singleToolset{name: "swap_ts", t: normalSwap}, true, nil).Tools(nil)
		if err != nil || len(confirmedTools) != 1 {
			t.Fatalf("tool.WithConfirmation.Tools() = %v, %v; want 1 tool", confirmedTools, err)
		}
		confirmedSwap := confirmedTools[0]

		runnerAgent, err := llmagent.New(llmagent.Config{
			Name:  "swap_runner",
			Tools: []tool.Tool{normalSwap},
		})
		if err != nil {
			t.Fatalf("llmagent.New(swap_runner) error: %v", err)
		}
		runCtx := &transferContextWithAgent{mockInvocationContext: mCtx, rootAgent: runnerAgent}

		done := make(chan struct{})
		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; ; i++ {
				select {
				case <-done:
					return
				default:
				}
				nextTool := tool.Tool(fnTool)
				nextAgents := []agent.Agent{localAgent}
				if i%2 == 1 {
					nextTool = remoteA2AAgentTool
					nextAgents = []agent.Agent{remoteA2AAgent}
				}
				normalSwap.mu.Lock()
				normalSwap.innerTool = nextTool
				normalSwap.mu.Unlock()

				transferSwap.mu.Lock()
				transferSwap.innerTool = nextTool
				transferSwap.supportedAgents = nextAgents
				transferSwap.mu.Unlock()
			}
		}()

		var errs callbackErrors
		for i := range 25 {
			args := map[string]any{"iter": i}
			txArgs := map[string]any{"agent_name": "remote_specialist"}
			errs.add(p.BeforeToolCallback()(mCtx, normalSwap, args))
			errs.add(p.AfterToolCallback()(mCtx, normalSwap, args, map[string]any{"ok": true}, nil))
			_, err := p.OnToolErrorCallback()(mCtx, confirmedSwap, args, errors.New("boom"))
			errs.add(nil, err)
			errs.add(p.BeforeToolCallback()(mCtx, transferSwap, txArgs))
			errs.add(p.AfterToolCallback()(mCtx, transferSwap, txArgs, map[string]any{"ok": true}, nil))
			errs.add(p.OnUserMessageCallback()(runCtx, &genai.Content{
				Role: "user",
				Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
					ID:       fmt.Sprintf("call-swap-%d", i),
					Name:     "user_swap_tool",
					Response: map[string]any{"resumed": true},
				}}},
			}))
		}
		close(done)
		wg.Wait()
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
		if len(rowsOfType(rows, EventToolStarting)) == 0 {
			t.Fatal("expected TOOL_STARTING rows")
		}
	})

	t.Run("user wrapper with Unwrap acquires lock and resolves inner tool origin", func(t *testing.T) {
		p, requests, mCtx := setupTestPlugin(t)
		unwrapSwap := &swapUnwrapTool{name: "user_unwrap_swap", innerTool: fnTool}
		confirmedTools, err := tool.WithConfirmation(singleToolset{name: "unwrap_swap_ts", t: unwrapSwap}, true, nil).Tools(nil)
		if err != nil || len(confirmedTools) != 1 {
			t.Fatalf("tool.WithConfirmation.Tools() = %v, %v; want 1 tool", confirmedTools, err)
		}
		confirmedUnwrapSwap := confirmedTools[0]

		done := make(chan struct{})
		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; ; i++ {
				select {
				case <-done:
					return
				default:
				}
				nextTool := tool.Tool(fnTool)
				if i%2 == 1 {
					nextTool = remoteA2AAgentTool
				}
				unwrapSwap.mu.Lock()
				unwrapSwap.innerTool = nextTool
				unwrapSwap.mu.Unlock()
			}
		}()

		var errs callbackErrors
		for i := range 25 {
			args := map[string]any{"iter": i}
			errs.add(p.BeforeToolCallback()(mCtx, unwrapSwap, args))
			errs.add(p.AfterToolCallback()(mCtx, confirmedUnwrapSwap, args, map[string]any{"ok": true}, nil))
		}
		close(done)
		wg.Wait()
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
		toolRows := append(rowsOfType(rows, EventToolStarting), rowsOfType(rows, EventToolCompleted)...)
		if len(toolRows) == 0 {
			t.Fatal("expected tool event rows")
		}
		for _, row := range toolRows {
			content := jsonObjectColumn(t, row, "content")
			origin, _ := content["tool_origin"].(string)
			if origin != toolOriginLocal && origin != toolOriginA2A {
				t.Errorf("tool_origin = %q, want %q or %q", origin, toolOriginLocal, toolOriginA2A)
			}
		}
	})
}

// TestUserMessage_ResumedToolRecordsToolOrigin checks the tool_origin of the
// TOOL_COMPLETED row written for a function response in a user message that
// answers a paused long-running tool call. The tool is looked up by name
// among the tools configured on the agent that the runner runs and its
// sub-agents, then in the rest of the agent tree seen so far.
func TestUserMessage_ResumedToolRecordsToolOrigin(t *testing.T) {
	newLLMAgent := func(cfg llmagent.Config) agent.Agent {
		t.Helper()
		a, err := llmagent.New(cfg)
		if err != nil {
			t.Fatalf("llmagent.New(%q) error: %v", cfg.Name, err)
		}
		return a
	}
	deploy, err := functiontool.New(functiontool.Config{
		Name:          "deploy_cluster",
		Description:   "Deploys a cluster",
		IsLongRunning: true,
	}, func(_ agent.Context, _ struct{}) (struct{}, error) {
		return struct{}{}, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New(deploy_cluster) error: %v", err)
	}
	worker := newLLMAgent(llmagent.Config{Name: "worker", Tools: []tool.Tool{deploy}})
	peer := newLLMAgent(llmagent.Config{Name: "peer"})
	root := newLLMAgent(llmagent.Config{Name: "root", SubAgents: []agent.Agent{worker, peer}})
	// remoteDeploy has the name of deploy and another origin (fakeTool -> UNKNOWN).
	remoteDeploy := &fakeTool{name: "deploy_cluster", longRunning: true}
	ambiguousRoot := newLLMAgent(llmagent.Config{
		Name:      "ambiguous_root",
		Tools:     []tool.Tool{remoteDeploy},
		SubAgents: []agent.Agent{newLLMAgent(llmagent.Config{Name: "local_worker", Tools: []tool.Tool{deploy}})},
	})

	tests := []struct {
		name string
		// seen, if set, is the agent of an earlier callback, whose tree the
		// plugin learns.
		seen   agent.Agent
		runner agent.Agent
		tool   string
		want   string
	}{
		{name: "tool of the agent that made the call", runner: worker, tool: "deploy_cluster", want: toolOriginLocal},
		{name: "tool of a sub-agent", runner: root, tool: "deploy_cluster", want: toolOriginLocal},
		{name: "tool elsewhere in a tree seen before", seen: root, runner: peer, tool: "deploy_cluster", want: toolOriginLocal},
		{name: "tool elsewhere in a tree not seen", runner: peer, tool: "deploy_cluster", want: toolOriginUnknown},
		{name: "unknown tool", runner: root, tool: "missing_tool", want: toolOriginUnknown},
		{name: "tools of different origins", runner: ambiguousRoot, tool: "deploy_cluster", want: toolOriginUnknown},
		{name: "no agent", tool: "deploy_cluster", want: toolOriginUnknown},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, mCtx := setupTestPlugin(t)
			var errs callbackErrors
			if tc.seen != nil {
				errs.add(p.BeforeRunCallback()(&transferContextWithAgent{mockInvocationContext: mCtx, rootAgent: tc.seen}))
			}
			ctx := &transferContextWithAgent{mockInvocationContext: mCtx, rootAgent: tc.runner}
			errs.add(p.OnUserMessageCallback()(ctx, &genai.Content{
				Role: "user",
				Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
					ID:       "call-pause",
					Name:     tc.tool,
					Response: map[string]any{"cluster_id": "c-123"},
				}}},
			}))
			errs.check(t)
			p.AfterRunCallback()(mCtx)

			rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
			row := onlyRowOfType(t, rows, EventToolCompleted)
			if got := row["status"]; got != "OK" {
				t.Errorf("status = %v, want OK", got)
			}
			checkJSONAt(t, jsonObjectColumn(t, row, "content"), fmt.Sprintf("%q", tc.want), "tool_origin")
		})
	}
}

type valueWrapperTool struct {
	name         string
	description  string
	longRunning  bool
	runnableTool tool.Tool
}

func (t valueWrapperTool) Name() string        { return t.name }
func (t valueWrapperTool) Description() string { return t.description }
func (t valueWrapperTool) IsLongRunning() bool { return t.longRunning }

type valueUnwrapTool struct {
	name        string
	description string
	longRunning bool
	inner       tool.Tool
}

func (t valueUnwrapTool) Name() string        { return t.name }
func (t valueUnwrapTool) Description() string { return t.description }
func (t valueUnwrapTool) IsLongRunning() bool { return t.longRunning }
func (t valueUnwrapTool) Unwrap() tool.Tool   { return t.inner }

type selfRefUnwrapTool struct {
	fakeTool
	calls int
}

func (t *selfRefUnwrapTool) Unwrap() tool.Tool {
	t.calls++
	return t
}

type mutualUnwrapTool struct {
	fakeTool
	peer  tool.Tool
	calls *int
}

func (t *mutualUnwrapTool) Unwrap() tool.Tool {
	*t.calls++
	return t.peer
}

type allocUnwrapTool struct {
	fakeTool
	calls *int
}

func (t *allocUnwrapTool) Unwrap() tool.Tool {
	*t.calls++
	return &allocUnwrapTool{fakeTool: t.fakeTool, calls: t.calls}
}

func TestToolConfirmationRejected_CompletesCall(t *testing.T) {
	p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
		c.FinalResponseToolNames = []string{"deploy"}
	})
	mCtx.functionCallID = "call-reject-1"
	deployTool := &fakeTool{name: "deploy"}
	args := map[string]any{"region": "us-central1"}
	toolErr := fmt.Errorf("error tool %q %w", "deploy", tool.ErrConfirmationRejected)

	var errs callbackErrors
	errs.add(p.BeforeToolCallback()(mCtx, deployTool, args))
	if res, err := p.OnToolErrorCallback()(mCtx, deployTool, args, toolErr); res != nil || err != nil {
		t.Errorf("OnToolErrorCallback() = %v, %v; want nil, nil", res, err)
	}
	errs.add(p.AfterToolCallback()(mCtx, deployTool, args, nil, toolErr))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	wantTypes := []string{
		EventToolStarting,
		EventToolCompleted,
		EventInvocationCompleted,
	}
	if diff := cmp.Diff(wantTypes, eventTypes(rows)); diff != "" {
		t.Fatalf("event types mismatch (-want +got):\n%s", diff)
	}

	row := onlyRowOfType(t, rows, EventToolCompleted)
	if got := row["status"]; got != "OK" {
		t.Errorf("status = %v, want OK", got)
	}
	if got, ok := row["error_message"]; ok {
		t.Errorf("error_message = %v, want none", got)
	}
	wantJSON := fmt.Sprintf(`{"tool":"deploy","result":{"error":%q},"tool_origin":"UNKNOWN"}`, toolErr.Error())
	checkJSON(t, "TOOL_COMPLETED content", jsonObjectColumn(t, row, "content"), wantJSON)
	checkJSONAt(t, jsonObjectColumn(t, row, "attributes"), `"call-reject-1"`, "adk", "function_call_id")
	if got, want := row["span_id"], onlyRowOfType(t, rows, EventToolStarting)["span_id"]; got != want {
		t.Errorf("TOOL_COMPLETED span_id = %v, want TOOL_STARTING span_id %v", got, want)
	}
	if _, ok := row[columnLatencyMS]; !ok {
		t.Error("TOOL_COMPLETED row has no latency_ms")
	}
}

func TestA2ANamespacedSensitiveKeys(t *testing.T) {
	newA2AEvent := func() *session.Event {
		return &session.Event{
			ID:     "ev-a2a-redact",
			Author: "remote-agent",
			LLMResponse: model.LLMResponse{
				CustomMetadata: map[string]any{
					"a2a:request":      map[string]any{"task": "ping"},
					"a2a:task_id":      "task-123",
					"a2a:api_key":      "sk-secret",
					"a2a:accessToken":  "tok-secret",
					"a2a:temp:scratch": "temp-secret",
					"a2a:note":         `trace {"api_key":"secret-inline-a2a"}`,
				},
			},
		}
	}

	t.Run("default config redacts sensitive keys under a2a: prefix", func(t *testing.T) {
		p, requests, mCtx := setupTestPlugin(t)
		var errs callbackErrors
		errs.add(p.OnEventCallback()(mCtx, newA2AEvent()))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
		row := onlyRowOfType(t, rows, EventA2AInteraction)
		attrs := jsonObjectColumn(t, row, "attributes")
		checkJSONAt(t, attrs, `"task-123"`, "a2a_metadata", "a2a:task_id")
		checkJSONAt(t, attrs, `"[REDACTED]"`, "a2a_metadata", "a2a:api_key")
		checkJSONAt(t, attrs, `"[REDACTED]"`, "a2a_metadata", "a2a:accessToken")
		checkJSONAt(t, attrs, `"[REDACTED]"`, "a2a_metadata", "a2a:temp:scratch")
		checkJSONAt(t, attrs, `"trace {\"api_key\":\"[REDACTED]\"}"`, "a2a_metadata", "a2a:note")

		rawAttrs, _ := row["attributes"].(string)
		for _, secret := range []string{"sk-secret", "tok-secret", "temp-secret", "secret-inline-a2a"} {
			if strings.Contains(rawAttrs, secret) {
				t.Errorf("attributes leaked %q: %s", secret, rawAttrs)
			}
		}
	})

	t.Run("DisableSensitiveKeyRedaction preserves a2a: values", func(t *testing.T) {
		p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
			c.DisableSensitiveKeyRedaction = true
		})
		var errs callbackErrors
		errs.add(p.OnEventCallback()(mCtx, newA2AEvent()))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
		row := onlyRowOfType(t, rows, EventA2AInteraction)
		attrs := jsonObjectColumn(t, row, "attributes")
		checkJSONAt(t, attrs, `"sk-secret"`, "a2a_metadata", "a2a:api_key")
		checkJSONAt(t, attrs, `"tok-secret"`, "a2a_metadata", "a2a:accessToken")
		checkJSONAt(t, attrs, `"temp-secret"`, "a2a_metadata", "a2a:temp:scratch")
		checkJSONAt(t, attrs, `"trace {\"api_key\":\"secret-inline-a2a\"}"`, "a2a_metadata", "a2a:note")
	})
}

type sseChunk struct {
	resp *model.LLMResponse
	err  error
}

type sseStreamLLM struct {
	mu    sync.Mutex
	steps [][]sseChunk
}

func (m *sseStreamLLM) Name() string { return "sse-stream-model" }

func (m *sseStreamLLM) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.mu.Lock()
		if len(m.steps) == 0 {
			m.mu.Unlock()
			yield(nil, errors.New("sseStreamLLM: no step left"))
			return
		}
		chunks := m.steps[0]
		m.steps = m.steps[1:]
		m.mu.Unlock()
		for _, chunk := range chunks {
			if !yield(chunk.resp, chunk.err) {
				return
			}
			if chunk.err != nil {
				return
			}
		}
	}
}

func TestSSEStreaming_PauseAndStateDeltaEmittedOnce(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	type approvalArgs struct {
		Item string `json:"item"`
	}
	var toolRuns int
	waitTool, err := functiontool.New(
		functiontool.Config{
			Name:          "wait_approval",
			Description:   "Waits for external approval",
			IsLongRunning: true,
		},
		func(_ agent.Context, _ approvalArgs) (map[string]any, error) {
			toolRuns++
			return nil, nil
		},
	)
	if err != nil {
		t.Fatalf("functiontool.New(wait_approval): %v", err)
	}

	callPart := &genai.Part{
		FunctionCall: &genai.FunctionCall{
			ID:   "call-lr-1",
			Name: "wait_approval",
			Args: map[string]any{"item": "cluster-1"},
		},
	}
	llm := &sseStreamLLM{
		steps: [][]sseChunk{
			{
				{
					resp: &model.LLMResponse{
						Partial: true,
						Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{callPart}},
					},
				},
				{
					resp: &model.LLMResponse{
						Partial:      false,
						TurnComplete: true,
						Content:      &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{callPart}},
					},
				},
			},
		},
	}

	var stateSet bool
	root, err := llmagent.New(llmagent.Config{
		Name:  "sse_root",
		Model: llm,
		Tools: []tool.Tool{waitTool},
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{
			func(ctx agent.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
				if !stateSet {
					stateSet = true
					if sErr := ctx.State().Set("user:model_calls", 1); sErr != nil {
						return nil, sErr
					}
				}
				return nil, nil
			},
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New(sse_root): %v", err)
	}

	r := newTestRunner(t, root, p)
	for _, runErr := range r.Run(t.Context(), "u1", "s1", textContent("user", "deploy"), agent.RunConfig{StreamingMode: agent.StreamingModeSSE}) {
		if runErr != nil {
			t.Fatalf("r.Run() error: %v", runErr)
		}
	}

	rows := collectRows(t, requests, untilInvocationEnd)
	requireInvocationCompletedOK(t, rows)
	if toolRuns != 1 {
		t.Fatalf("long-running tool ran %d times, want 1", toolRuns)
	}
	if got := len(rowsOfType(rows, EventStateDelta)); got != 1 {
		t.Errorf("got %d STATE_DELTA rows, want 1; event types: %v", got, eventTypes(rows))
	}
	pausedRows := rowsOfType(rows, EventToolPaused)
	if len(pausedRows) != 1 {
		t.Fatalf("got %d TOOL_PAUSED rows, want 1; event types: %v", len(pausedRows), eventTypes(rows))
	}
	agentStart := onlyRowOfType(t, rows, EventAgentStarting)
	if got, want := pausedRows[0]["span_id"], agentStart["span_id"]; got != want {
		t.Errorf("TOOL_PAUSED span_id = %v, want AGENT_STARTING span_id %v", got, want)
	}
	if got := functionCallIDOf(t, pausedRows[0]); got != "call-lr-1" {
		t.Errorf("TOOL_PAUSED function_call_id = %q, want %q", got, "call-lr-1")
	}
}

func TestSSEStreaming_AbortedStreamDoesNotEmitPause(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	type approvalArgs struct {
		Item string `json:"item"`
	}
	var toolRuns int
	waitTool, err := functiontool.New(
		functiontool.Config{
			Name:          "wait_approval",
			Description:   "Waits for external approval",
			IsLongRunning: true,
		},
		func(_ agent.Context, _ approvalArgs) (map[string]any, error) {
			toolRuns++
			return nil, nil
		},
	)
	if err != nil {
		t.Fatalf("functiontool.New(wait_approval): %v", err)
	}

	streamErr := errors.New("stream reset by peer")
	llm := &sseStreamLLM{
		steps: [][]sseChunk{
			{
				{
					resp: &model.LLMResponse{
						Partial: true,
						Content: &genai.Content{
							Role: genai.RoleModel,
							Parts: []*genai.Part{
								{
									FunctionCall: &genai.FunctionCall{
										ID:   "call-lr-abort",
										Name: "wait_approval",
										Args: map[string]any{"item": "cluster-1"},
									},
								},
							},
						},
					},
				},
				{err: streamErr},
			},
		},
	}

	root, err := llmagent.New(llmagent.Config{
		Name:  "sse_abort_root",
		Model: llm,
		Tools: []tool.Tool{waitTool},
	})
	if err != nil {
		t.Fatalf("llmagent.New(sse_abort_root): %v", err)
	}

	r := newTestRunner(t, root, p)
	var runErrs []error
	for _, runErr := range r.Run(t.Context(), "u1", "s1", textContent("user", "deploy"), agent.RunConfig{StreamingMode: agent.StreamingModeSSE}) {
		if runErr != nil {
			runErrs = append(runErrs, runErr)
		}
	}
	if len(runErrs) == 0 {
		t.Fatal("expected r.Run() to yield stream error")
	}

	rows := collectRows(t, requests, untilInvocationEnd)
	if toolRuns != 0 {
		t.Errorf("long-running tool ran %d times on aborted stream, want 0", toolRuns)
	}
	if got := len(rowsOfType(rows, EventToolStarting)); got != 0 {
		t.Errorf("got %d TOOL_STARTING rows on aborted stream, want 0; event types: %v", got, eventTypes(rows))
	}
	if got := len(rowsOfType(rows, EventToolPaused)); got != 0 {
		t.Errorf("got %d TOOL_PAUSED rows on aborted stream, want 0; event types: %v", got, eventTypes(rows))
	}
}

func TestAgentTool_SkipSummarizationLogsFinalAgentResponse(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	subAgent, err := llmagent.New(llmagent.Config{
		Name:        "sub_agent",
		Description: "Answers delegate questions",
		Model: &scriptedLLM{
			responses: []*model.LLMResponse{
				{Content: textContent("model", "sub-agent final answer"), TurnComplete: true},
			},
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New(sub_agent): %v", err)
	}
	subTool := agenttool.New(subAgent, &agenttool.Config{SkipSummarization: true})
	if _, ok := any(subTool).(interface{ DisplayResultOnSkipSummarization() bool }); !ok {
		t.Skip("agenttool.DisplayResultOnSkipSummarization was added after ADK v2.0.0")
	}
	rootAgent, err := llmagent.New(llmagent.Config{
		Name:  "root_agent",
		Tools: []tool.Tool{subTool},
		Model: &scriptedLLM{
			responses: []*model.LLMResponse{
				{
					Content: &genai.Content{
						Role: genai.RoleModel,
						Parts: []*genai.Part{
							genai.NewPartFromFunctionCall("sub_agent", map[string]any{"request": "help me"}),
						},
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New(root_agent): %v", err)
	}

	r := newTestRunner(t, rootAgent, p)
	for _, runErr := range r.Run(t.Context(), "u1", "s1", textContent("user", "ask sub-agent"), agent.RunConfig{}) {
		if runErr != nil {
			t.Fatalf("r.Run() error: %v", runErr)
		}
	}

	rows := collectRows(t, requests, untilInvocationEnd)
	requireInvocationCompletedOK(t, rows)

	var foundSkipSummaryResponse bool
	for _, row := range rowsOfType(rows, EventAgentResponse) {
		attrs := jsonObjectColumn(t, row, "attributes")
		if skip, _ := lookup(attrs, "adk", "skip_summarization"); skip == true {
			foundSkipSummaryResponse = true
			checkJSON(t, "SkipSummarization AGENT_RESPONSE content", jsonObjectColumn(t, row, "content"),
				`{"response":"text: 'sub-agent final answer'"}`)
		}
	}
	if !foundSkipSummaryResponse {
		t.Fatalf("no AGENT_RESPONSE row emitted for SkipSummarization agenttool display event; event types: %v", eventTypes(rows))
	}
}

func TestCompaction_CyclicMetadataDoesNotHang(t *testing.T) {
	var cyclic any
	cyclic = &cyclic

	newCyclicEvent := func() *session.Event {
		return &session.Event{
			ID:     "ev-cycle-compact",
			Author: "test-agent",
			LLMResponse: model.LLMResponse{
				Content: textContent("model", "ok"),
				CustomMetadata: map[string]any{
					"adk_compaction": cyclic,
				},
			},
		}
	}

	tests := []struct {
		name     string
		denylist []string
	}{
		{name: "default config"},
		{name: "EventEventCompaction denied", denylist: []string{EventEventCompaction}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
				c.EventDenylist = tt.denylist
			})
			done := make(chan error, 1)
			go func() {
				_, err := p.OnEventCallback()(mCtx, newCyclicEvent())
				done <- err
			}()

			select {
			case err := <-done:
				if err != nil {
					t.Fatalf("OnEventCallback() error: %v", err)
				}
			case <-time.After(250 * time.Millisecond):
				t.Fatal("OnEventCallback() hung on cyclic compaction pointer")
			}

			p.AfterRunCallback()(mCtx)
			rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
			if got := len(rowsOfType(rows, EventEventCompaction)); got != 0 {
				t.Errorf("got %d EVENT_COMPACTION rows for cyclic pointer, want 0", got)
			}
			if got := len(rowsOfType(rows, EventAgentResponse)); got != 1 {
				t.Errorf("got %d AGENT_RESPONSE rows, want 1", got)
			}
		})
	}
}

func TestCustomMetadataAllowlist_GatingOnLifecycleEvents(t *testing.T) {
	t.Run("AGENT_STATE_CHECKPOINT respects CustomMetadataAllowlist", func(t *testing.T) {
		// When allowlist does not include agent_state, AGENT_STATE_CHECKPOINT is not emitted.
		p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
			c.CustomMetadataAllowlist = []string{"unrelated_key"}
		})
		var errs callbackErrors
		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			Author: "state-agent",
			LLMResponse: model.LLMResponse{
				CustomMetadata: map[string]any{
					"agent_state": map[string]any{"step": 1},
				},
			},
		}))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
		if got := len(rowsOfType(rows, EventAgentStateCheckpoint)); got != 0 {
			t.Errorf("got %d AGENT_STATE_CHECKPOINT rows when not allowlisted, want 0", got)
		}

		// When allowlist includes agent_state, AGENT_STATE_CHECKPOINT is emitted.
		p2, requests2, mCtx2 := setupTestPluginWithConfig(t, func(c *Config) {
			c.CustomMetadataAllowlist = []string{"agent_state"}
		})
		var errs2 callbackErrors
		errs2.add(p2.OnEventCallback()(mCtx2, &session.Event{
			Author: "state-agent",
			LLMResponse: model.LLMResponse{
				CustomMetadata: map[string]any{
					"agent_state": map[string]any{"step": 1},
				},
			},
		}))
		errs2.check(t)
		p2.AfterRunCallback()(mCtx2)

		rows2 := collectRows(t, requests2, untilEventType(EventInvocationCompleted))
		if got := len(rowsOfType(rows2, EventAgentStateCheckpoint)); got != 1 {
			t.Fatalf("got %d AGENT_STATE_CHECKPOINT rows when allowlisted, want 1", got)
		}
		cpRow := onlyRowOfType(t, rows2, EventAgentStateCheckpoint)
		checkJSONAt(t, jsonObjectColumn(t, cpRow, "content"), `1`, "agent_state", "step")
	})

	t.Run("EVENT_COMPACTION via metadata respects CustomMetadataAllowlist", func(t *testing.T) {
		compactionData := map[string]any{
			"start_timestamp":   1700000000.0,
			"end_timestamp":     1700000100.0,
			"compacted_content": "summary of conversation",
		}

		// When allowlist does not include adk_compaction, EVENT_COMPACTION is not emitted.
		p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
			c.CustomMetadataAllowlist = []string{"unrelated_key"}
		})
		var errs callbackErrors
		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			Author: "compaction-agent",
			LLMResponse: model.LLMResponse{
				CustomMetadata: map[string]any{
					"adk_compaction": compactionData,
				},
			},
		}))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
		if got := len(rowsOfType(rows, EventEventCompaction)); got != 0 {
			t.Errorf("got %d EVENT_COMPACTION rows when not allowlisted, want 0", got)
		}

		// When allowlist includes adk_compaction, EVENT_COMPACTION is emitted.
		p2, requests2, mCtx2 := setupTestPluginWithConfig(t, func(c *Config) {
			c.CustomMetadataAllowlist = []string{"adk_compaction"}
		})
		var errs2 callbackErrors
		errs2.add(p2.OnEventCallback()(mCtx2, &session.Event{
			Author: "compaction-agent",
			LLMResponse: model.LLMResponse{
				CustomMetadata: map[string]any{
					"adk_compaction": compactionData,
				},
			},
		}))
		errs2.check(t)
		p2.AfterRunCallback()(mCtx2)

		rows2 := collectRows(t, requests2, untilEventType(EventInvocationCompleted))
		if got := len(rowsOfType(rows2, EventEventCompaction)); got != 1 {
			t.Fatalf("got %d EVENT_COMPACTION rows when allowlisted, want 1", got)
		}
	})

	t.Run("a2a:* metadata bypasses CustomMetadataAllowlist with sensitive keys redacted", func(t *testing.T) {
		p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
			c.CustomMetadataAllowlist = []string{"allowed_user_key"}
		})
		var errs callbackErrors
		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			Author: "a2a-remote-agent",
			LLMResponse: model.LLMResponse{
				CustomMetadata: map[string]any{
					"allowed_user_key":  "user-visible-val",
					"blocked_user_key":  "blocked-val",
					"a2a:request":       map[string]any{"task": "summarize"},
					"a2a:response":      map[string]any{"result": "done"},
					"a2a:api_key":       "super-secret-key",
					"a2a:auth_code":     "secret-oauth-code",
					"a2a:code_verifier": "secret-verifier",
				},
			},
		}))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
		a2aRow := onlyRowOfType(t, rows, EventA2AInteraction)
		attrs := jsonObjectColumn(t, a2aRow, "attributes")

		// a2a:* bypassed allowlist
		checkJSONAt(t, attrs, `"summarize"`, "a2a_metadata", "a2a:request", "task")
		checkJSONAt(t, attrs, `"[REDACTED]"`, "a2a_metadata", "a2a:api_key")
		checkJSONAt(t, attrs, `"[REDACTED]"`, "a2a_metadata", "a2a:auth_code")
		checkJSONAt(t, attrs, `"[REDACTED]"`, "a2a_metadata", "a2a:code_verifier")

		// custom_metadata only contains allowed_user_key
		checkJSONAt(t, attrs, `"user-visible-val"`, "custom_metadata", "allowed_user_key")
		rawAttrs, _ := a2aRow["attributes"].(string)
		if strings.Contains(rawAttrs, "blocked-val") {
			t.Errorf("blocked_user_key leaked into attributes: %s", rawAttrs)
		}
		for _, secret := range []string{"super-secret-key", "secret-oauth-code", "secret-verifier"} {
			if strings.Contains(rawAttrs, secret) {
				t.Errorf("secret %q leaked into attributes: %s", secret, rawAttrs)
			}
		}
	})
}

func TestEventCompaction_SiblingKeysOmitted(t *testing.T) {
	p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
		c.CustomMetadataAllowlist = []string{"adk_compaction"}
	})
	var errs callbackErrors
	errs.add(p.OnEventCallback()(mCtx, &session.Event{
		Author: "compaction-agent",
		LLMResponse: model.LLMResponse{
			CustomMetadata: map[string]any{
				"adk_compaction": map[string]any{
					"start_timestamp":   1700000000.0,
					"end_timestamp":     1700000100.0,
					"compacted_content": "conversation summary text",
					"ssn":               "123-45-6789",
					"password":          "dev-secret-password",
					"internal_tokens":   []string{"tok1", "tok2"},
				},
			},
		},
	}))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	row := onlyRowOfType(t, rows, EventEventCompaction)
	content := jsonObjectColumn(t, row, "content")

	checkJSONAt(t, content, `1.7e+09`, "start_timestamp")
	checkJSONAt(t, content, `1.7000001e+09`, "end_timestamp")
	checkJSONAt(t, content, `"conversation summary text"`, "compacted_content")

	rawContent, _ := row["content"].(string)
	for _, forbidden := range []string{"ssn", "password", "internal_tokens", "123-45-6789", "dev-secret-password"} {
		if strings.Contains(rawContent, forbidden) {
			t.Errorf("content leaked %q: %s", forbidden, rawContent)
		}
	}
}

func TestAfterTool_EventDenylistDoesNotDropAgentResponse(t *testing.T) {
	// Using logEvent in afterTool ensures EventDenylist: ["EVENT"]
	// does not drop AGENT_RESPONSE emitted by final response tools.
	p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
		c.FinalResponseToolNames = []string{"submit_final"}
		c.EventDenylist = []string{"EVENT"}
	})
	submitTool := &fakeTool{name: "submit_final"}
	args := map[string]any{"decision": "approved"}
	result := map[string]any{"status": "ok", "message": "all done"}

	var errs callbackErrors
	errs.add(p.BeforeToolCallback()(mCtx, submitTool, args))
	errs.add(p.AfterToolCallback()(mCtx, submitTool, args, result, nil))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	agentResponses := rowsOfType(rows, EventAgentResponse)
	if len(agentResponses) == 0 {
		t.Fatalf("expected AGENT_RESPONSE row in afterTool when EventDenylist: ['EVENT'], but got none; events: %v", eventTypes(rows))
	}
}

func TestSensitiveKeys_StandardRedactionInCompactionAndA2A(t *testing.T) {
	p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
		c.CustomMetadataAllowlist = []string{"adk_compaction"}
	})
	var errs callbackErrors
	errs.add(p.OnEventCallback()(mCtx, &session.Event{
		Author: "remote-agent",
		LLMResponse: model.LLMResponse{
			CustomMetadata: map[string]any{
				"adk_compaction": map[string]any{
					"start_timestamp":   1700000000.0,
					"end_timestamp":     1700000100.0,
					"compacted_content": "login auth_code=MY_AUTH_CODE and code_verifier=MY_VERIFIER_STRING",
				},
				"a2a:request": map[string]any{
					"note": "exchange auth_code=MY_A2A_CODE and code_verifier=MY_A2A_VERIFIER",
				},
				"a2a:response": map[string]any{
					"summary": "success with auth_code=MY_A2A_CODE",
				},
			},
		},
	}))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))

	compRow := onlyRowOfType(t, rows, EventEventCompaction)
	compContent, _ := compRow["content"].(string)
	if strings.Contains(compContent, "MY_AUTH_CODE") || strings.Contains(compContent, "MY_VERIFIER_STRING") {
		t.Errorf("EVENT_COMPACTION content leaked sensitive keys: %s", compContent)
	}
	if !strings.Contains(compContent, "[REDACTED]") {
		t.Errorf("EVENT_COMPACTION content missing [REDACTED]: %s", compContent)
	}

	a2aRow := onlyRowOfType(t, rows, EventA2AInteraction)
	attrs := jsonObjectColumn(t, a2aRow, "attributes")
	rawAttrs, _ := a2aRow["attributes"].(string)
	if strings.Contains(rawAttrs, "MY_A2A_CODE") || strings.Contains(rawAttrs, "MY_A2A_VERIFIER") {
		t.Errorf("A2A_INTERACTION attributes leaked sensitive keys: %s", rawAttrs)
	}
	checkJSONAt(t, attrs, `"exchange auth_code=[REDACTED] and code_verifier=[REDACTED]"`, "a2a_metadata", "a2a:request", "note")
}
