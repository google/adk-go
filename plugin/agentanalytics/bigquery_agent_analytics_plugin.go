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

// Package agentanalytics implements the BigQuery Agent Analytics Plugin.
package agentanalytics

import (
	"context"
	"fmt"
	"time"

	bq "cloud.google.com/go/bigquery"
	bqstorage "cloud.google.com/go/bigquery/storage/apiv1"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
	baseplugin "google.golang.org/adk/plugin"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

// NewBigQueryAgentAnalyticsPlugin creates a newly configured analytics plugin with default config.
func NewBigQueryAgentAnalyticsPlugin(
	ctx context.Context,
	projectID string,
	datasetID string,
	tableID string,
) (*baseplugin.Plugin, error) {
	config := DefaultConfig()
	config.ProjectID = projectID
	config.DatasetID = datasetID
	config.TableName = tableID
	return NewBigQueryAgentAnalyticsPluginWithConfig(ctx, config)
}

// NewBigQueryAgentAnalyticsPluginWithConfig creates a newly configured analytics plugin.
func NewBigQueryAgentAnalyticsPluginWithConfig(
	ctx context.Context,
	config Config,
) (*baseplugin.Plugin, error) {
	if !config.Enabled {
		// Return empty plugin wrapped
		cfg := baseplugin.Config{Name: "bigquery_agent_analytics"}
		return baseplugin.New(cfg)
	}

	// Set up BigQuery data client for generic operations like ensure table exists
	// We use the normal BigQuery client here
	bqClient, err := bq.NewClient(ctx, config.ProjectID, config.ClientOptions...)
	if err != nil {
		return nil, fmt.Errorf("failed to create bigquery client: %w", err)
	}
	defer func() {
		if err := bqClient.Close(); err != nil {
			if config.Logger != nil {
				config.Logger.Printf("failed to close bigquery client: %v", err)
			}
		}
	}()

	// Setup Storage Write Client
	writeClient, err := bqstorage.NewBigQueryWriteClient(ctx, config.ClientOptions...)
	if err != nil {
		return nil, fmt.Errorf("failed to create BigQuery write client: %w", err)
	}

	return NewBigQueryAgentAnalyticsPluginWithClients(ctx, config, bqClient, writeClient)
}

// NewBigQueryAgentAnalyticsPluginWithClients creates a newly configured analytics plugin using the provided BigQuery clients.
func NewBigQueryAgentAnalyticsPluginWithClients(
	ctx context.Context,
	config Config,
	bqClient *bq.Client,
	writeClient *bqstorage.BigQueryWriteClient,
) (*baseplugin.Plugin, error) {
	if !config.Enabled {
		// Return empty plugin wrapped
		cfg := baseplugin.Config{Name: "bigquery_agent_analytics"}
		return baseplugin.New(cfg)
	}

	// Ensure table exists (done once per instance)
	tableRef := bqClient.Dataset(config.DatasetID).Table(config.TableName)
	_, err := tableRef.Metadata(ctx)
	if err != nil {
		if config.Logger != nil {
			config.Logger.Printf("Table %s not found. Creating it...", config.TableName)
		}
		err = tableRef.Create(ctx, &bq.TableMetadata{
			Schema: EventsSchema(),
			Clustering: &bq.Clustering{
				Fields: config.ClusteringFields,
			},
		})
		if err != nil {
			if config.Logger != nil {
				config.Logger.Printf("Failed to create BigQuery table %v: %v", config.TableName, err)
			}
		}
	}

	// Determine destination stream
	streamName := fmt.Sprintf("projects/%s/datasets/%s/tables/%s/streams/_default", config.ProjectID, config.DatasetID, config.TableName)

	processor, err := NewBatchProcessor(ctx, writeClient, streamName, config)
	if err != nil {
		if closeErr := writeClient.Close(); closeErr != nil {
			if config.Logger != nil {
				config.Logger.Printf("failed to close BigQuery write client: %v", closeErr)
			}
		}
		return nil, fmt.Errorf("failed to initialize batch processor: %w", err)
	}
	processor.Start()

	// Helper closure func to construct log events
	logEvent := func(ctx context.Context, eventType string, content any, extraAttrs map[string]any) {
		row := make(map[string]any)
		row["timestamp"] = time.Now()
		row["event_type"] = eventType

		if rCtx, ok := ctx.(agent.ReadonlyContext); ok {
			if rCtx.AgentName() != "" {
				row["agent"] = rCtx.AgentName()
			}
			if rCtx.SessionID() != "" {
				row["session_id"] = rCtx.SessionID()
			}
			row["invocation_id"] = rCtx.InvocationID()
			row["user_id"] = rCtx.UserID()
		} else if iCtx, ok := ctx.(agent.InvocationContext); ok {
			if iCtx.Agent() != nil && iCtx.Agent().Name() != "" {
				row["agent"] = iCtx.Agent().Name()
			}
			if iCtx.Session() != nil {
				if iCtx.Session().ID() != "" {
					row["session_id"] = iCtx.Session().ID()
				}
				row["user_id"] = iCtx.Session().UserID()
			}
			row["invocation_id"] = iCtx.InvocationID()
		}

		isTruncated := false
		if c, ok := content.(*genai.Content); ok {
			row["content_parts"] = FormatContentParts(c, config.MaxContentLen)
			truncContent, truncated, _ := SmartTruncate(content, config.MaxContentLen)
			row["content"] = string(truncContent)
			isTruncated = isTruncated || truncated
		} else if content != nil {
			truncContent, truncated, _ := SmartTruncate(content, config.MaxContentLen)
			row["content"] = string(truncContent)
			isTruncated = isTruncated || truncated
		}

		attrs := make(map[string]any)
		for k, v := range config.CustomTags {
			attrs[k] = v
		}
		for k, v := range extraAttrs {
			switch k {
			case "error_message", "status":
				if strVal, ok := v.(string); ok {
					if len(strVal) > config.MaxContentLen {
						row[k] = strVal[:config.MaxContentLen]
						isTruncated = true
					} else {
						row[k] = strVal
					}
				} else {
					// If not a string, fallback to SmartTruncate (which will JSON encode)
					truncVal, truncated, _ := SmartTruncate(v, config.MaxContentLen)
					row[k] = string(truncVal)
					isTruncated = isTruncated || truncated
				}
			case "latency_ms":
				// latency_ms is typically numeric, SmartTruncate is fine here.
				truncVal, truncated, _ := SmartTruncate(v, config.MaxContentLen)
				row[k] = string(truncVal)
				isTruncated = isTruncated || truncated
			default:
				attrs[k] = v
			}
		}

		if len(attrs) > 0 {
			truncAttrs, truncated, _ := SmartTruncate(attrs, config.MaxContentLen)
			row["attributes"] = string(truncAttrs)
			isTruncated = isTruncated || truncated
		}
		row["is_truncated"] = isTruncated

		// Add trace info dynamically resolving the context span
		spanCtx := trace.SpanContextFromContext(ctx)
		if spanCtx.IsValid() {
			row["trace_id"] = spanCtx.TraceID().String()
			row["span_id"] = spanCtx.SpanID().String()
		}

		processor.Append(row)
	}

	// Create callbacks
	cfg := baseplugin.Config{
		Name: "bigquery_agent_analytics",

		OnUserMessageCallback: func(ctx agent.InvocationContext, msg *genai.Content) (*genai.Content, error) {
			logEvent(ctx, "USER_MESSAGE", msg, nil)
			return msg, nil
		},
		BeforeRunCallback: func(ctx agent.InvocationContext) (*genai.Content, error) {
			logEvent(ctx, "INVOCATION_START", nil, nil)
			return nil, nil
		},
		AfterRunCallback: func(ctx agent.InvocationContext) {
			logEvent(ctx, "INVOCATION_END", nil, nil)
			processor.flush() // flush queue synchronously
		},
		OnEventCallback: func(ctx agent.InvocationContext, ev *session.Event) (*session.Event, error) {
			attrs := map[string]any{"event_author": ev.Author}
			logEvent(ctx, "EVENT", ev.Content, attrs)
			return ev, nil
		},

		BeforeAgentCallback: func(ctx agent.CallbackContext) (*genai.Content, error) {
			logEvent(ctx, "AGENT_START", nil, nil)
			return nil, nil
		},
		AfterAgentCallback: func(ctx agent.CallbackContext) (*genai.Content, error) {
			logEvent(ctx, "AGENT_END", nil, nil)
			return nil, nil
		},

		BeforeModelCallback: func(ctx agent.CallbackContext, req *model.LLMRequest) (*model.LLMResponse, error) {
			attrs := map[string]any{"model": "unknown"}
			if req.Model != "" {
				attrs["model"] = req.Model
			}
			logEvent(ctx, "MODEL_REQUEST", req, attrs)
			return nil, nil
		},
		AfterModelCallback: func(ctx agent.CallbackContext, res *model.LLMResponse, err error) (*model.LLMResponse, error) {
			attrs := map[string]any{}
			if res != nil && res.UsageMetadata != nil {
				attrs["usage_metadata"] = res.UsageMetadata
			}
			logEvent(ctx, "MODEL_RESPONSE", res, attrs)
			return nil, nil
		},
		OnModelErrorCallback: func(ctx agent.CallbackContext, req *model.LLMRequest, err error) (*model.LLMResponse, error) {
			attrs := map[string]any{"error_message": err.Error()}
			logEvent(ctx, "MODEL_ERROR", nil, attrs)
			return nil, nil
		},

		BeforeToolCallback: func(ctx tool.Context, t tool.Tool, args map[string]any) (map[string]any, error) {
			attrs := map[string]any{"tool_name": t.Name()}
			logEvent(ctx, "TOOL_START", args, attrs)
			return nil, nil
		},
		AfterToolCallback: func(ctx tool.Context, t tool.Tool, args, res map[string]any, err error) (map[string]any, error) {
			attrs := map[string]any{"tool_name": t.Name()}
			logEvent(ctx, "TOOL_END", res, attrs)
			return nil, nil
		},
		OnToolErrorCallback: func(ctx tool.Context, t tool.Tool, args map[string]any, err error) (map[string]any, error) {
			attrs := map[string]any{
				"tool_name":     t.Name(),
				"error_message": err.Error(),
			}
			logEvent(ctx, "TOOL_ERROR", nil, attrs)
			return nil, nil
		},

		CloseFunc: func() error {
			processor.Close()
			if err := writeClient.Close(); err != nil {
				return fmt.Errorf("failed to close BigQuery write client: %w", err)
			}
			return nil
		},
	}

	return baseplugin.New(cfg)
}
