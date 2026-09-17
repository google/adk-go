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
	"net/http"
	"strings"
	"time"

	bq "cloud.google.com/go/bigquery"
	bqstorage "cloud.google.com/go/bigquery/storage/apiv1"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/api/googleapi"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	baseplugin "google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
)

const (
	schemaVersion         = "1"
	schemaVersionLabelKey = "adk_schema_version"
	viewSQLTemplate       = "CREATE OR REPLACE VIEW `%s.%s.%s` AS\nSELECT\n  %s\nFROM\n  `%s.%s.%s`\nWHERE\n  event_type = '%s'"
)

var viewCommonColumns = []string{
	"timestamp",
	"event_type",
	"agent",
	"session_id",
	"invocation_id",
	"user_id",
	"trace_id",
	"span_id",
	"parent_span_id",
	"status",
	"error_message",
	"is_truncated",
}

var eventViewDefs = map[string][]string{
	"USER_MESSAGE": {},
	"MODEL_REQUEST": {
		"JSON_VALUE(attributes, '$.model') AS model",
		"content AS request_content",
		"JSON_QUERY(attributes, '$.llm_config') AS llm_config",
		"JSON_QUERY(attributes, '$.tools') AS tools",
	},
	"MODEL_RESPONSE": {
		"JSON_QUERY(content, '$.response') AS response",
		"CAST(JSON_VALUE(content, '$.usage.prompt') AS INT64) AS usage_prompt_tokens",
		"CAST(JSON_VALUE(content, '$.usage.completion') AS INT64) AS usage_completion_tokens",
		"CAST(JSON_VALUE(content, '$.usage.total') AS INT64) AS usage_total_tokens",
		"CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64) AS total_ms",
		"CAST(JSON_VALUE(latency_ms, '$.time_to_first_token_ms') AS INT64) AS ttft_ms",
		"JSON_VALUE(attributes, '$.model_version') AS model_version",
		"JSON_QUERY(attributes, '$.usage_metadata') AS usage_metadata",
	},
	"MODEL_ERROR": {
		"CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64) AS total_ms",
	},
	"TOOL_START": {
		"JSON_VALUE(content, '$.tool') AS tool_name",
		"JSON_QUERY(content, '$.args') AS tool_args",
		"JSON_VALUE(content, '$.tool_origin') AS tool_origin",
	},
	"TOOL_END": {
		"JSON_VALUE(content, '$.tool') AS tool_name",
		"JSON_QUERY(content, '$.result') AS tool_result",
		"JSON_VALUE(content, '$.tool_origin') AS tool_origin",
		"CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64) AS total_ms",
	},
	"TOOL_ERROR": {
		"JSON_VALUE(content, '$.tool') AS tool_name",
		"JSON_QUERY(content, '$.args') AS tool_args",
		"JSON_VALUE(content, '$.tool_origin') AS tool_origin",
		"CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64) AS total_ms",
	},
	"AGENT_START": {
		"JSON_VALUE(content, '$.text_summary') AS agent_instruction",
	},
	"AGENT_END": {
		"CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64) AS total_ms",
	},
	"INVOCATION_START": {},
	"INVOCATION_END":   {},
	"EVENT":            {},
	"STATE_DELTA": {
		"JSON_QUERY(attributes, '$.state_delta') AS state_delta",
	},
	"HITL_CREDENTIAL_REQUEST": {
		"JSON_VALUE(content, '$.tool') AS tool_name",
		"JSON_QUERY(content, '$.args') AS tool_args",
	},
	"HITL_CONFIRMATION_REQUEST": {
		"JSON_VALUE(content, '$.tool') AS tool_name",
		"JSON_QUERY(content, '$.args') AS tool_args",
	},
	"HITL_INPUT_REQUEST": {
		"JSON_VALUE(content, '$.tool') AS tool_name",
		"JSON_QUERY(content, '$.args') AS tool_args",
	},
	"A2A_INTERACTION": {
		"content AS response_content",
		"JSON_VALUE(attributes, '$.a2a_metadata.\"a2a:task_id\"') AS a2a_task_id",
		"JSON_VALUE(attributes, '$.a2a_metadata.\"a2a:context_id\"') AS a2a_context_id",
		"JSON_QUERY(attributes, '$.a2a_metadata.\"a2a:request\"') AS a2a_request",
		"JSON_QUERY(attributes, '$.a2a_metadata.\"a2a:response\"') AS a2a_response",
	},
}

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
	meta, err := tableRef.Metadata(ctx)
	if err != nil {
		if apiErr, ok := err.(*googleapi.Error); ok && apiErr.Code == http.StatusNotFound {
			if config.Logger != nil {
				config.Logger.Printf("Table %s not found. Creating it...", config.TableName)
			}
			err = tableRef.Create(ctx, &bq.TableMetadata{
				Schema: EventsSchema(),
				TimePartitioning: &bq.TimePartitioning{
					Field: "timestamp",
					Type:  bq.DayPartitioningType,
				},
				Clustering: &bq.Clustering{
					Fields: config.ClusteringFields,
				},
				Labels: map[string]string{
					schemaVersionLabelKey: schemaVersion,
				},
			})
			if err != nil {
				return nil, fmt.Errorf("failed to create BigQuery table: %w", err)
			}
		} else {
			return nil, fmt.Errorf("failed to retrieve BigQuery table metadata: %w", err)
		}
	} else {
		if config.AutoSchemaUpgrade {
			if err := maybeUpgradeSchema(ctx, tableRef, meta, config.Logger); err != nil {
				return nil, fmt.Errorf("failed to auto-upgrade BigQuery table schema: %w", err)
			}
		}
	}

	if config.CreateViews {
		if err := createAnalyticsViews(ctx, bqClient, config); err != nil && config.Logger != nil {
			config.Logger.Printf("Views creation failed: %v", err)
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

		BeforeAgentCallback: func(ctx agent.Context) (*genai.Content, error) {
			logEvent(ctx, "AGENT_START", nil, nil)
			return nil, nil
		},
		AfterAgentCallback: func(ctx agent.Context) (*genai.Content, error) {
			logEvent(ctx, "AGENT_END", nil, nil)
			return nil, nil
		},

		BeforeModelCallback: func(ctx agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
			attrs := map[string]any{"model": "unknown"}
			if req.Model != "" {
				attrs["model"] = req.Model
			}
			logEvent(ctx, "MODEL_REQUEST", req, attrs)
			return nil, nil
		},
		AfterModelCallback: func(ctx agent.Context, res *model.LLMResponse, err error) (*model.LLMResponse, error) {
			if res != nil && res.Partial {
				return nil, nil
			}
			attrs := map[string]any{}
			if res != nil && res.UsageMetadata != nil {
				attrs["usage_metadata"] = res.UsageMetadata
			}
			logEvent(ctx, "MODEL_RESPONSE", res, attrs)
			return nil, nil
		},
		OnModelErrorCallback: func(ctx agent.Context, req *model.LLMRequest, err error) (*model.LLMResponse, error) {
			attrs := map[string]any{"error_message": err.Error()}
			logEvent(ctx, "MODEL_ERROR", nil, attrs)
			return nil, nil
		},

		BeforeToolCallback: func(ctx agent.Context, t tool.Tool, args map[string]any) (map[string]any, error) {
			attrs := map[string]any{"tool_name": t.Name()}
			logEvent(ctx, "TOOL_START", args, attrs)
			return nil, nil
		},
		AfterToolCallback: func(ctx agent.Context, t tool.Tool, args, res map[string]any, err error) (map[string]any, error) {
			attrs := map[string]any{"tool_name": t.Name()}
			logEvent(ctx, "TOOL_END", res, attrs)
			return nil, nil
		},
		OnToolErrorCallback: func(ctx agent.Context, t tool.Tool, args map[string]any, err error) (map[string]any, error) {
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

func schemaFieldsMatch(existing, desired bq.Schema) (newFields, updatedRecords bq.Schema) {
	existingByName := make(map[string]*bq.FieldSchema)
	for _, f := range existing {
		existingByName[f.Name] = f
	}

	for _, desiredField := range desired {
		existingField, exists := existingByName[desiredField.Name]
		if !exists {
			newFields = append(newFields, desiredField)
		} else if desiredField.Type == bq.RecordFieldType && existingField.Type == bq.RecordFieldType && len(desiredField.Schema) > 0 {
			subNew, subUpdated := schemaFieldsMatch(existingField.Schema, desiredField.Schema)
			if len(subNew) > 0 || len(subUpdated) > 0 {
				// Build a merged sub-field list
				mergedSub := make(bq.Schema, len(existingField.Schema))
				copy(mergedSub, existingField.Schema)

				updatedNames := make(map[string]bool)
				for _, f := range subUpdated {
					updatedNames[f.Name] = true
				}

				// Replace updated nested records in-place
				for i, f := range mergedSub {
					if updatedNames[f.Name] {
						for _, u := range subUpdated {
							if u.Name == f.Name {
								mergedSub[i] = u
								break
							}
						}
					}
				}

				// Append entirely new sub-fields
				mergedSub = append(mergedSub, subNew...)

				// Create updated RECORD field schema
				updatedField := *existingField
				updatedField.Schema = mergedSub
				updatedRecords = append(updatedRecords, &updatedField)
			}
		}
	}
	return newFields, updatedRecords
}

func maybeUpgradeSchema(ctx context.Context, existingTable *bq.Table, meta *bq.TableMetadata, logger Logger) error {
	storedVersion := meta.Labels[schemaVersionLabelKey]
	if storedVersion == schemaVersion {
		return nil
	}

	newFields, updatedRecords := schemaFieldsMatch(meta.Schema, EventsSchema())

	if len(newFields) > 0 || len(updatedRecords) > 0 {
		updatedNames := make(map[string]bool)
		for _, f := range updatedRecords {
			updatedNames[f.Name] = true
		}

		merged := make(bq.Schema, 0, len(meta.Schema)+len(newFields))
		for _, f := range meta.Schema {
			if updatedNames[f.Name] {
				for _, u := range updatedRecords {
					if u.Name == f.Name {
						merged = append(merged, u)
						break
					}
				}
			} else {
				merged = append(merged, f)
			}
		}
		merged = append(merged, newFields...)

		// Update schema in metadata update
		update := bq.TableMetadataToUpdate{
			Schema: merged,
		}
		update.SetLabel(schemaVersionLabelKey, schemaVersion)

		_, err := existingTable.Update(ctx, update, meta.ETag)
		if err != nil {
			return fmt.Errorf("failed to update table schema: %w", err)
		}

		if logger != nil {
			var changeDesc []string
			if len(newFields) > 0 {
				var names []string
				for _, f := range newFields {
					names = append(names, f.Name)
				}
				changeDesc = append(changeDesc, fmt.Sprintf("new columns %v", names))
			}
			if len(updatedRecords) > 0 {
				var names []string
				for _, f := range updatedRecords {
					names = append(names, f.Name)
				}
				changeDesc = append(changeDesc, fmt.Sprintf("updated RECORD fields %v", names))
			}
			logger.Printf("Auto-upgraded table schema: %s", strings.Join(changeDesc, ", "))
		}
	} else {
		// No schema upgrade needed, just update label if missing or outdated
		if storedVersion != schemaVersion {
			update := bq.TableMetadataToUpdate{}
			update.SetLabel(schemaVersionLabelKey, schemaVersion)

			_, err := existingTable.Update(ctx, update, meta.ETag)
			if err != nil {
				return fmt.Errorf("failed to update table labels: %w", err)
			}
		}
	}
	return nil
}

func createAnalyticsViews(ctx context.Context, client *bq.Client, config Config) error {
	for eventType, extraCols := range eventViewDefs {
		viewName := fmt.Sprintf("%s_%s", config.ViewPrefix, strings.ToLower(eventType))

		allCols := make([]string, 0, len(viewCommonColumns)+len(extraCols))
		allCols = append(allCols, viewCommonColumns...)
		allCols = append(allCols, extraCols...)
		columnsStr := strings.Join(allCols, ",\n  ")

		sql := fmt.Sprintf(
			viewSQLTemplate,
			config.ProjectID, config.DatasetID, viewName,
			columnsStr,
			config.ProjectID, config.DatasetID, config.TableName,
			eventType,
		)

		q := client.Query(sql)
		job, err := q.Run(ctx)
		if err != nil {
			return fmt.Errorf("failed to run view creation query for %s: %w", viewName, err)
		}
		status, err := job.Wait(ctx)
		if err != nil {
			return fmt.Errorf("view creation query failed for %s: %w", viewName, err)
		}
		if err := status.Err(); err != nil {
			return fmt.Errorf("view creation query execution failed for %s: %w", viewName, err)
		}
	}
	return nil
}
