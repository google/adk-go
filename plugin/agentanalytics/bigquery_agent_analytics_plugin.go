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
	"strings"
	"time"

	bq "cloud.google.com/go/bigquery"
	bqstorage "cloud.google.com/go/bigquery/storage/apiv1"
	gcs "cloud.google.com/go/storage"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/api/option"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	baseplugin "google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
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
			TimePartitioning: &bq.TimePartitioning{
				Field: "timestamp",
				Type:  bq.DayPartitioningType,
			},
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

	// Initialize GCS client if bucket offload is enabled
	var offloader *gcsOffloader
	if config.GCSBucketName != "" {
		var err error
		offloader, err = newGCSOffloader(ctx, config.ProjectID, config.GCSBucketName, config.ClientOptions...)
		if err != nil {
			return nil, fmt.Errorf("failed to create GCS client: %w", err)
		}
	}

	// Determine destination stream
	streamName := fmt.Sprintf("projects/%s/datasets/%s/tables/%s/streams/_default", config.ProjectID, config.DatasetID, config.TableName)

	processor, err := NewBatchProcessor(ctx, writeClient, streamName, config, offloader)
	if err != nil {
		if closeErr := writeClient.Close(); closeErr != nil {
			if config.Logger != nil {
				config.Logger.Printf("failed to close BigQuery write client: %v", closeErr)
			}
		}
		if offloader != nil {
			_ = offloader.close()
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

		// Store raw content and extra attributes to be parsed asynchronously in the background!
		row["_raw_content"] = content

		attrsCloned := make(map[string]any, len(extraAttrs))
		for k, v := range extraAttrs {
			attrsCloned[k] = v
		}
		row["_extra_attrs"] = attrsCloned
		row["is_truncated"] = false

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
			if offloader != nil {
				_ = offloader.close()
			}
			return nil
		},
	}

	return baseplugin.New(cfg)
}

type gcsOffloader struct {
	client *gcs.Client
	bucket *gcs.BucketHandle
}

func newGCSOffloader(ctx context.Context, projectID, bucketName string, opts ...option.ClientOption) (*gcsOffloader, error) {
	client, err := gcs.NewClient(ctx, opts...)
	if err != nil {
		return nil, err
	}
	return &gcsOffloader{
		client: client,
		bucket: client.Bucket(bucketName),
	}, nil
}

func (o *gcsOffloader) uploadContent(ctx context.Context, data []byte, contentType, path string) (string, error) {
	obj := o.bucket.Object(path)
	w := obj.NewWriter(ctx)
	w.ContentType = contentType

	if _, err := w.Write(data); err != nil {
		_ = w.Close()
		return "", err
	}
	if err := w.Close(); err != nil {
		return "", err
	}
	return fmt.Sprintf("gs://%s/%s", o.bucket.BucketName(), path), nil
}

func (o *gcsOffloader) close() error {
	if o.client != nil {
		return o.client.Close()
	}
	return nil
}

type hybridContentParser struct {
	offloader       *gcsOffloader
	connectionID    string
	maxLen          int
	inlineTextLimit int
}

func newHybridContentParser(offloader *gcsOffloader, connectionID string, maxLen int) *hybridContentParser {
	return &hybridContentParser{
		offloader:       offloader,
		connectionID:    connectionID,
		maxLen:          maxLen,
		inlineTextLimit: 32 * 1024,
	}
}

func (p *hybridContentParser) parseContentObject(ctx context.Context, content *genai.Content, traceID, spanID string) (string, []map[string]any, bool) {
	if content == nil || len(content.Parts) == 0 {
		return "", nil, false
	}

	var contentParts []map[string]any
	var summaryText []string
	isTruncated := false

	for idx, part := range content.Parts {
		partData := map[string]any{
			"part_index":      int64(idx),
			"mime_type":       "text/plain",
			"uri":             nil,
			"text":            nil,
			"part_attributes": "{}",
			"storage_mode":    "INLINE",
			"object_ref":      nil,
		}

		if part.FileData != nil {
			partData["storage_mode"] = "EXTERNAL_URI"
			partData["uri"] = part.FileData.FileURI
			partData["mime_type"] = part.FileData.MIMEType
		} else if part.InlineData != nil {
			if p.offloader != nil {
				ext := ".bin"
				switch part.InlineData.MIMEType {
				case "image/png":
					ext = ".png"
				case "image/jpeg":
					ext = ".jpg"
				}

				path := fmt.Sprintf("%s/%s/%s_p%d%s", time.Now().Format("2006-01-02"), traceID, spanID, idx, ext)
				uri, err := p.offloader.uploadContent(ctx, part.InlineData.Data, part.InlineData.MIMEType, path)
				if err == nil {
					partData["storage_mode"] = "GCS_REFERENCE"
					partData["uri"] = uri
					partData["text"] = "[MEDIA OFFLOADED]"
					partData["mime_type"] = part.InlineData.MIMEType

					detailsMap := map[string]any{
						"gcs_metadata": map[string]any{"content_type": part.InlineData.MIMEType},
					}
					detailsRaw, _, _ := SmartTruncate(detailsMap, p.maxLen)

					objectRef := map[string]any{
						"uri":        uri,
						"version":    nil,
						"authorizer": p.connectionID,
						"details":    string(detailsRaw),
					}
					partData["object_ref"] = objectRef
				} else {
					partData["text"] = "[UPLOAD FAILED]"
				}
			} else {
				partData["text"] = "[BINARY DATA]"
			}
		} else if part.Text != "" {
			textLen := len([]byte(part.Text))
			threshold := p.inlineTextLimit
			if p.maxLen != -1 && p.maxLen < threshold {
				threshold = p.maxLen
			}

			if p.offloader != nil && textLen > threshold {
				path := fmt.Sprintf("%s/%s/%s_p%d.txt", time.Now().Format("2006-01-02"), traceID, spanID, idx)
				uri, err := p.offloader.uploadContent(ctx, []byte(part.Text), "text/plain", path)
				if err == nil {
					partData["storage_mode"] = "GCS_REFERENCE"
					partData["uri"] = uri
					partData["mime_type"] = "text/plain"

					truncatedText, _ := truncateString(part.Text, 200)
					partData["text"] = truncatedText + "... [OFFLOADED]"

					detailsMap := map[string]any{
						"gcs_metadata": map[string]any{"content_type": "text/plain"},
					}
					detailsRaw, _, _ := SmartTruncate(detailsMap, p.maxLen)

					objectRef := map[string]any{
						"uri":        uri,
						"version":    nil,
						"authorizer": p.connectionID,
						"details":    string(detailsRaw),
					}
					partData["object_ref"] = objectRef
				} else {
					cleanText, trunc := truncateString(part.Text, p.maxLen)
					isTruncated = isTruncated || trunc
					partData["text"] = cleanText
					summaryText = append(summaryText, cleanText)
				}
			} else {
				cleanText, trunc := truncateString(part.Text, p.maxLen)
				isTruncated = isTruncated || trunc
				partData["text"] = cleanText
				summaryText = append(summaryText, cleanText)
			}
		} else if part.FunctionCall != nil {
			partData["mime_type"] = "application/json"
			partData["text"] = fmt.Sprintf("Function: %s", part.FunctionCall.Name)
			attrsMap := map[string]any{"function_name": part.FunctionCall.Name}
			attrsRaw, _, _ := SmartTruncate(attrsMap, p.maxLen)
			partData["part_attributes"] = string(attrsRaw)
		}

		contentParts = append(contentParts, partData)
	}

	summaryStr, trunc := truncateString(strings.Join(summaryText, " | "), p.maxLen)
	isTruncated = isTruncated || trunc

	return summaryStr, contentParts, isTruncated
}

func (p *hybridContentParser) parse(ctx context.Context, content any, traceID, spanID string) (any, []map[string]any, bool) {
	if content == nil {
		return nil, nil, false
	}

	if c, ok := content.(*genai.Content); ok {
		summary, parts, trunc := p.parseContentObject(ctx, c, traceID, spanID)
		return map[string]any{"text_summary": summary}, parts, trunc
	}

	if req, ok := content.(*model.LLMRequest); ok {
		var contentParts []map[string]any
		var messages []map[string]any
		isTruncated := false

		for _, c := range req.Contents {
			role := c.Role
			if role == "" {
				role = "unknown"
			}
			summary, parts, trunc := p.parseContentObject(ctx, c, traceID, spanID)
			isTruncated = isTruncated || trunc
			contentParts = append(contentParts, parts...)
			messages = append(messages, map[string]any{"role": role, "content": summary})
		}

		jsonPayload := make(map[string]any)
		if len(messages) > 0 {
			jsonPayload["prompt"] = messages
		}

		if req.Config != nil && req.Config.SystemInstruction != nil {
			si := req.Config.SystemInstruction
			summary, parts, trunc := p.parseContentObject(ctx, si, traceID, spanID)
			isTruncated = isTruncated || trunc
			contentParts = append(contentParts, parts...)
			jsonPayload["system_prompt"] = summary
		}

		return jsonPayload, contentParts, isTruncated
	}

	if res, ok := content.(*model.LLMResponse); ok {
		if res.Content != nil {
			summary, parts, trunc := p.parseContentObject(ctx, res.Content, traceID, spanID)
			return map[string]any{"text_summary": summary}, parts, trunc
		}
		return nil, nil, false
	}

	if str, ok := content.(string); ok {
		truncated, trunc := truncateString(str, p.maxLen)
		return truncated, nil, trunc
	}

	truncObj, truncated, _ := recursiveSmartTruncate(content, p.maxLen)
	return truncObj, nil, truncated
}
