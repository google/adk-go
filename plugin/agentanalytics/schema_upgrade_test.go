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
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"

	bq "cloud.google.com/go/bigquery"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/api/googleapi"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type metadataResponse struct {
	md  *bq.TableMetadata
	err error
}

type updateCall struct {
	update bq.TableMetadataToUpdate
	etag   string
}

type updateResponse struct {
	md  *bq.TableMetadata
	err error
}

type mockTableOps struct {
	metadataSeq   []metadataResponse
	metadataCalls int

	createErr   error
	createCalls int
	createdMeta *bq.TableMetadata

	updateSeq   []updateResponse
	updateCalls int
	updates     []updateCall
}

func (m *mockTableOps) Metadata(_ context.Context) (*bq.TableMetadata, error) {
	idx := m.metadataCalls
	m.metadataCalls++
	if idx < len(m.metadataSeq) {
		return m.metadataSeq[idx].md, m.metadataSeq[idx].err
	}
	if len(m.metadataSeq) > 0 {
		last := m.metadataSeq[len(m.metadataSeq)-1]
		return last.md, last.err
	}
	return &bq.TableMetadata{Schema: EventsSchema()}, nil
}

func (m *mockTableOps) Create(_ context.Context, meta *bq.TableMetadata) error {
	m.createCalls++
	m.createdMeta = meta
	return m.createErr
}

func (m *mockTableOps) Update(_ context.Context, update bq.TableMetadataToUpdate, etag string) (*bq.TableMetadata, error) {
	idx := m.updateCalls
	m.updateCalls++
	m.updates = append(m.updates, updateCall{update: update, etag: etag})
	if idx < len(m.updateSeq) {
		return m.updateSeq[idx].md, m.updateSeq[idx].err
	}
	if len(m.updateSeq) > 0 {
		last := m.updateSeq[len(m.updateSeq)-1]
		return last.md, last.err
	}
	return &bq.TableMetadata{Schema: update.Schema}, nil
}

func TestEnsureTableAndSchema_NotFoundCreatesTableWithVersionLabel(t *testing.T) {
	for _, disableUpgrade := range []bool{true, false} {
		name := "DisableSchemaUpgrade=false"
		if disableUpgrade {
			name = "DisableSchemaUpgrade=true"
		}
		t.Run(name, func(t *testing.T) {
			cfg := DefaultConfig()
			cfg.DisableSchemaUpgrade = disableUpgrade
			cfg.TableName = "agent_events"
			cfg.ClusteringFields = []string{"event_type", "agent"}

			table := &mockTableOps{
				metadataSeq: []metadataResponse{
					{err: &googleapi.Error{Code: http.StatusNotFound, Message: "not found"}},
				},
			}

			if err := ensureTableAndSchema(context.Background(), table, cfg, nil); err != nil {
				t.Fatalf("ensureTableAndSchema() error = %v", err)
			}
			if table.createCalls != 1 {
				t.Fatalf("createCalls = %d, want 1", table.createCalls)
			}
			if table.updateCalls != 0 {
				t.Errorf("updateCalls = %d, want 0", table.updateCalls)
			}
			if got := table.createdMeta.Labels[schemaVersionKey]; got != SchemaVersion {
				t.Errorf("created table label %q = %q, want %q", schemaVersionKey, got, SchemaVersion)
			}
			if table.createdMeta.TimePartitioning == nil || table.createdMeta.TimePartitioning.Field != "timestamp" || table.createdMeta.TimePartitioning.Type != bq.DayPartitioningType {
				t.Errorf("TimePartitioning = %+v, want DAY on timestamp", table.createdMeta.TimePartitioning)
			}
			if diff := cmp.Diff(cfg.ClusteringFields, table.createdMeta.Clustering.Fields); diff != "" {
				t.Errorf("Clustering.Fields mismatch (-want +got):\n%s", diff)
			}
			if len(table.createdMeta.Schema) != len(EventsSchema()) {
				t.Errorf("created schema field count = %d, want %d", len(table.createdMeta.Schema), len(EventsSchema()))
			}
		})
	}
}

func TestEnsureTableAndSchema_ConflictOnCreateRecovery(t *testing.T) {
	t.Run("DisableSchemaUpgrade=true treats 409 as success without re-fetching", func(t *testing.T) {
		cfg := DefaultConfig()
		cfg.DisableSchemaUpgrade = true
		table := &mockTableOps{
			metadataSeq: []metadataResponse{
				{err: &googleapi.Error{Code: http.StatusNotFound}},
			},
			createErr: &googleapi.Error{Code: http.StatusConflict, Message: "already exists"},
		}
		if err := ensureTableAndSchema(context.Background(), table, cfg, nil); err != nil {
			t.Fatalf("ensureTableAndSchema() error = %v", err)
		}
		if table.metadataCalls != 1 {
			t.Errorf("metadataCalls = %d, want 1", table.metadataCalls)
		}
		if table.updateCalls != 0 {
			t.Errorf("updateCalls = %d, want 0", table.updateCalls)
		}
	})

	t.Run("default config re-fetches metadata and skips Update when clean", func(t *testing.T) {
		cfg := DefaultConfig()
		table := &mockTableOps{
			metadataSeq: []metadataResponse{
				{err: &googleapi.Error{Code: http.StatusNotFound}},
				{md: &bq.TableMetadata{Schema: EventsSchema(), ETag: "etag-after-409"}},
			},
			createErr: &googleapi.Error{Code: http.StatusConflict, Message: "already exists"},
		}
		if err := ensureTableAndSchema(context.Background(), table, cfg, nil); err != nil {
			t.Fatalf("ensureTableAndSchema() error = %v", err)
		}
		if table.metadataCalls != 2 {
			t.Errorf("metadataCalls = %d, want 2", table.metadataCalls)
		}
		if table.updateCalls != 0 {
			t.Errorf("updateCalls = %d, want 0", table.updateCalls)
		}
	})

	t.Run("default config re-fetches metadata and upgrades outdated schema", func(t *testing.T) {
		cfg := DefaultConfig()
		table := &mockTableOps{
			metadataSeq: []metadataResponse{
				{err: &googleapi.Error{Code: http.StatusNotFound}},
				{md: &bq.TableMetadata{
					Schema: bq.Schema{{Name: "timestamp", Type: bq.TimestampFieldType, Required: true}},
					ETag:   "etag-after-409",
				}},
			},
			createErr: &googleapi.Error{Code: http.StatusConflict, Message: "already exists"},
		}
		if err := ensureTableAndSchema(context.Background(), table, cfg, nil); err != nil {
			t.Fatalf("ensureTableAndSchema() error = %v", err)
		}
		if table.updateCalls != 1 {
			t.Fatalf("updateCalls = %d, want 1", table.updateCalls)
		}
		if table.updates[0].etag != "etag-after-409" {
			t.Errorf("update etag = %q, want %q", table.updates[0].etag, "etag-after-409")
		}
	})

	t.Run("default config fails closed on 403 re-fetch failure after 409", func(t *testing.T) {
		cfg := DefaultConfig()
		table := &mockTableOps{
			metadataSeq: []metadataResponse{
				{err: &googleapi.Error{Code: http.StatusNotFound}},
				{err: &googleapi.Error{Code: http.StatusForbidden, Message: "access denied"}},
			},
			createErr: status.Error(codes.AlreadyExists, "already exists"),
		}
		err := ensureTableAndSchema(context.Background(), table, cfg, nil)
		if err == nil {
			t.Fatal("ensureTableAndSchema() error = nil, want 403 error after 409")
		}
		var apiErr *googleapi.Error
		if !errors.As(err, &apiErr) || apiErr.Code != http.StatusForbidden {
			t.Errorf("ensureTableAndSchema() error = %v, want wrapped 403 googleapi.Error", err)
		}
	})

	t.Run("default config propagates non-403 re-fetch failure after 409", func(t *testing.T) {
		cfg := DefaultConfig()
		table := &mockTableOps{
			metadataSeq: []metadataResponse{
				{err: &googleapi.Error{Code: http.StatusNotFound}},
				{err: context.Canceled},
			},
			createErr: status.Error(codes.AlreadyExists, "already exists"),
		}
		err := ensureTableAndSchema(context.Background(), table, cfg, nil)
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("ensureTableAndSchema() error = %v, want wrapped context.Canceled", err)
		}
	})
}

func TestEnsureTableAndSchema_PermissionDeniedFailsClosed(t *testing.T) {
	tests := []struct {
		name           string
		disableUpgrade bool
		table          *mockTableOps
		wantErr        bool
	}{
		{
			name:           "HTTP 403 on Metadata with DisableSchemaUpgrade=true returns error",
			disableUpgrade: true,
			table: &mockTableOps{
				metadataSeq: []metadataResponse{{err: &googleapi.Error{Code: http.StatusForbidden, Message: "access denied"}}},
			},
			wantErr: true,
		},
		{
			name: "gRPC PermissionDenied on Metadata with default config returns error",
			table: &mockTableOps{
				metadataSeq: []metadataResponse{{err: status.Error(codes.PermissionDenied, "denied")}},
			},
			wantErr: true,
		},
		{
			name:           "HTTP 403 on Create after 404 returns error because table is known missing",
			disableUpgrade: true,
			table: &mockTableOps{
				metadataSeq: []metadataResponse{{err: &googleapi.Error{Code: http.StatusNotFound}}},
				createErr:   &googleapi.Error{Code: http.StatusForbidden, Message: "cannot create"},
			},
			wantErr: true,
		},
		{
			name: "HTTP 403 on Update during schema upgrade returns error because schema is known incomplete",
			table: &mockTableOps{
				metadataSeq: []metadataResponse{{md: &bq.TableMetadata{
					Schema: bq.Schema{{Name: "timestamp", Type: bq.TimestampFieldType, Required: true}},
					ETag:   "etag-1",
				}}},
				updateSeq: []updateResponse{{err: &googleapi.Error{Code: http.StatusForbidden, Message: "cannot update"}}},
			},
			wantErr: true,
		},
		{
			name: "HTTP 500 on Metadata returns error",
			table: &mockTableOps{
				metadataSeq: []metadataResponse{{err: &googleapi.Error{Code: http.StatusInternalServerError, Message: "internal"}}},
			},
			wantErr: true,
		},
		{
			name:           "HTTP 500 on Create after 404 returns error",
			disableUpgrade: true,
			table: &mockTableOps{
				metadataSeq: []metadataResponse{{err: &googleapi.Error{Code: http.StatusNotFound}}},
				createErr:   &googleapi.Error{Code: http.StatusInternalServerError, Message: "internal"},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := DefaultConfig()
			cfg.DisableSchemaUpgrade = tt.disableUpgrade

			err := ensureTableAndSchema(context.Background(), tt.table, cfg, nil)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ensureTableAndSchema() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestEnsureTableAndSchema_NoOpSchemaDoesNotCallUpdate(t *testing.T) {
	tests := []struct {
		name   string
		md     *bq.TableMetadata
		mutate func(*Config)
	}{
		{
			name: "exact schema with nil labels does not call Update",
			md:   &bq.TableMetadata{Schema: EventsSchema(), Labels: nil, ETag: "etag-clean"},
		},
		{
			name: "exact schema with stale adk_schema_version label does not call Update",
			md: &bq.TableMetadata{
				Schema: EventsSchema(),
				Labels: map[string]string{schemaVersionKey: "1"},
				ETag:   "etag-stale-label",
			},
		},
		{
			name: "case-insensitive field names plus extra user column does not call Update",
			md: func() *bq.TableMetadata {
				var custom bq.Schema
				custom = append(custom, &bq.FieldSchema{Name: "custom_user_col", Type: bq.StringFieldType})
				for _, f := range EventsSchema() {
					cp := *f
					cp.Name = strings.ToUpper(f.Name)
					custom = append(custom, &cp)
				}
				return &bq.TableMetadata{Schema: custom, ETag: "etag-custom"}
			}(),
		},
		{
			name: "outdated schema with DisableSchemaUpgrade=true does not call Update",
			md: &bq.TableMetadata{
				Schema: bq.Schema{{Name: "timestamp", Type: bq.TimestampFieldType}},
				ETag:   "etag-disable-flag",
			},
			mutate: func(c *Config) { c.DisableSchemaUpgrade = true },
		},
		{
			name: "existing table with NULLABLE timestamp and DisableSchemaUpgrade=false does not call Update",
			md: func() *bq.TableMetadata {
				schema := EventsSchema()
				for i, f := range schema {
					if f.Name == "timestamp" {
						cp := *f
						cp.Required = false
						schema[i] = &cp
					}
				}
				return &bq.TableMetadata{Schema: schema, ETag: "etag-nullable-ts"}
			}(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := DefaultConfig()
			if tt.mutate != nil {
				tt.mutate(&cfg)
			}
			table := &mockTableOps{
				metadataSeq: []metadataResponse{{md: tt.md}},
			}
			if err := ensureTableAndSchema(context.Background(), table, cfg, nil); err != nil {
				t.Fatalf("ensureTableAndSchema() error = %v", err)
			}
			if table.updateCalls != 0 {
				t.Errorf("updateCalls = %d, want 0 (must never call Update when schema is unchanged or upgrade is disabled)", table.updateCalls)
			}
		})
	}
}

func TestEnsureTableAndSchema_AdditiveTopLevelAndNestedRecordUpgrade(t *testing.T) {
	// Build a legacy v1 table schema that:
	// 1. Has a custom column at index 0 ("custom_tenant_id") to verify order preservation.
	// 2. Lacks top-level "event_id".
	// 3. Has "content_parts" RECORD, but without nested "object_ref" or "storage_mode".
	var legacySchema bq.Schema
	legacySchema = append(legacySchema, &bq.FieldSchema{
		Name: "custom_tenant_id",
		Type: bq.StringFieldType,
	})
	for _, f := range EventsSchema() {
		if f.Name == "event_id" {
			continue
		}
		if f.Name == "content_parts" {
			cp := *f
			cp.Schema = bq.Schema{
				{Name: "mime_type", Type: bq.StringFieldType},
				{Name: "uri", Type: bq.StringFieldType},
				{Name: "text", Type: bq.StringFieldType},
				{Name: "part_index", Type: bq.IntegerFieldType},
				{Name: "part_attributes", Type: bq.StringFieldType},
			}
			legacySchema = append(legacySchema, &cp)
			continue
		}
		cp := *f
		legacySchema = append(legacySchema, &cp)
	}

	cfg := DefaultConfig()
	table := &mockTableOps{
		metadataSeq: []metadataResponse{{
			md: &bq.TableMetadata{
				Schema: legacySchema,
				ETag:   "etag-v1",
			},
		}},
	}

	if err := ensureTableAndSchema(context.Background(), table, cfg, nil); err != nil {
		t.Fatalf("ensureTableAndSchema() error = %v", err)
	}
	if table.updateCalls != 1 {
		t.Fatalf("updateCalls = %d, want 1", table.updateCalls)
	}

	gotCall := table.updates[0]
	if gotCall.etag != "etag-v1" {
		t.Errorf("etag = %q, want %q", gotCall.etag, "etag-v1")
	}

	updatedSchema := gotCall.update.Schema
	// Custom column at index 0 must be preserved.
	if updatedSchema[0].Name != "custom_tenant_id" {
		t.Errorf("updatedSchema[0].Name = %q, want %q", updatedSchema[0].Name, "custom_tenant_id")
	}

	// Missing top-level "event_id" must be appended at the end as NULLABLE.
	lastField := updatedSchema[len(updatedSchema)-1]
	if lastField.Name != "event_id" || lastField.Type != bq.StringFieldType || lastField.Required {
		t.Errorf("appended field = %+v, want NULLABLE STRING event_id", lastField)
	}

	// "content_parts" must have been updated in-place with nested "object_ref" and "storage_mode".
	var contentParts *bq.FieldSchema
	for _, f := range updatedSchema {
		if f.Name == "content_parts" {
			contentParts = f
			break
		}
	}
	if contentParts == nil {
		t.Fatal("content_parts missing from updatedSchema")
	}
	subNames := make(map[string]*bq.FieldSchema, len(contentParts.Schema))
	for _, sf := range contentParts.Schema {
		subNames[sf.Name] = sf
	}
	objRef, ok := subNames["object_ref"]
	if !ok || objRef.Type != bq.RecordFieldType || objRef.Required {
		t.Fatalf("content_parts.object_ref = %+v, want NULLABLE RECORD", objRef)
	}
	if len(objRef.Schema) != 4 {
		t.Errorf("len(object_ref.Schema) = %d, want 4", len(objRef.Schema))
	}
	for _, rf := range objRef.Schema {
		if rf.Required {
			t.Errorf("object_ref.%s Required = true, want false", rf.Name)
		}
	}
	if sm, ok := subNames["storage_mode"]; !ok || sm.Type != bq.StringFieldType || sm.Required {
		t.Errorf("content_parts.storage_mode = %+v, want NULLABLE STRING", sm)
	}
}

func TestEnsureTableAndSchema_PreconditionFailedRetriesWithFreshETag(t *testing.T) {
	outdated := bq.Schema{{Name: "timestamp", Type: bq.TimestampFieldType, Required: true}}

	t.Run("succeeds on retry with fresh ETag", func(t *testing.T) {
		cfg := DefaultConfig()
		table := &mockTableOps{
			metadataSeq: []metadataResponse{
				{md: &bq.TableMetadata{Schema: outdated, ETag: "etag-1"}},
				{md: &bq.TableMetadata{Schema: outdated, ETag: "etag-2"}},
			},
			updateSeq: []updateResponse{
				{err: &googleapi.Error{Code: http.StatusPreconditionFailed, Message: "etag mismatch"}},
				{md: &bq.TableMetadata{Schema: EventsSchema(), ETag: "etag-3"}},
			},
		}
		if err := ensureTableAndSchema(context.Background(), table, cfg, nil); err != nil {
			t.Fatalf("ensureTableAndSchema() error = %v", err)
		}
		if table.metadataCalls != 2 {
			t.Errorf("metadataCalls = %d, want 2", table.metadataCalls)
		}
		if table.updateCalls != 2 {
			t.Fatalf("updateCalls = %d, want 2", table.updateCalls)
		}
		if table.updates[0].etag != "etag-1" || table.updates[1].etag != "etag-2" {
			t.Errorf("update etags = [%q, %q], want [etag-1, etag-2]", table.updates[0].etag, table.updates[1].etag)
		}
	})

	t.Run("stops without second Update when winning replica already upgraded schema", func(t *testing.T) {
		cfg := DefaultConfig()
		table := &mockTableOps{
			metadataSeq: []metadataResponse{
				{md: &bq.TableMetadata{Schema: outdated, ETag: "etag-1"}},
				{md: &bq.TableMetadata{Schema: EventsSchema(), ETag: "etag-2"}},
			},
			updateSeq: []updateResponse{
				{err: status.Error(codes.FailedPrecondition, "etag mismatch")},
			},
		}
		if err := ensureTableAndSchema(context.Background(), table, cfg, nil); err != nil {
			t.Fatalf("ensureTableAndSchema() error = %v", err)
		}
		if table.metadataCalls != 2 {
			t.Errorf("metadataCalls = %d, want 2", table.metadataCalls)
		}
		if table.updateCalls != 1 {
			t.Errorf("updateCalls = %d, want 1", table.updateCalls)
		}
	})

	t.Run("returns error after exhausting 2 precondition retries", func(t *testing.T) {
		cfg := DefaultConfig()
		table := &mockTableOps{
			metadataSeq: []metadataResponse{
				{md: &bq.TableMetadata{Schema: outdated, ETag: "etag-1"}},
				{md: &bq.TableMetadata{Schema: outdated, ETag: "etag-2"}},
				{md: &bq.TableMetadata{Schema: outdated, ETag: "etag-3"}},
			},
			updateSeq: []updateResponse{
				{err: &googleapi.Error{Code: http.StatusPreconditionFailed}},
				{err: &googleapi.Error{Code: http.StatusPreconditionFailed}},
				{err: &googleapi.Error{Code: http.StatusPreconditionFailed}},
			},
		}
		err := ensureTableAndSchema(context.Background(), table, cfg, nil)
		if err == nil {
			t.Fatal("ensureTableAndSchema() error = nil, want precondition error after exhausting retries")
		}
		if table.updateCalls != 3 {
			t.Errorf("updateCalls = %d, want 3 (1 initial + 2 retries)", table.updateCalls)
		}
	})

	t.Run("returns error on 403 metadata re-fetch failure after 412", func(t *testing.T) {
		cfg := DefaultConfig()
		table := &mockTableOps{
			metadataSeq: []metadataResponse{
				{md: &bq.TableMetadata{Schema: outdated, ETag: "etag-1"}},
				{err: &googleapi.Error{Code: http.StatusForbidden}},
			},
			updateSeq: []updateResponse{
				{err: &googleapi.Error{Code: http.StatusPreconditionFailed}},
			},
		}
		err := ensureTableAndSchema(context.Background(), table, cfg, nil)
		if err == nil {
			t.Fatal("ensureTableAndSchema() error = nil, want 403 re-fetch error after 412")
		}
		var apiErr *googleapi.Error
		if !errors.As(err, &apiErr) || apiErr.Code != http.StatusForbidden {
			t.Errorf("ensureTableAndSchema() error = %v, want wrapped 403 googleapi.Error", err)
		}
	})

	t.Run("returns error on non-403 metadata re-fetch failure after 412", func(t *testing.T) {
		cfg := DefaultConfig()
		table := &mockTableOps{
			metadataSeq: []metadataResponse{
				{md: &bq.TableMetadata{Schema: outdated, ETag: "etag-1"}},
				{err: &googleapi.Error{Code: http.StatusInternalServerError}},
			},
			updateSeq: []updateResponse{
				{err: &googleapi.Error{Code: http.StatusPreconditionFailed}},
			},
		}
		if err := ensureTableAndSchema(context.Background(), table, cfg, nil); err == nil {
			t.Fatal("ensureTableAndSchema() error = nil, want re-fetch error")
		}
	})
}

func TestEnsureTableAndSchema_IncompatibleSchemaReturnsError(t *testing.T) {
	tests := []struct {
		name   string
		schema bq.Schema
		want   string
	}{
		{
			name: "top-level field type mismatch",
			schema: bq.Schema{
				{Name: "timestamp", Type: bq.TimestampFieldType, Required: true},
				{Name: "event_id", Type: bq.IntegerFieldType},
			},
			want: "incompatible BigQuery schema field event_id: existing=INTEGER/NULLABLE, desired=STRING/NULLABLE",
		},
		{
			name: "top-level field repeated mismatch",
			schema: bq.Schema{
				{Name: "timestamp", Type: bq.TimestampFieldType, Required: true},
				{Name: "content_parts", Type: bq.RecordFieldType, Repeated: false},
			},
			want: "incompatible BigQuery schema field content_parts: existing=RECORD/NULLABLE, desired=RECORD/REPEATED",
		},
		{
			name: "top-level REQUIRED field where writer allows NULL",
			schema: bq.Schema{
				{Name: "timestamp", Type: bq.TimestampFieldType, Required: true},
				{Name: "content", Type: bq.JSONFieldType, Required: true},
			},
			want: "incompatible BigQuery schema field content: existing=JSON/REQUIRED, desired=JSON/NULLABLE",
		},
		{
			name: "nested RECORD subfield type mismatch",
			schema: bq.Schema{
				{Name: "timestamp", Type: bq.TimestampFieldType, Required: true},
				{
					Name:     "content_parts",
					Type:     bq.RecordFieldType,
					Repeated: true,
					Schema: bq.Schema{
						{Name: "part_index", Type: bq.StringFieldType},
					},
				},
			},
			want: "incompatible BigQuery schema field content_parts.part_index: existing=STRING/NULLABLE, desired=INTEGER/NULLABLE",
		},
		{
			name: "nested RECORD subfield REQUIRED where writer allows NULL",
			schema: bq.Schema{
				{Name: "timestamp", Type: bq.TimestampFieldType, Required: true},
				{
					Name:     "content_parts",
					Type:     bq.RecordFieldType,
					Repeated: true,
					Schema: bq.Schema{
						{Name: "mime_type", Type: bq.StringFieldType, Required: true},
					},
				},
			},
			want: "incompatible BigQuery schema field content_parts.mime_type: existing=STRING/REQUIRED, desired=STRING/NULLABLE",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := DefaultConfig()

			table := &mockTableOps{
				metadataSeq: []metadataResponse{{md: &bq.TableMetadata{Schema: tt.schema, ETag: "etag-bad"}}},
			}
			err := ensureTableAndSchema(context.Background(), table, cfg, nil)
			if err == nil {
				t.Fatalf("ensureTableAndSchema() error = nil, want incompatible schema error containing %q", tt.want)
			}
			if table.updateCalls != 0 {
				t.Errorf("updateCalls = %d, want 0", table.updateCalls)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error %q does not contain %q", err.Error(), tt.want)
			}
		})
	}
}

func TestEnsureTableAndSchema_StructLiteralConfigDefaultsToAutoSchemaUpgrade(t *testing.T) {
	outdated := bq.Schema{
		{Name: "timestamp", Type: bq.TimestampFieldType, Required: true},
	}

	t.Run("struct literal Config without DefaultConfig upgrades outdated schema by default", func(t *testing.T) {
		cfg := applyDefaults(Config{
			ProjectID: "p",
			DatasetID: "d",
			TableName: "t",
			Enabled:   true,
		})
		if cfg.DisableSchemaUpgrade {
			t.Fatal("cfg.DisableSchemaUpgrade = true for struct-literal Config{}, want false")
		}
		table := &mockTableOps{
			metadataSeq: []metadataResponse{{md: &bq.TableMetadata{Schema: outdated, ETag: "etag-struct"}}},
		}
		if err := ensureTableAndSchema(context.Background(), table, cfg, nil); err != nil {
			t.Fatalf("ensureTableAndSchema() error = %v", err)
		}
		if table.updateCalls != 1 {
			t.Fatalf("updateCalls = %d, want 1", table.updateCalls)
		}
	})

	t.Run("struct literal Config with DisableSchemaUpgrade=true skips upgrade", func(t *testing.T) {
		cfg := applyDefaults(Config{
			ProjectID:            "p",
			DatasetID:            "d",
			TableName:            "t",
			Enabled:              true,
			DisableSchemaUpgrade: true,
		})
		if !cfg.DisableSchemaUpgrade {
			t.Fatal("cfg.DisableSchemaUpgrade = false when DisableSchemaUpgrade=true, want true")
		}
		table := &mockTableOps{
			metadataSeq: []metadataResponse{{md: &bq.TableMetadata{Schema: outdated, ETag: "etag-struct"}}},
		}
		if err := ensureTableAndSchema(context.Background(), table, cfg, nil); err != nil {
			t.Fatalf("ensureTableAndSchema() error = %v", err)
		}
		if table.updateCalls != 0 {
			t.Fatalf("updateCalls = %d, want 0", table.updateCalls)
		}
	})

	t.Run("missing REQUIRED field timestamp is appended as NULLABLE", func(t *testing.T) {
		cfg := DefaultConfig()
		table := &mockTableOps{
			metadataSeq: []metadataResponse{{
				md: &bq.TableMetadata{
					Schema: bq.Schema{{Name: "event_id", Type: bq.StringFieldType}},
					ETag:   "etag-missing-ts",
				},
			}},
		}
		if err := ensureTableAndSchema(context.Background(), table, cfg, nil); err != nil {
			t.Fatalf("ensureTableAndSchema() error = %v", err)
		}
		if table.updateCalls != 1 {
			t.Fatalf("updateCalls = %d, want 1", table.updateCalls)
		}
		var tsField *bq.FieldSchema
		for _, f := range table.updates[0].update.Schema {
			if f.Name == "timestamp" {
				tsField = f
				break
			}
		}
		if tsField == nil {
			t.Fatal("timestamp field missing from upgraded schema")
		}
		if tsField.Required {
			t.Error("appended timestamp field has Required = true, want false (BigQuery forbids adding REQUIRED columns)")
		}
	})
}

func TestEnsureTableAndSchema_IncompatibleSchemaErrorWrapsTableName(t *testing.T) {
	cfg := DefaultConfig()
	cfg.TableName = "my_custom_table"
	incompatibleSchema := bq.Schema{
		{Name: "event_id", Type: bq.IntegerFieldType},
	}
	table := &mockTableOps{
		metadataSeq: []metadataResponse{{
			md: &bq.TableMetadata{
				Schema: incompatibleSchema,
				ETag:   "etag-incompatible",
			},
		}},
	}
	err := ensureTableAndSchema(context.Background(), table, cfg, nil)
	if err == nil {
		t.Fatal("ensureTableAndSchema() error = nil, want error")
	}
	wantSubstr := "incompatible BigQuery table my_custom_table schema"
	if !strings.Contains(err.Error(), wantSubstr) {
		t.Errorf("error = %q, want substring %q", err.Error(), wantSubstr)
	}
}

func TestUpgradeTableSchema_ExistingNullableDesiredRequired(t *testing.T) {
	existing := bq.Schema{
		{Name: "timestamp", Type: bq.TimestampFieldType, Required: false},
	}
	desired := bq.Schema{
		{Name: "timestamp", Type: bq.TimestampFieldType, Required: true},
	}
	merged, changed, err := upgradeTableSchema(existing, desired)
	if err != nil {
		t.Fatalf("upgradeTableSchema() unexpected error: %v", err)
	}
	if changed {
		t.Errorf("upgradeTableSchema() changed = true, want false (existing column preserved as-is)")
	}
	if len(merged) != 1 || merged[0].Required {
		t.Errorf("merged schema = %v, want existing NULLABLE column preserved", merged)
	}
}

func TestUpgradeTableSchema_CaseInsensitiveNestedRecordLookup(t *testing.T) {
	existing := bq.Schema{
		{
			Name:     "Content_Parts",
			Type:     bq.RecordFieldType,
			Repeated: true,
			Schema: bq.Schema{
				{Name: "Part_Index", Type: bq.IntegerFieldType},
			},
		},
	}
	desired := bq.Schema{
		{
			Name:     "content_parts",
			Type:     bq.RecordFieldType,
			Repeated: true,
			Schema: bq.Schema{
				{Name: "part_index", Type: bq.IntegerFieldType},
				{Name: "new_subfield", Type: bq.StringFieldType},
			},
		},
	}
	merged, changed, err := upgradeTableSchema(existing, desired)
	if err != nil {
		t.Fatalf("upgradeTableSchema() unexpected error: %v", err)
	}
	if !changed {
		t.Errorf("upgradeTableSchema() changed = false, want true (new_subfield added)")
	}
	if len(merged) != 1 || len(merged[0].Schema) != 2 {
		t.Fatalf("merged subfields count = %d, want 2", len(merged[0].Schema))
	}
}
