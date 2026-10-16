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
	"fmt"
	"net/http"
	"strings"

	bq "cloud.google.com/go/bigquery"
	"google.golang.org/api/googleapi"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const maxPreconditionRetries = 2

// tableMetadataOps abstracts the BigQuery Table metadata RPCs so that all
// startup and schema upgrade paths can be unit-tested deterministically.
type tableMetadataOps interface {
	Metadata(ctx context.Context) (*bq.TableMetadata, error)
	Create(ctx context.Context, meta *bq.TableMetadata) error
	Update(ctx context.Context, update bq.TableMetadataToUpdate, etag string) (*bq.TableMetadata, error)
}

type bqTableAdapter struct {
	table *bq.Table
}

func (a bqTableAdapter) Metadata(ctx context.Context) (*bq.TableMetadata, error) {
	return a.table.Metadata(ctx)
}

func (a bqTableAdapter) Create(ctx context.Context, meta *bq.TableMetadata) error {
	return a.table.Create(ctx, meta)
}

func (a bqTableAdapter) Update(ctx context.Context, update bq.TableMetadataToUpdate, etag string) (*bq.TableMetadata, error) {
	return a.table.Update(ctx, update, etag)
}

// ensureTableAndSchema ensures the BigQuery destination table exists and,
// unless DisableAutoSchemaUpgrade is set, performs a permission-safe additive
// schema upgrade without issuing metadata writes when all fields already match.
func ensureTableAndSchema(ctx context.Context, table tableMetadataOps, config Config, denied map[string]struct{}) error {
	desired := projectEventsSchema(denied)
	md, err := table.Metadata(ctx)
	if err != nil {
		if isNotFoundError(err) {
			config.logf("Table %s not found. Creating it...", config.TableName)
			createErr := table.Create(ctx, &bq.TableMetadata{
				Schema: desired,
				// Partition by DAY on timestamp for parity with adk-python and adk-java.
				TimePartitioning: &bq.TimePartitioning{
					Field: timestampField,
					Type:  bq.DayPartitioningType,
				},
				Clustering: &bq.Clustering{
					Fields: config.ClusteringFields,
				},
				Labels: map[string]string{
					SchemaVersionKey: SchemaVersion,
				},
			})
			if createErr == nil {
				return nil
			}
			if isConflictError(createErr) {
				if !config.autoSchemaUpgrade() {
					return nil
				}
				md, err = table.Metadata(ctx)
				if err != nil {
					if !isPermissionDeniedError(err) {
						return fmt.Errorf("failed to inspect concurrently created BigQuery table metadata: %w", err)
					}
					config.logf("agentanalytics: table %s created concurrently (409), failed to re-fetch metadata: %v", config.TableName, err)
					return nil
				}
				return maybeUpgradeSchema(ctx, table, md, config, desired)
			}
			return fmt.Errorf("failed to create BigQuery table %s: %w", config.TableName, createErr)
		}
		if isPermissionDeniedError(err) {
			config.logf("agentanalytics: insufficient permissions to inspect table %s metadata (proceeding assuming table exists): %v", config.TableName, err)
			return nil
		}
		return fmt.Errorf("failed to inspect BigQuery table %s metadata: %w", config.TableName, err)
	}

	if !config.autoSchemaUpgrade() {
		return nil
	}
	return maybeUpgradeSchema(ctx, table, md, config, desired)
}

func maybeUpgradeSchema(ctx context.Context, table tableMetadataOps, md *bq.TableMetadata, config Config, desired bq.Schema) error {
	for attempt := 0; ; attempt++ {
		if attempt > 0 {
			var err error
			md, err = table.Metadata(ctx)
			if err != nil {
				return fmt.Errorf("failed to re-fetch BigQuery table %s metadata after 412: %w", config.TableName, err)
			}
		}

		updatedSchema, schemaChanged, err := diffAndMergeSchema(md.Schema, desired)
		if err != nil {
			return fmt.Errorf("incompatible BigQuery table %s schema: %w", config.TableName, err)
		}
		if !schemaChanged {
			// Intentionally skip a label-only Update RPC when all columns already
			// match so startup remains read-only on up-to-date tables.
			return nil
		}

		var update bq.TableMetadataToUpdate
		update.Schema = updatedSchema
		update.SetLabel(SchemaVersionKey, SchemaVersion)

		_, updateErr := table.Update(ctx, update, md.ETag)
		if updateErr == nil {
			return nil
		}
		if isPreconditionFailedError(updateErr) && attempt < maxPreconditionRetries {
			continue
		}
		return fmt.Errorf("failed to upgrade BigQuery table %s schema: %w", config.TableName, updateErr)
	}
}

// diffAndMergeSchema compares existing against desired recursively, preserving
// existing field ordering and appending missing fields as NULLABLE.
func diffAndMergeSchema(existing, desired bq.Schema) (bq.Schema, bool, error) {
	existingIndex := make(map[string]int, len(existing))
	merged := make(bq.Schema, len(existing))
	for i, ef := range existing {
		existingIndex[strings.ToLower(ef.Name)] = i
		merged[i] = ef
	}

	schemaChanged := false
	for _, df := range desired {
		idx, ok := existingIndex[strings.ToLower(df.Name)]
		if !ok {
			merged = append(merged, cloneNullableFieldSchema(df))
			schemaChanged = true
			continue
		}

		ef := merged[idx]
		if ef.Type != df.Type {
			return nil, false, fmt.Errorf("field %q has type %s, want %s", df.Name, ef.Type, df.Type)
		}
		if ef.Repeated != df.Repeated {
			return nil, false, fmt.Errorf("field %q has repeated=%v, want %v", df.Name, ef.Repeated, df.Repeated)
		}
		if ef.Required && !df.Required {
			return nil, false, fmt.Errorf("field %q is REQUIRED but the writer allows NULL", df.Name)
		}
		if df.Type == bq.RecordFieldType {
			subMerged, subChanged, err := diffAndMergeSchema(ef.Schema, df.Schema)
			if err != nil {
				return nil, false, fmt.Errorf("field %q: %w", df.Name, err)
			}
			if subChanged {
				updated := *ef
				updated.Schema = subMerged
				merged[idx] = &updated
				schemaChanged = true
			}
		}
	}

	return merged, schemaChanged, nil
}

func cloneNullableFieldSchema(f *bq.FieldSchema) *bq.FieldSchema {
	cloned := *f
	cloned.Required = false
	if len(f.Schema) > 0 {
		cloned.Schema = make(bq.Schema, len(f.Schema))
		for i, sf := range f.Schema {
			cloned.Schema[i] = cloneNullableFieldSchema(sf)
		}
	}
	return &cloned
}

func hasErrorCode(err error, httpCode int, grpcCode codes.Code) bool {
	if err == nil {
		return false
	}
	var apiErr *googleapi.Error
	if errors.As(err, &apiErr) {
		return apiErr.Code == httpCode
	}
	if st, ok := status.FromError(err); ok {
		return st.Code() == grpcCode
	}
	return false
}

func isNotFoundError(err error) bool {
	return hasErrorCode(err, http.StatusNotFound, codes.NotFound)
}

func isPermissionDeniedError(err error) bool {
	return hasErrorCode(err, http.StatusForbidden, codes.PermissionDenied)
}

func isConflictError(err error) bool {
	return hasErrorCode(err, http.StatusConflict, codes.AlreadyExists)
}

func isPreconditionFailedError(err error) bool {
	return hasErrorCode(err, http.StatusPreconditionFailed, codes.FailedPrecondition)
}
