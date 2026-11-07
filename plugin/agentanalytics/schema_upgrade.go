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
// unless DisableSchemaUpgrade is set, performs an additive schema upgrade
// without issuing metadata writes when all fields already match.
func ensureTableAndSchema(ctx context.Context, table tableMetadataOps, config Config, denied map[string]struct{}) error {
	desired := projectEventsSchema(denied)
	md, err := table.Metadata(ctx)
	if err != nil {
		if !isNotFoundError(err) {
			return fmt.Errorf("failed to get BigQuery table %s metadata: %w", config.TableName, err)
		}
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
				schemaVersionKey: SchemaVersion,
			},
		})
		if createErr == nil {
			return nil
		}
		if isConflictError(createErr) {
			if config.DisableSchemaUpgrade {
				return nil
			}
			md, err = table.Metadata(ctx)
			if err != nil {
				return fmt.Errorf("failed to get BigQuery table %s metadata: %w", config.TableName, err)
			}
			return maybeUpgradeSchema(ctx, table, md, config, desired)
		}
		return fmt.Errorf("failed to create BigQuery table %s: %w", config.TableName, createErr)
	}

	if config.DisableSchemaUpgrade {
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
				return fmt.Errorf("failed to get BigQuery table %s metadata: %w", config.TableName, err)
			}
		}

		updatedSchema, schemaChanged, err := upgradeTableSchema(md.Schema, desired)
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
		update.SetLabel(schemaVersionKey, SchemaVersion)

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

func isConflictError(err error) bool {
	return hasErrorCode(err, http.StatusConflict, codes.AlreadyExists)
}

func isPreconditionFailedError(err error) bool {
	return hasErrorCode(err, http.StatusPreconditionFailed, codes.FailedPrecondition)
}
