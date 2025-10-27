// Copyright 2025 Google LLC
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

package database

import (
	"context"
	"database/sql/driver"
	"encoding/json"
	"errors"
	"fmt"

	"gorm.io/gorm"
	"gorm.io/gorm/clause"
	"gorm.io/gorm/schema"
)

// stateMap is a custom type for map[string]any that handles its own
// JSON serialization and deserialization for the database by implementing gorm.Serializer.
type stateMap map[string]any

// GormDataType defines the generic fallback data type, implements GormDataTypeInterface
func (stateMap) GormDataType() string {
	return "text"
}

// GormDBDataType defines database specific data types, implements GormDBDataTypeInterface
func (stateMap) GormDBDataType(db *gorm.DB, field *schema.Field) string {
	switch db.Dialector.Name() {
	case "postgres":
		return "JSONB"
	case "mysql":
		return "LONGTEXT"
	case "spanner":
		return "STRING(MAX)"
	default:
		return ""
	}
}

// Value implements the gorm.Serializer Value method.
func (sm stateMap) Value() (driver.Value, error) {
	if sm == nil {
		sm = make(map[string]any) // Serialize as '{}' instead of NULL
	}
	// For all other databases, return a JSON string.
	b, err := json.Marshal(sm)
	if err != nil {
		return nil, err
	}
	return string(b), nil
}

// Scan implements the gorm.Serializer Scan method.
func (sm *stateMap) Scan(value any) error {
	if value == nil {
		*sm = make(map[string]any)
		return nil
	}

	var bytes []byte

	switch v := value.(type) {
	case []byte: // Postgres, MySQL
		bytes = v
	case string: // Some drivers
		bytes = []byte(v)
	default:
		return fmt.Errorf("failed to unmarshal JSON value: %T", value)
	}

	if len(bytes) == 0 {
		*sm = make(map[string]any)
		return nil
	}

	return json.Unmarshal(bytes, sm)
}

func (sm stateMap) GormValue(ctx context.Context, db *gorm.DB) clause.Expr {
	data, _ := json.Marshal(sm)
	// TODO log the expression result
	return gorm.Expr("?", string(data))
}

// dynamicJSON defined JSON data type, that implements driver.Valuer, sql.Scanner interface
type dynamicJSON json.RawMessage

// Value return json value, implement driver.Valuer interface
func (j dynamicJSON) Value() (driver.Value, error) {
	if len(j) == 0 {
		return nil, nil
	}
	return string(j), nil
}

// Scan implements the gorm.Serializer Scan method.
func (j *dynamicJSON) Scan(value any) error {
	if value == nil {
		*j = nil
		return nil
	}
	var bytes []byte
	switch v := value.(type) {
	case []byte:
		if len(v) == 0 {
			*j = nil
			return nil
		}
		bytes = make([]byte, len(v))
		copy(bytes, v)
	case string:
		if v == "" {
			*j = nil
			return nil
		}
		bytes = []byte(v)
	default:
		return errors.New(fmt.Sprint("Failed to unmarshal JSONB value:", value))
	}

	if !json.Valid(bytes) {
		return fmt.Errorf("invalid JSON received from database: %s", string(bytes))
	}
	*j = dynamicJSON(bytes)
	return nil
}

func (j dynamicJSON) String() string {
	return string(j)
}

// GormDataType defines the generic fallback data type, implements GormDataTypeInterface
func (dynamicJSON) GormDataType() string {
	return "text"
}

// GormDBDataType defines database specific data types, implements GormDBDataTypeInterface
func (dynamicJSON) GormDBDataType(db *gorm.DB, field *schema.Field) string {
	switch db.Dialector.Name() {
	case "mysql":
		return "LONGTEXT"
	case "postgres":
		return "JSONB"
	case "spanner":
		return "STRING(MAX)"
	}
	return ""
}

func (js dynamicJSON) GormValue(ctx context.Context, db *gorm.DB) clause.Expr {
	if len(js) == 0 {
		return gorm.Expr("NULL")
	}
	// TODO log the expression result
	return gorm.Expr("?", string(js))
}
