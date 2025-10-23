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

// StateMap is a custom type for map[string]any that handles its own
// JSON serialization and deserialization for the database.
// StateMap is a custom type for map[string]any that handles
// dialect-specific JSON serialization by implementing gorm.Serializer.
type StateMap map[string]any

// GormDataType / GormDBDataType (For Schema/Migrations)

func (StateMap) GormDataType() string {
	return "text"
}

func (StateMap) GormDBDataType(db *gorm.DB, field *schema.Field) string {
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
func (sm StateMap) Value() (driver.Value, error) {
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
func (sm *StateMap) Scan(value any) error {
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

func (sm StateMap) GormValue(ctx context.Context, db *gorm.DB) clause.Expr {
	data, _ := json.Marshal(sm)
	//switch db.Dialector.Name() {
	return gorm.Expr("?", string(data))
}

// DynamicJSON defined JSON data type, that implements driver.Valuer, sql.Scanner interface
type DynamicJSON json.RawMessage

// Value return json value, implement driver.Valuer interface
func (j DynamicJSON) Value() (driver.Value, error) {
	if len(j) == 0 {
		return nil, nil
	}
	return string(j), nil
}

// Scan implements the gorm.Serializer Scan method.
// scan value into Jsonb, implements sql.Scanner interface
func (j *DynamicJSON) Scan(value any) error {
	if value == nil {
		*j = DynamicJSON("null")
		return nil
	}
	var bytes []byte
	if s, ok := value.(fmt.Stringer); ok {
		bytes = []byte(s.String())
	} else {
		switch v := value.(type) {
		case []byte:
			if len(v) > 0 {
				bytes = make([]byte, len(v))
				copy(bytes, v)
			}
		case string:
			bytes = []byte(v)
		default:
			return errors.New(fmt.Sprint("Failed to unmarshal JSONB value:", value))
		}
	}

	result := json.RawMessage(bytes)
	*j = DynamicJSON(result)
	return nil
}

// MarshalJSON to output non base64 encoded []byte
func (j DynamicJSON) MarshalJSON() ([]byte, error) {
	return json.RawMessage(j).MarshalJSON()
}

// UnmarshalJSON to deserialize []byte
func (j *DynamicJSON) UnmarshalJSON(b []byte) error {
	result := json.RawMessage{}
	err := result.UnmarshalJSON(b)
	*j = DynamicJSON(result)
	return err
}

func (j DynamicJSON) String() string {
	return string(j)
}

// GormDataType gorm common data type
func (DynamicJSON) GormDataType() string {
	return "text"
}

// GormDBDataType gorm db data type
func (DynamicJSON) GormDBDataType(db *gorm.DB, field *schema.Field) string {
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

func (js DynamicJSON) GormValue(ctx context.Context, db *gorm.DB) clause.Expr {
	if len(js) == 0 {
		return gorm.Expr("NULL")
	}
	data, _ := js.MarshalJSON()
	return gorm.Expr("?", string(data))
}
