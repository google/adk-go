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

package spannertoolset

import (
	"cmp"

	"google.golang.org/adk/v2/agent"
)

// The metadata tools. Each runs adk-python's INFORMATION_SCHEMA queries and
// returns what adk-python's tool of the same name returns.

type databaseArgs struct {
	ProjectID  string `json:"project_id"`
	InstanceID string `json:"instance_id"`
	DatabaseID string `json:"database_id"`
}

func (a databaseArgs) path() (string, error) {
	return databasePath(a.ProjectID, a.InstanceID, a.DatabaseID)
}

type listTableNamesArgs struct {
	databaseArgs
	NamedSchema string `json:"named_schema,omitempty"`
}

func (s *spanner) listTableNames(ctx agent.Context, args listTableNamesArgs) (any, error) {
	db, err := args.path()
	if err != nil {
		return nil, err
	}
	dialect, err := s.dialect(ctx, db)
	if err != nil {
		return nil, err
	}
	// adk-python's Database.list_tables, with the dialect's default schema
	// when none is named.
	var stmt statement
	if dialect == dialectPostgreSQL {
		stmt.SQL = "SELECT TABLE_NAME\nFROM INFORMATION_SCHEMA.TABLES\nWHERE TABLE_SCHEMA = $1\n"
		stmt.Params, stmt.ParamTypes = stringParams(map[string]string{"p1": cmp.Or(args.NamedSchema, "public")})
	} else {
		stmt.SQL = "SELECT TABLE_NAME\nFROM INFORMATION_SCHEMA.TABLES\nWHERE TABLE_SCHEMA = @schema AND SPANNER_STATE = 'COMMITTED'\n"
		stmt.Params, stmt.ParamTypes = stringParams(map[string]string{"schema": args.NamedSchema})
	}
	tables := []any{}
	err = s.withSession(ctx, db, "", func(ss *dbSession) error {
		rs, err := ss.query(ctx, stmt, singleUseReadOnly, -1)
		if err != nil {
			return err
		}
		for _, row := range rs.rows {
			tables = append(tables, row[0])
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return success("results", tables), nil
}

const (
	columnsQuery = `
      SELECT
          COLUMN_NAME,
          TABLE_SCHEMA,
          SPANNER_TYPE,
          ORDINAL_POSITION,
          COLUMN_DEFAULT,
          IS_NULLABLE,
          IS_GENERATED,
          GENERATION_EXPRESSION,
          IS_STORED
      FROM
          INFORMATION_SCHEMA.COLUMNS
      WHERE
          TABLE_NAME = @table_name
          AND TABLE_SCHEMA = @named_schema
      ORDER BY
          ORDINAL_POSITION
  `

	keyColumnUsageQuery = `
      SELECT
          COLUMN_NAME,
          CONSTRAINT_NAME,
          ORDINAL_POSITION,
          POSITION_IN_UNIQUE_CONSTRAINT
      FROM
          INFORMATION_SCHEMA.KEY_COLUMN_USAGE
      WHERE
          TABLE_NAME = @table_name
          AND TABLE_SCHEMA = @named_schema
  `

	tableMetadataQuery = `
      SELECT
          TABLE_SCHEMA,
          TABLE_NAME,
          TABLE_TYPE,
          PARENT_TABLE_NAME,
          ON_DELETE_ACTION,
          SPANNER_STATE,
          INTERLEAVE_TYPE,
          ROW_DELETION_POLICY_EXPRESSION
      FROM
          INFORMATION_SCHEMA.TABLES
      WHERE
          TABLE_NAME = @table_name
          AND TABLE_SCHEMA = @named_schema;
  `
)

var (
	columnKeys         = []string{"TABLE_SCHEMA", "SPANNER_TYPE", "ORDINAL_POSITION", "COLUMN_DEFAULT", "IS_NULLABLE", "IS_GENERATED", "GENERATION_EXPRESSION", "IS_STORED"}
	keyColumnUsageKeys = []string{"CONSTRAINT_NAME", "ORDINAL_POSITION", "POSITION_IN_UNIQUE_CONSTRAINT"}
	tableMetadataKeys  = []string{"TABLE_SCHEMA", "TABLE_NAME", "TABLE_TYPE", "PARENT_TABLE_NAME", "ON_DELETE_ACTION", "SPANNER_STATE", "INTERLEAVE_TYPE", "ROW_DELETION_POLICY_EXPRESSION"}
)

type getTableSchemaArgs struct {
	databaseArgs
	TableName   string `json:"table_name"`
	NamedSchema string `json:"named_schema,omitempty"`
}

func (s *spanner) getTableSchema(ctx agent.Context, args getTableSchemaArgs) (any, error) {
	db, err := args.path()
	if err != nil {
		return nil, err
	}
	dialect, err := s.dialect(ctx, db)
	if err != nil {
		return nil, err
	}
	if dialect == dialectPostgreSQL {
		// adk-python words this one error without the final period.
		return errorDetails("PostgreSQL dialect is not supported"), nil
	}
	params, types := stringParams(map[string]string{"table_name": args.TableName, "named_schema": args.NamedSchema})
	query := func(sql string) statement { return statement{SQL: sql, Params: params, ParamTypes: types} }

	schema := map[string]any{}
	metadata := []any{}
	// The three queries read in one transaction, so they see one version of
	// the schema, as adk-python's multi-use snapshot does.
	err = s.withSession(ctx, db, "", func(ss *dbSession) error {
		columns, err := ss.query(ctx, query(columnsQuery), beginReadOnly, -1)
		if err != nil {
			return err
		}
		for _, row := range columns.rows {
			name, _ := row[0].(string)
			schema[name] = rowObject(columnKeys, row[1:])
		}
		txn := useTransaction(columns.transactionID)

		keys, err := ss.query(ctx, query(keyColumnUsageQuery), txn, -1)
		if err != nil {
			return err
		}
		for _, row := range keys.rows {
			name, _ := row[0].(string)
			column, ok := schema[name].(map[string]any)
			if !ok {
				continue
			}
			usage, _ := column["KEY_COLUMN_USAGE"].([]any)
			column["KEY_COLUMN_USAGE"] = append(usage, rowObject(keyColumnUsageKeys, row[1:]))
		}

		tables, err := ss.query(ctx, query(tableMetadataQuery), txn, -1)
		if err != nil {
			return err
		}
		for _, row := range tables.rows {
			metadata = append(metadata, rowObject(tableMetadataKeys, row))
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return success("results", map[string]any{"schema": schema, "metadata": metadata}), nil
}

type listTableIndexesArgs struct {
	databaseArgs
	TableID string `json:"table_id"`
}

var (
	indexKeys       = []string{"INDEX_NAME", "TABLE_SCHEMA", "INDEX_TYPE", "PARENT_TABLE_NAME", "IS_UNIQUE", "IS_NULL_FILTERED", "INDEX_STATE"}
	indexColumnKeys = []string{"INDEX_NAME", "TABLE_SCHEMA", "COLUMN_NAME", "ORDINAL_POSITION", "IS_NULLABLE", "SPANNER_TYPE"}
)

func (s *spanner) listTableIndexes(ctx agent.Context, args listTableIndexesArgs) (any, error) {
	sql := "SELECT INDEX_NAME, TABLE_SCHEMA, INDEX_TYPE," +
		" PARENT_TABLE_NAME, IS_UNIQUE, IS_NULL_FILTERED, INDEX_STATE " +
		"FROM INFORMATION_SCHEMA.INDEXES " +
		"WHERE TABLE_NAME = @table_id "
	return s.tableObjects(ctx, args, sql, indexKeys)
}

func (s *spanner) listTableIndexColumns(ctx agent.Context, args listTableIndexesArgs) (any, error) {
	sql := "SELECT INDEX_NAME, TABLE_SCHEMA, COLUMN_NAME," +
		" ORDINAL_POSITION, IS_NULLABLE, SPANNER_TYPE " +
		"FROM INFORMATION_SCHEMA.INDEX_COLUMNS " +
		"WHERE TABLE_NAME = @table_id "
	return s.tableObjects(ctx, args, sql, indexColumnKeys)
}

// tableObjects runs sql for the table args names, in a GoogleSQL database,
// and returns each row as an object with the given keys.
func (s *spanner) tableObjects(ctx agent.Context, args listTableIndexesArgs, sql string, keys []string) (any, error) {
	params, types := stringParams(map[string]string{"table_id": args.TableID})
	var objects []any
	result, err := s.googleSQLRows(ctx, args.databaseArgs, statement{SQL: sql, Params: params, ParamTypes: types}, func(row []any) {
		objects = append(objects, rowObject(keys, row))
	})
	if result != nil || err != nil {
		return result, err
	}
	return success("results", orEmpty(objects)), nil
}

func (s *spanner) listNamedSchemas(ctx agent.Context, args databaseArgs) (any, error) {
	sql := `
    SELECT
        SCHEMA_NAME
    FROM
        INFORMATION_SCHEMA.SCHEMATA
    WHERE
        SCHEMA_NAME NOT IN ('', 'INFORMATION_SCHEMA', 'SPANNER_SYS');
    `
	var schemas []any
	result, err := s.googleSQLRows(ctx, args, statement{SQL: sql}, func(row []any) {
		schemas = append(schemas, row[0])
	})
	if result != nil || err != nil {
		return result, err
	}
	return success("results", orEmpty(schemas)), nil
}

// googleSQLRows runs stmt in the database args names and calls each with
// every row. For a PostgreSQL database it runs nothing and returns
// adk-python's error result instead.
func (s *spanner) googleSQLRows(ctx agent.Context, args databaseArgs, stmt statement, each func([]any)) (map[string]any, error) {
	db, err := args.path()
	if err != nil {
		return nil, err
	}
	dialect, err := s.dialect(ctx, db)
	if err != nil {
		return nil, err
	}
	if dialect == dialectPostgreSQL {
		return errorDetails(postgreSQLNotSupported), nil
	}
	return nil, s.withSession(ctx, db, "", func(ss *dbSession) error {
		rs, err := ss.query(ctx, stmt, singleUseReadOnly, -1)
		if err != nil {
			return err
		}
		for _, row := range rs.rows {
			each(row)
		}
		return nil
	})
}

// rowObject returns an object from each key to the value at its position in
// row, or null past the end of row.
func rowObject(keys []string, row []any) map[string]any {
	out := make(map[string]any, len(keys))
	for i, k := range keys {
		var v any
		if i < len(row) {
			v = row[i]
		}
		out[k] = v
	}
	return out
}

func orEmpty(values []any) []any {
	if values == nil {
		return []any{}
	}
	return values
}
