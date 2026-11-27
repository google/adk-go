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
	"google.golang.org/adk/v2/agent"
)

type executeSQLArgs struct {
	databaseArgs
	Query string `json:"query"`
}

// executeSQL runs a query in a strong read-only transaction, which Spanner
// fails for any statement that writes, and returns at most s.maxRows rows.
func (s *spanner) executeSQL(ctx agent.Context, args executeSQLArgs) (any, error) {
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
	var rs *resultSet
	err = s.withSession(ctx, db, s.role, func(ss *dbSession) error {
		rs, err = ss.query(ctx, statement{SQL: args.Query}, singleUseReadOnly, s.maxRows)
		return err
	})
	if err != nil {
		return nil, err
	}

	rows := make([]any, len(rs.rows))
	for i, row := range rs.rows {
		if s.resultMode != QueryResultModeDictList {
			rows[i] = row
			continue
		}
		object := make(map[string]any, len(row))
		for j, v := range row {
			object[rs.fields[j].Name] = v
		}
		rows[i] = object
	}
	result := success("rows", rows)
	if rs.truncated {
		result["result_is_likely_truncated"] = true
	}
	return result, nil
}
