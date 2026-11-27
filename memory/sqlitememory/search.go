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

package sqlitememory

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/memory"
)

const ftsSearch = `
SELECT e.id, e.session_id, e.event_id, e.author, e.timestamp, e.content_json
FROM memory_events_fts
JOIN memory_events AS e ON e.id = memory_events_fts.rowid
WHERE e.app_name = ? AND e.user_id = ? AND memory_events_fts.search_text MATCH ?
ORDER BY bm25(memory_events_fts), e.timestamp DESC
LIMIT ?`

func (s *store) SearchMemory(ctx context.Context, req *memory.SearchRequest) (*memory.SearchResponse, error) {
	if req == nil {
		return nil, errors.New("sqlitememory: search request must not be nil")
	}
	response := &memory.SearchResponse{Memories: []memory.Entry{}}
	tokens := extractTokens(req.Query)
	if len(tokens) == 0 {
		return response, nil
	}

	var query string
	args := []any{req.AppName, req.UserID}
	// unicode61 indexes runs of unspaced script as a whole, so look inside
	// those runs with LIKE.
	if s.ftsEnabled && !slices.ContainsFunc(tokens, hasUnspacedScript) {
		for i, token := range tokens {
			tokens[i] = `"` + strings.ReplaceAll(token, `"`, `""`) + `"`
		}
		query = ftsSearch
		args = append(args, strings.Join(tokens, " OR "))
	} else {
		query, args = likeSearch(tokens, args)
	}
	rows, err := s.db.QueryContext(ctx, query, append(args, s.maxResults)...)
	if err != nil {
		return nil, fmt.Errorf("sqlitememory: query matching events: %w", err)
	}
	defer func() { _ = rows.Close() }()

	for rows.Next() {
		var rowID int64
		var sessionID, eventID, author string
		var timestamp float64
		var contentJSON []byte
		if err := rows.Scan(&rowID, &sessionID, &eventID, &author, &timestamp, &contentJSON); err != nil {
			return nil, errors.New("sqlitememory: read matching event row")
		}
		var content genai.Content
		if err := json.Unmarshal(contentJSON, &content); err != nil {
			return nil, errors.New("sqlitememory: decode stored event content")
		}
		content.Parts = slices.DeleteFunc(content.Parts, func(part *genai.Part) bool {
			return part == nil || part.Thought
		})
		seconds, fraction := math.Modf(timestamp)
		response.Memories = append(response.Memories, memory.Entry{
			ID:             strconv.FormatInt(rowID, 10),
			Content:        &content,
			Author:         author,
			Timestamp:      time.Unix(int64(seconds), int64(math.Round(fraction*1e9))).UTC(),
			CustomMetadata: map[string]any{"session_id": sessionID, "event_id": eventID},
		})
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("sqlitememory: read matching events: %w", err)
	}
	return response, nil
}

var likeEscaper = strings.NewReplacer(`\`, `\\`, `%`, `\%`, `_`, `\_`)

func likeSearch(tokens []string, args []any) (string, []any) {
	const match = `search_text LIKE ? ESCAPE '\'`
	const score = `(CASE WHEN ` + match + ` THEN 1 ELSE 0 END)`
	for _, token := range tokens {
		args = append(args, "%"+likeEscaper.Replace(token)+"%")
	}
	args = append(args, args[len(args)-len(tokens):]...)
	return `
SELECT id, session_id, event_id, author, timestamp, content_json
FROM memory_events
WHERE app_name = ? AND user_id = ? AND (` + match + strings.Repeat(" OR "+match, len(tokens)-1) + `)
ORDER BY (` + score + strings.Repeat(" + "+score, len(tokens)-1) + `) DESC, timestamp DESC
LIMIT ?`, args
}
