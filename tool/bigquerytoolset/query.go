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

package bigquerytoolset

import (
	"errors"
	"fmt"
	"maps"
	"net/url"
	"strconv"

	"google.golang.org/adk/v2/agent"
)

const executeSQLName = "execute_sql"

type executeSQLArgs struct {
	ProjectID string `json:"project_id"`
	Query     string `json:"query"`
	DryRun    bool   `json:"dry_run,omitempty"`
}

// executeSQL runs a read-only query, as adk-python's execute_sql does with
// its default write mode, WriteMode.BLOCKED: a dry run first finds the
// statement type, and anything but a SELECT is refused.
func (b *bigQuery) executeSQL(ctx agent.Context, args executeSQLArgs) (any, error) {
	projectID := b.project(args.ProjectID)
	project, err := pathSegment("project_id", projectID)
	if err != nil {
		return nil, err
	}
	if b.computeProjectID != "" && projectID != b.computeProjectID {
		return errorDetails(fmt.Sprintf("Cannot execute query in the project %s, as the tool is restricted to execute queries only in the project %s.", projectID, b.computeProjectID)), nil
	}
	labels := b.jobLabels(executeSQLName)

	job, err := b.dryRun(ctx, project, projectID, args.Query, labels)
	if err != nil {
		return nil, err
	}
	if statementType(job) != "SELECT" {
		return errorDetails("Read-only mode only supports SELECT statements."), nil
	}
	if args.DryRun {
		return map[string]any{"status": "SUCCESS", "dry_run_info": job}, nil
	}

	rows, truncated, err := b.query(ctx, project, args.Query, labels)
	if err != nil {
		return nil, err
	}
	result := map[string]any{"status": "SUCCESS", "rows": rows}
	if truncated {
		result["result_is_likely_truncated"] = true
	}
	return result, nil
}

// jobLabels returns Config.JobLabels with the labels adk-python adds to
// attribute a job to the tool and the application.
func (b *bigQuery) jobLabels(toolName string) map[string]string {
	labels := maps.Clone(b.labels)
	if labels == nil {
		labels = map[string]string{}
	}
	labels[reservedLabelPrefix+"tool"] = toolName
	if b.applicationName != "" {
		labels[reservedLabelPrefix+"application-name"] = b.applicationName
	}
	return labels
}

// dryRun validates query without running it and returns the dry run job as
// the REST API sent it.
func (b *bigQuery) dryRun(ctx agent.Context, project, projectID, query string, labels map[string]string) (map[string]any, error) {
	type jobReference struct {
		ProjectID string `json:"projectId"`
		Location  string `json:"location,omitempty"`
	}
	body := map[string]any{
		"jobReference": jobReference{ProjectID: projectID, Location: b.location},
		"configuration": map[string]any{
			"dryRun": true,
			"labels": labels,
			"query":  map[string]any{"query": query, "useLegacySql": false},
		},
	}
	var job map[string]any
	if err := b.client.post(ctx, executeSQLName, "projects/"+project+"/jobs", body, &job); err != nil {
		return nil, err
	}
	return job, nil
}

// statementType returns statistics.query.statementType of a job resource.
func statementType(job map[string]any) string {
	stats, _ := job["statistics"].(map[string]any)
	query, _ := stats["query"].(map[string]any)
	t, _ := query["statementType"].(string)
	return t
}

// queryResults is the part of a jobs.query or jobs.getQueryResults response
// the tool reads.
type queryResults struct {
	JobComplete  bool `json:"jobComplete"`
	JobReference struct {
		JobID    string `json:"jobId"`
		Location string `json:"location"`
	} `json:"jobReference"`
	Schema struct {
		Fields []field `json:"fields"`
	} `json:"schema"`
	Rows      []map[string]any `json:"rows"`
	TotalRows string           `json:"totalRows"`
	PageToken string           `json:"pageToken"`
}

// query runs query, waits for it to finish and returns up to b.maxRows rows,
// and whether rows were left out. adk-python instead flags every result that
// reaches the limit as likely truncated. Like adk-python's query_and_wait, it
// asks for timestamps as integer microseconds, which keeps their precision.
func (b *bigQuery) query(ctx agent.Context, project, query string, labels map[string]string) ([]map[string]any, bool, error) {
	body := map[string]any{
		"query":         query,
		"useLegacySql":  false,
		"formatOptions": map[string]any{"useInt64Timestamp": true},
		"maxResults":    b.maxRows,
		"labels":        labels,
	}
	if b.maxBytesBilled != 0 {
		body["maximumBytesBilled"] = strconv.FormatInt(b.maxBytesBilled, 10)
	}
	if b.location != "" {
		body["location"] = b.location
	}
	var res queryResults
	if err := b.client.post(ctx, executeSQLName, "projects/"+project+"/queries", body, &res); err != nil {
		return nil, false, err
	}

	rows := []map[string]any{}
	seen := map[string]bool{}
	for {
		if res.JobComplete {
			for _, r := range res.Rows {
				if len(rows) == b.maxRows {
					return rows, true, nil
				}
				row, err := rowValues(res.Schema.Fields, r)
				if err != nil {
					return nil, false, err
				}
				rows = append(rows, row)
			}
			// BigQuery returns a page token only when more rows follow.
			if res.PageToken == "" {
				return rows, false, nil
			}
			if len(rows) == b.maxRows {
				return rows, true, nil
			}
			if seen[res.PageToken] {
				return nil, false, errors.New("BigQuery returned a page token it had already returned")
			}
			seen[res.PageToken] = true
		}
		next, err := b.queryResults(ctx, project, res, b.maxRows-len(rows))
		if err != nil {
			return nil, false, err
		}
		res = next
	}
}

// queryResults fetches the next page of results of the query prev belongs
// to, or, while the query runs, waits for its first page.
func (b *bigQuery) queryResults(ctx agent.Context, project string, prev queryResults, maxRows int) (queryResults, error) {
	job, err := pathSegment("job id", prev.JobReference.JobID)
	if err != nil {
		return queryResults{}, fmt.Errorf("BigQuery did not return the query's job: %w", err)
	}
	params := url.Values{
		"maxResults":                      {strconv.Itoa(maxRows)},
		"formatOptions.useInt64Timestamp": {"true"},
	}
	if prev.JobReference.Location != "" {
		params.Set("location", prev.JobReference.Location)
	}
	if prev.JobComplete {
		params.Set("pageToken", prev.PageToken)
	}
	var res queryResults
	if err := b.client.get(ctx, executeSQLName, "projects/"+project+"/queries/"+job, params, &res); err != nil {
		return queryResults{}, err
	}
	// A page of a finished query need not repeat the job reference.
	if res.JobReference.JobID == "" {
		res.JobReference = prev.JobReference
	}
	return res, nil
}
