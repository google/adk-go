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
	"net/url"

	"google.golang.org/adk/v2/agent"
)

// The metadata tools. Each returns what adk-python's tool of the same name
// returns: a list of ids, or the resource exactly as the BigQuery REST API
// sent it, including fields this package does not know about.

type listDatasetIDsArgs struct {
	ProjectID string `json:"project_id"`
}

func (b *bigQuery) listDatasetIDs(ctx agent.Context, args listDatasetIDsArgs) (any, error) {
	project, err := pathSegment("project_id", b.project(args.ProjectID))
	if err != nil {
		return nil, err
	}
	type page struct {
		Datasets []struct {
			DatasetReference struct {
				DatasetID string `json:"datasetId"`
			} `json:"datasetReference"`
		} `json:"datasets"`
		NextPageToken string `json:"nextPageToken"`
	}
	ids := []string{}
	err = pages(b, ctx, "list_dataset_ids", "projects/"+project+"/datasets", func(p *page) string {
		for _, d := range p.Datasets {
			ids = append(ids, d.DatasetReference.DatasetID)
		}
		return p.NextPageToken
	})
	if err != nil {
		return nil, err
	}
	return ids, nil
}

type getDatasetInfoArgs struct {
	ProjectID string `json:"project_id"`
	DatasetID string `json:"dataset_id"`
}

func (b *bigQuery) getDatasetInfo(ctx agent.Context, args getDatasetInfoArgs) (any, error) {
	path, err := datasetPath(b.project(args.ProjectID), b.dataset(args.DatasetID))
	if err != nil {
		return nil, err
	}
	return b.resource(ctx, "get_dataset_info", path, nil)
}

type listTableIDsArgs struct {
	ProjectID string `json:"project_id"`
	DatasetID string `json:"dataset_id"`
}

func (b *bigQuery) listTableIDs(ctx agent.Context, args listTableIDsArgs) (any, error) {
	path, err := datasetPath(b.project(args.ProjectID), b.dataset(args.DatasetID))
	if err != nil {
		return nil, err
	}
	type page struct {
		Tables []struct {
			TableReference struct {
				TableID string `json:"tableId"`
			} `json:"tableReference"`
		} `json:"tables"`
		NextPageToken string `json:"nextPageToken"`
	}
	ids := []string{}
	err = pages(b, ctx, "list_table_ids", path+"/tables", func(p *page) string {
		for _, t := range p.Tables {
			ids = append(ids, t.TableReference.TableID)
		}
		return p.NextPageToken
	})
	if err != nil {
		return nil, err
	}
	return ids, nil
}

type getTableInfoArgs struct {
	ProjectID string `json:"project_id"`
	DatasetID string `json:"dataset_id"`
	TableID   string `json:"table_id"`
}

func (b *bigQuery) getTableInfo(ctx agent.Context, args getTableInfoArgs) (any, error) {
	path, err := datasetPath(b.project(args.ProjectID), b.dataset(args.DatasetID))
	if err != nil {
		return nil, err
	}
	table, err := pathSegment("table_id", args.TableID)
	if err != nil {
		return nil, err
	}
	return b.resource(ctx, "get_table_info", path+"/tables/"+table, nil)
}

type getJobInfoArgs struct {
	ProjectID string `json:"project_id"`
	JobID     string `json:"job_id"`
}

func (b *bigQuery) getJobInfo(ctx agent.Context, args getJobInfoArgs) (any, error) {
	project, err := pathSegment("project_id", b.project(args.ProjectID))
	if err != nil {
		return nil, err
	}
	job, err := pathSegment("job_id", args.JobID)
	if err != nil {
		return nil, err
	}
	var query url.Values
	if b.location != "" {
		query = url.Values{"location": {b.location}}
	}
	return b.resource(ctx, "get_job_info", "projects/"+project+"/jobs/"+job, query)
}

func datasetPath(projectID, datasetID string) (string, error) {
	project, err := pathSegment("project_id", projectID)
	if err != nil {
		return "", err
	}
	dataset, err := pathSegment("dataset_id", datasetID)
	if err != nil {
		return "", err
	}
	return "projects/" + project + "/datasets/" + dataset, nil
}

// resource fetches one REST resource as a JSON object.
func (b *bigQuery) resource(ctx agent.Context, toolName, path string, query url.Values) (map[string]any, error) {
	var out map[string]any
	if err := b.client.get(ctx, toolName, path, query, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// pages fetches every page of a REST list. add takes in one page and returns
// the token of the next, or "" after the last page.
func pages[P any](b *bigQuery, ctx agent.Context, toolName, path string, add func(*P) string) error {
	query := url.Values{}
	seen := map[string]bool{}
	for {
		var p P
		if err := b.client.get(ctx, toolName, path, query, &p); err != nil {
			return err
		}
		next := add(&p)
		if next == "" {
			return nil
		}
		if seen[next] {
			return errors.New("BigQuery returned a page token it had already returned")
		}
		seen[next] = true
		query.Set("pageToken", next)
	}
}
