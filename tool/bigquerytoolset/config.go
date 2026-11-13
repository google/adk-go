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
	"strings"

	"google.golang.org/api/option"
)

// Config configures the BigQuery toolset. The zero value is valid: the tools
// authenticate with Application Default Credentials and the model names the
// project and dataset in every call.
type Config struct {
	// ClientOptions configure the connection to the BigQuery API, for example
	// [option.WithAuthCredentialsFile], [option.WithTokenSource] or
	// [option.WithEndpoint]. Without credentials among them, the tools use
	// Application Default Credentials with the BigQuery scope.
	ClientOptions []option.ClientOption

	// ExternalAccessTokenKey, when set, is the session state key holding the
	// OAuth access token each call authenticates with, for an application that
	// obtains the end user's token itself. A call fails, and the model is told
	// so, when the state has no non-empty string under the key. New rejects
	// [option.WithAPIKey], [option.WithTokenSource], [option.WithCredentials],
	// [option.WithAuthCredentials], [option.WithAuthCredentialsFile] and
	// [option.WithAuthCredentialsJSON] alongside it. It cannot see inside an
	// [option.WithHTTPClient]: such a client must not authenticate requests
	// itself, or its credentials replace the end user's.
	ExternalAccessTokenKey string

	// ApplicationName is added to the user agent of BigQuery requests, so
	// BigQuery usage can be attributed to the agent. An [option.WithUserAgent]
	// in ClientOptions replaces the whole header, including this name. It must
	// not contain spaces or characters a request header cannot carry.
	ApplicationName string

	// Location is the BigQuery location, such as "US" or "europe-west4", that
	// execute_sql runs queries in and get_job_info looks jobs up in. Unset,
	// BigQuery picks the location of a query from the tables it reads, and
	// jobs outside the US and EU multi-regions are not found.
	Location string

	// ComputeProjectID, when set, is the only project execute_sql runs
	// queries in, as a guardrail on where compute is billed. The model still
	// names the project, and a query in any other project fails.
	// DefaultProjectID, if also set, must be the same project.
	ComputeProjectID string

	// MaxQueryResultRows is the most rows execute_sql returns from a query.
	// Zero means 50, adk-python's default. When rows are left out, the result
	// has result_is_likely_truncated set to true. adk-python also sets it when
	// a query returns exactly this many rows.
	MaxQueryResultRows int

	// MaximumBytesBilled, when set, limits the bytes a query run by
	// execute_sql can bill: BigQuery fails a query that would bill more. It
	// must be at least 10485760 (10 MiB), the least BigQuery bills for a
	// query.
	MaximumBytesBilled int64

	// JobLabels are added to the BigQuery jobs execute_sql creates, for
	// tracking and monitoring. There can be at most 20, and keys starting
	// with "adk-bigquery-" are reserved for the labels the tools add
	// themselves: adk-bigquery-tool with the tool name, and
	// adk-bigquery-application-name with ApplicationName when it is set.
	JobLabels map[string]string

	// DefaultProjectID is the project the tools use when the model does not
	// name one. Setting it makes the project_id argument optional, so the
	// model no longer has to ask the user for a project. The model can still
	// name another project.
	DefaultProjectID string

	// DefaultDatasetID is the dataset the tools that take a dataset_id
	// argument use when the model does not name one. Setting it makes that
	// argument optional. The model can still name another dataset.
	DefaultDatasetID string
}

const (
	defaultMaxQueryResultRows = 50

	// minMaximumBytesBilled is the least BigQuery bills for a query, so a
	// lower MaximumBytesBilled would fail every query.
	minMaximumBytesBilled = 10 << 20

	maxJobLabels        = 20
	reservedLabelPrefix = "adk-bigquery-"
)

func (c Config) validate() error {
	for _, r := range c.ApplicationName {
		if r <= ' ' || r > '~' {
			return errors.New("bigquerytoolset: Config.ApplicationName must contain only printable ASCII characters other than space")
		}
	}
	if c.DefaultProjectID != "" && c.ComputeProjectID != "" && c.DefaultProjectID != c.ComputeProjectID {
		return fmt.Errorf("bigquerytoolset: Config.DefaultProjectID %q differs from Config.ComputeProjectID %q, so execute_sql would default to a project it rejects; leave DefaultProjectID unset to let the model pick the data project", c.DefaultProjectID, c.ComputeProjectID)
	}
	if c.MaxQueryResultRows < 0 {
		return errors.New("bigquerytoolset: Config.MaxQueryResultRows must not be negative")
	}
	if c.MaximumBytesBilled != 0 && c.MaximumBytesBilled < minMaximumBytesBilled {
		return fmt.Errorf("bigquerytoolset: Config.MaximumBytesBilled must be at least %d, the least BigQuery bills for a query", minMaximumBytesBilled)
	}
	if len(c.JobLabels) > maxJobLabels {
		return fmt.Errorf("bigquerytoolset: Config.JobLabels has %d labels, more than the %d allowed", len(c.JobLabels), maxJobLabels)
	}
	for key := range c.JobLabels {
		if key == "" {
			return errors.New("bigquerytoolset: Config.JobLabels must not have an empty key")
		}
		if strings.HasPrefix(key, reservedLabelPrefix) {
			return fmt.Errorf("bigquerytoolset: Config.JobLabels key %q starts with %q, which is reserved for the labels the tools add", key, reservedLabelPrefix)
		}
	}
	return nil
}
