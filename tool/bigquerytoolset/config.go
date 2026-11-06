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

	"google.golang.org/api/option"
)

// Config configures the BigQuery toolset. The zero value is valid: the tools
// authenticate with Application Default Credentials and the model names the
// project and dataset in every call.
type Config struct {
	// ClientOptions configure the connection to the BigQuery API, for example
	// [option.WithCredentialsFile], [option.WithTokenSource] or
	// [option.WithEndpoint]. Without credentials among them, the tools use
	// Application Default Credentials with the BigQuery scope.
	ClientOptions []option.ClientOption

	// ExternalAccessTokenKey, when set, is the session state key holding the
	// OAuth access token each call authenticates with, for an application that
	// obtains the end user's token itself. A call fails, and the model is told
	// so, when the state has no non-empty string under the key. New rejects
	// ClientOptions that provide credentials alongside it, but cannot see
	// inside an [option.WithHTTPClient]: such a client must not authenticate
	// requests itself, or its credentials replace the end user's.
	ExternalAccessTokenKey string

	// ApplicationName is added to the user agent of every BigQuery request,
	// so BigQuery usage can be attributed to the agent. It must not contain
	// spaces or characters a request header cannot carry.
	ApplicationName string

	// Location is the BigQuery location the job metadata tool looks jobs up
	// in, such as "US" or "europe-west4". Jobs outside the US and EU
	// multi-regions are found only when it is set.
	Location string

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

func (c Config) validate() error {
	for _, r := range c.ApplicationName {
		if r <= ' ' || r > '~' {
			return errors.New("bigquerytoolset: Config.ApplicationName must contain only printable ASCII characters other than space")
		}
	}
	return nil
}
