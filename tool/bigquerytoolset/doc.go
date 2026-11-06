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

// Package bigquerytoolset provides a [tool.Toolset] whose tools let an agent
// explore BigQuery: list the datasets in a project and read the metadata of a
// dataset.
//
// The tools have the same names, arguments and results as adk-python's
// BigQueryToolset, so prompts and skills written for it work unchanged. When
// BigQuery rejects a call, for example because a dataset does not exist, or
// an id argument is empty, the tool returns
// {"status": "ERROR", "error_details": "..."} to the model so it can correct
// the call. A canceled or expired context fails the tool call instead.
//
// By default the tools authenticate with Application Default Credentials:
//
//	bq, err := bigquerytoolset.New(ctx, bigquerytoolset.Config{
//		DefaultProjectID: "my-project",
//	})
//	if err != nil {
//		return err
//	}
//	agent, err := llmagent.New(llmagent.Config{
//		// ...
//		Toolsets: []tool.Toolset{bq},
//	})
//
// To expose only some of the tools, wrap the toolset with
// [tool.FilterToolset].
package bigquerytoolset
