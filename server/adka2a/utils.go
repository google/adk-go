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

package adka2a

import "github.com/a2aproject/a2a-go/a2a"

// WithoutPartialArtifacts returns a slice of artifacts without partial artifacts.
// Partial artifacts are usually discarded (contain no parts) after agent invocation is finished.
func WithoutPartialArtifacts(artifacts []*a2a.Artifact) []*a2a.Artifact {
	var result []*a2a.Artifact
	for _, artifact := range artifacts {
		if IsPartial(artifact.Metadata) {
			continue
		}
		result = append(result, artifact)
	}
	return result
}
