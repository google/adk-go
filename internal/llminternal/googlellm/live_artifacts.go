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

package googlellm

import (
	"strings"

	"google.golang.org/genai"
)

// LiveArtifactNamespace is the path segment the live audio cache stores its
// blobs under when it builds an artifact:// reference. Callers that assemble
// such a URI and callers that recognize one must agree on this value.
const LiveArtifactNamespace = "_adk_live"

const (
	artifactURIScheme       = "artifact://"
	liveArtifactPathSegment = "/" + LiveArtifactNamespace + "/"
)

// IsInternalLiveArtifactPart reports whether part is an artifact:// reference
// to a blob the live flow cached for itself, such as the recorded input and
// output audio the audio cache manager saves.
//
// These references are ADK bookkeeping rather than conversation content, which
// is what separates them from the user-visible artifact:// references the same
// history carries and keeps. The API rejects them either way - see #1764, where
// replaying one answers generateContent with a 400 and closes the Live socket -
// so callers that hand a session's contents to a model drop them first.
func IsInternalLiveArtifactPart(part *genai.Part) bool {
	if part == nil || part.FileData == nil {
		return false
	}
	return strings.HasPrefix(part.FileData.FileURI, artifactURIScheme) &&
		strings.Contains(part.FileData.FileURI, liveArtifactPathSegment)
}
