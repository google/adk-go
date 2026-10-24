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

package version

import (
	"runtime"
	"strings"
)

// TrackingHeader returns the client labels ADK puts at the front of the
// user-agent and x-goog-api-client headers so that services can attribute
// traffic to ADK Go, for example "google-adk/2.6.0 gl-go/1.26.5". Other tokens
// may follow them in the header. It has the same form as the
// labels adk-python builds in google/adk/utils/_client_labels_utils.py.
func TrackingHeader() string {
	return trackingHeader
}

var trackingHeader = "google-adk/" + Version + " gl-go/" + strings.TrimPrefix(runtime.Version(), "go")
