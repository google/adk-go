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

package aiplatform

// HostUrl returns aiplatform host url for a given region
// Example: "us-central1-aiplatform.googleapis.com"
func HostUrl(region string) string {
	if region == "" || region == "global" {
		return "aiplatform.googleapis.com"
	}
	return region + "-aiplatform.googleapis.com"
}

// HostUrl returns aiplatform host url for a given region with port number
// Example: "us-central1-aiplatform.googleapis.com:443"
func HostPortUrl(region string) string {
	return HostUrl(region) + ":443"
}
