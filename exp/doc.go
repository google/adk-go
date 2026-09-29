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

// Package exp holds ADK Go features that are not yet stable API, one
// subpackage per feature.
//
// Anything under exp may change, or be removed, in a later minor version
// without notice. A feature that proves itself moves out of exp into the
// stable packages; one that does not is removed, at the latest by the next
// major version.
//
// The subpackages are:
//
//   - [google.golang.org/adk/v2/exp/appinfo]: the app-info endpoint of the ADK
//     REST API, which describes an app's agents, instructions and tools
package exp
