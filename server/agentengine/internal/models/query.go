// Copyright 2025 Google LLC
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

// package models defines Request / Response models for generic Query and for more focused methods
package models

// Query is generic JSON format for payload coming from aiplatform. It has Input field which type is specific to a particular class_method.
// Other models are providing those structures for particular class_methods
type Query struct {
	ClassMethod string `json:"class_method"`
	Input       any    `json:"input"`
}
