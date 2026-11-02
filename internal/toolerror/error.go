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

// Package toolerror identifies framework failures raised before a tool handler runs.
package toolerror

// BeforeHandler preserves argument-validation and confirmation errors for the
// model without treating unclassified custom-tool errors as safe to disclose.
// Unwrap retains sentinel matching for direct RunStream callers.
type BeforeHandler struct{ Err error }

func (e *BeforeHandler) Error() string { return e.Err.Error() }
func (e *BeforeHandler) Unwrap() error { return e.Err }
