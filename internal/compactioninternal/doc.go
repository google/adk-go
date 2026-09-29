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

// Package compactioninternal implements the context-compaction algorithms:
// choosing which events to summarize, substituting summaries into a prompt, and
// recovering function calls that a summary swallowed.
//
// These are mechanics rather than API. The user-facing surface is
// [google.golang.org/adk/v2/session/compaction], which holds the configuration
// and the Summarizer extension point. Keeping the algorithms here lets them
// change without breaking anyone.
package compactioninternal
