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

package agent

import "errors"

// ErrInvalidTransferTarget is matched, with [errors.Is], by the error that
// fails an LLM agent's run when a transfer_to_agent call names an agent of the
// agent tree that is not among the caller's allowed transfer targets: its
// sub-agents, its parent and its peers, minus those its configuration
// disallows. The run fails after the function response event of the call has
// been yielded.
//
// It never matches a transfer that names the caller itself, nor one that names
// no agent of the tree.
var ErrInvalidTransferTarget = errors.New("invalid transfer target")
