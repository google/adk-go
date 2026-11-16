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

// Package fencing marks untrusted text in model requests so that server-supplied
// tool metadata and relayed peer content are presented as data rather than
// instructions.
//
// Part of what a model request carries can be written by an attacker: a tool
// description supplied by a third-party MCP server, another agent's turn, a
// tool result. That text travels on the same channel as the developer's
// instructions and the user's messages, so a payload posing as a directive is
// otherwise indistinguishable from one.
//
// Fencing encloses such a payload between a begin marker and an end marker and
// tells the model, in the request itself, that what sits between the markers is
// data to read and not instructions to follow. Two pairs of markers exist:
//
//   - [UntrustedToolDescriptionBegin] and [UntrustedToolDescriptionEnd] enclose
//     a description that a tool or its server supplied ([ToolDescription],
//     [Schema]). A request carrying such descriptions also gets
//     [ToolDescriptionPreamble] in its system instruction.
//   - [QuotedContentBegin] and [QuotedContentEnd] enclose text relayed from
//     another agent or peer ([Quote]).
//
// Before enclosing a payload, every function replaces the markers that the
// payload could use to forge the end of its block by [QuotedContentElided].
// [SystemInstructionBegin] and [SystemInstructionEnd], which enclose a dynamic
// system instruction carried in the request contents, are elided from tool
// descriptions for the same reason. Markers match exact code points: Unicode
// lookalikes are left as they are, and they cannot close a block either.
//
// Fencing raises the bar rather than closes the class: a model can still be
// talked round by text it was told to distrust. What it removes is the
// structural ambiguity about where untrusted text starts and ends.
//
// The marker texts and the preamble equal adk-python's
// (flows/llm_flows/context/_fencing.py) byte for byte, so requests built by
// either implementation read the same to a model. Every identifier in this
// package is a constant or a pure function, safe for concurrent use.
package fencing
