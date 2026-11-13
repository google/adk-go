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

// Package planner makes an LLM agent plan and reason before it answers. Set a
// planner on llmagent.Config.Planner; the agent's flow then consults it around
// every model call of the agent:
//
//   - Before the call, a built-in planner (a BuiltInPlanner, see BuiltIn) sets
//     the request's thinking config. Any other planner returns a planning
//     instruction, which the flow appends to the request's system instruction,
//     and the flow clears the Thought flag on every part of the conversation
//     history in the request.
//   - After the call, if the planner implements RewritingPlanner, the flow
//     passes it the parts of the model response and replaces them with the
//     parts it returns; an empty result keeps the model's parts.
//
// The package provides two planners. BuiltInPlanner turns on the model's own
// thinking feature through a genai.ThinkingConfig, for models that support
// thinking. PlanReActPlanner is for models that do not: it tells the model to
// write a plan, its reasoning and a final answer under tags such as
// /*PLANNING*/ and /*FINAL_ANSWER*/, and rewrites each response so that the
// plan and the reasoning become thought parts (genai.Part.Thought is true) and
// the final answer is plain text. In both cases the agent's events carry the
// model's reasoning as thought parts, so an application that shows only the
// answer skips the parts whose Thought is true. In live mode
// (runner.Runner.RunLive), BuiltInPlanner has no effect and PlanReActPlanner
// marks only part of the reasoning as thoughts; their doc comments say why.
//
// To write a planner of your own, implement Planner when the planner only adds
// an instruction, or RewritingPlanner when it also rewrites model responses. A
// RewritingPlanner can write session state through its agent.Context, and the
// flow records that change in an event of its own before the model response
// event.
//
// EXPERIMENTAL: This package is experimental; its API and behavior may change or be removed in a later version.
package planner
