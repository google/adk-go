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

// Package workflow runs graphs of nodes as an agent. A workflow is a
// set of edges connecting nodes; the scheduler walks the graph from
// Start, runs each node, and routes its output to successors.
//
// # Schemas and validation
//
// Every node carries an optional input schema and an optional output
// schema (see [Node.InputSchema] and [Node.OutputSchema], both
// *jsonschema.Resolved). A nil schema disables validation for that
// direction. Schemas are enforced by the scheduler, which is the single
// authoritative validation point — nodes do not validate inline.
//
// Input and output validation are symmetric:
//
//   - [Node.ValidateInput] is called once per activation, before the
//     node's Run, on the value handed to the node.
//   - [Node.ValidateOutput] is called on every yielded event whose
//     Output is non-nil, before the event is forwarded to the consumer
//     and to successor nodes. Events without output (progress, routing,
//     or human-input-request events) bypass output validation.
//
// Both methods return the validated value, which may be coerced or
// transformed (for example, a map[string]any coerced into the declared
// shape), or an error. A failing ValidateOutput ends the activation and
// surfaces the error to the caller.
//
// # Default output validation and the Content fallback
//
// [BaseNode] supplies the default [BaseNode.ValidateOutput], which
// delegates to defaultValidateOutput. Its strategy is:
//
//  1. A nil schema returns the value unchanged.
//  2. Framework control values (*session.Event, *session.RequestInput)
//     pass through unchanged: they are routed through Event.Output by
//     some nodes but are not user output payloads and must never be
//     schema-validated.
//  3. The value is validated against the schema. On success it is
//     returned unchanged.
//  4. If validation fails and the value is a *genai.Content, the helper
//     falls back to extracting the concatenated text of the content's
//     parts and either returns the text directly (when the schema's
//     root type is "string") or JSON-parses the text and re-validates
//     the parsed value.
//  5. If neither the standard path nor the fallback succeeds, the
//     original validation error is returned.
//
// The Content fallback mirrors adk-python's _validate_output_data and
// lets LlmAgent-like nodes that yield raw model output as *genai.Content
// project it onto their declared output schema. Note the order is the
// inverse of input validation: input validation tries the Content path
// first (input from Start is typically Content), while output validation
// tries standard validation first (nodes usually emit structured
// values) and only falls back to Content on failure.
//
// # Per-node specifics
//
// ToolNode overrides [ToolNode.ValidateOutput] to add a tool-specific
// fallback: when a map output of shape {"result": X} fails direct
// validation, it retries against the unwrapped X and returns that value
// on success. This preserves the FunctionTool convention of wrapping a
// scalar return as {"result": ...}. It is deliberately not a general
// default, because unwrapping a "result" key could mask genuine
// validation errors in other node types.
//
// JoinNode inherits the default ValidateOutput. Its output is the
// aggregated map[string]any of its predecessors' outputs, keyed by
// predecessor name; a JoinNode output schema (if set) is validated
// against that whole map.
//
// AgentNode participates in two independent validation layers — the
// LlmAgent.OutputSchema (LLM-generation schema, validated inside the
// agent before writing to state[OutputKey]) and the AgentNode's own
// output schema (the workflow-boundary gate applied by the scheduler).
// See [AgentNode] for the full discussion. The recommended pattern is
// to declare the same Go type at both layers:
//
//	type Review struct {
//	    Summary string `json:"summary"`
//	    Score   int    `json:"score"`
//	}
//
//	// Layer 1: the LlmAgent validates the model's text output against
//	// OutputSchema before writing the parsed object to state["review"].
//	reviewer, _ := llmagent.New(llmagent.Config{
//	    Name:         "reviewer",
//	    Model:        model,
//	    OutputSchema: reviewSchema, // *genai.Schema for Review
//	    OutputKey:    "review",
//	})
//
//	// Layer 2: AgentNode's Output type parameter is the same Review
//	// type, so the scheduler validates each yielded event's output at
//	// the workflow boundary using the same shape.
//	node, _ := workflow.NewAgentNodeTyped[any, Review](reviewer, cfg)
//
// # Adjacent-node schema agreement
//
// The output schema of a node and the input schema of its successor
// describe the same value as it crosses the edge between them. Keeping
// them in agreement (typically the same Go type) means the value a node
// emits validates cleanly as the next node's input. OutputSchema is the
// producing side of that contract and InputSchema the consuming side.
//
// # Single terminal output
//
// A workflow has at most one terminal output. A terminal node is one
// with no outgoing edges (excluding Start). After a clean run the
// scheduler checks the terminal nodes that actually produced output: if
// more than one did, the run fails with [ErrMultipleTerminalOutputs]
// naming them, because the workflow's output would otherwise be
// ambiguous.
//
// This is a runtime check on the nodes that produced output in a given
// activation, not a topological constraint. Fan-out and
// conditional-routing graphs frequently have several terminal branches;
// they remain valid as long as at most one terminal branch yields output
// per run. To combine several output-producing branches into a single
// terminal, fan them in to a [JoinNode]. The check is skipped while a
// run is draining (consumer stop or node failure) and while any node is
// paused on a human-input interrupt (the run has not finished, so its
// terminal output is undetermined). It mirrors adk-python's
// Workflow._finalize.
package workflow
