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

package planner

import (
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
)

const (
	planningTag    = "/*PLANNING*/"
	replanningTag  = "/*REPLANNING*/"
	reasoningTag   = "/*REASONING*/"
	actionTag      = "/*ACTION*/"
	finalAnswerTag = "/*FINAL_ANSWER*/"
)

// The six preambles of the Plan-Re-Act instruction. Each starts and ends with
// a newline, as the triple-quoted strings in adk-python's
// planners/plan_re_act_planner.py do, so the joined text matches Python's
// byte for byte.
const (
	highLevelPreamble = `
When answering the question, try to leverage the available tools to gather the information instead of your memorized knowledge.

Follow this process when answering the question: (1) first come up with a plan in natural language text format; (2) Then use tools to execute the plan and provide reasoning between tool code snippets to make a summary of current state and next step. Tool code snippets and reasoning should be interleaved with each other. (3) In the end, return one final answer.

Follow this format when answering the question: (1) The planning part should be under ` + planningTag + `. (2) The tool code snippets should be under ` + actionTag + `, and the reasoning parts should be under ` + reasoningTag + `. (3) The final answer part should be under ` + finalAnswerTag + `.
`

	planningPreamble = `
Below are the requirements for the planning:
The plan is made to answer the user query if following the plan. The plan is coherent and covers all aspects of information from user query, and only involves the tools that are accessible by the agent. The plan contains the decomposed steps as a numbered list where each step should use one or multiple available tools. By reading the plan, you can intuitively know which tools to trigger or what actions to take.
If the initial plan cannot be successfully executed, you should learn from previous execution results and revise your plan. The revised plan should be under ` + replanningTag + `. Then use tools to follow the new plan.
`

	reasoningPreamble = `
Below are the requirements for the reasoning:
The reasoning makes a summary of the current trajectory based on the user query and tool outputs. Based on the tool outputs and plan, the reasoning also comes up with instructions to the next steps, making the trajectory closer to the final answer.
`

	finalAnswerPreamble = `
Below are the requirements for the final answer:
The final answer should be precise and follow query formatting requirements. Some queries may not be answerable with the available tools and information. In those cases, inform the user why you cannot process their query and ask for more information.
`

	// Only contains the requirements for custom tools and libraries.
	toolCodeWithoutPythonLibrariesPreamble = `
Below are the requirements for the tool code:

**Custom Tools:** The available tools are described in the context and can be directly used.
- Code must be valid self-contained Python snippets with no imports and no references to tools or Python libraries that are not in the context.
- You cannot use any parameters or fields that are not explicitly defined in the APIs in the context.
- The code snippets should be readable, efficient, and directly relevant to the user query and reasoning steps.
- When using the tools, you should use the library name together with the function name, e.g., vertex_search.search().
- If Python libraries are not provided in the context, NEVER write your own code other than the function calls using the provided tools.
`

	userInputPreamble = `
VERY IMPORTANT instruction that you MUST follow in addition to the above instructions:

You should ask for clarification if you need more information to answer the question.
You should prefer using the information available in the context instead of repeated tool use.
`
)

// planReActInstruction is the instruction BuildPlanningInstruction returns.
var planReActInstruction = strings.Join([]string{
	highLevelPreamble,
	planningPreamble,
	reasoningPreamble,
	finalAnswerPreamble,
	toolCodeWithoutPythonLibrariesPreamble,
	userInputPreamble,
}, "\n\n")

// PlanReActPlanner constrains the model to generate a natural-language plan and
// reasoning between tool calls using /*PLANNING*/, /*REPLANNING*/,
// /*REASONING*/, /*ACTION*/, and /*FINAL_ANSWER*/ tags, and rewrites the
// response parts to strip planning tags and mark reasoning/planning sections as
// thoughts. It implements Planner and RewritingPlanner.
//
// Use it with a model that has no built-in thinking feature; for a model that
// has one, BuiltInPlanner lets the model plan on its own. In an agent run, the
// agent's events carry the plan and the reasoning as parts whose Thought is
// true and the final answer as a part whose Thought is false. A turn that the
// planner reduces to thought parts only ends the agent's step, as any final
// response does.
//
// With streaming (agent.StreamingModeSSE), the planner rewrites each partial
// response on its own. A partial response is marked as a thought only if its
// own text starts with a planning tag, so a partial event can show reasoning
// as plain text that the final, aggregated event marks as a thought, and a tag
// split across two partial responses is not recognized in either.
//
// PlanReActPlanner is stateless; its zero value and NewPlanReActPlanner() are
// identical and safe for concurrent use.
type PlanReActPlanner struct{}

// NewPlanReActPlanner returns a non-nil *PlanReActPlanner.
func NewPlanReActPlanner() *PlanReActPlanner {
	return &PlanReActPlanner{}
}

// BuildPlanningInstruction implements Planner. It ignores ctx and req, which
// may be nil, and always returns the same non-empty Plan-Re-Act system
// instruction, six preambles joined by "\n\n", and a nil error.
func (p *PlanReActPlanner) BuildPlanningInstruction(ctx agent.ReadonlyContext, req *model.LLMRequest) (string, error) {
	return planReActInstruction, nil
}

// ProcessPlanningResponse implements RewritingPlanner. It splits and filters
// responseParts according to the Plan-Re-Act tag rules. It ignores ctx, which
// may be nil, and never returns a non-nil error.
//
// For nil or empty responseParts it returns (nil, nil). Otherwise it returns a
// non-nil slice of the preserved parts. The leading parts before the first
// function call with a name are processed as follows:
//
//   - A part whose Text contains "/*FINAL_ANSWER*/" is split at the last
//     occurrence of that tag. The text before it has all four planning tags
//     ("/*PLANNING*/", "/*REASONING*/", "/*ACTION*/", "/*REPLANNING*/")
//     stripped and, if non-empty, becomes a part with Thought set to true.
//     The text after it is kept as is and, if non-empty, becomes a part with
//     Thought set to false. The other fields of the original part,
//     ThoughtSignature included, go to the final-answer part when that text
//     is non-empty, and to the reasoning part when the final-answer text is
//     empty and the reasoning text is not.
//   - For a part whose Text starts with one of the four planning tags, the
//     method strips all four planning tags from Text in place and sets
//     Thought to true on the original *genai.Part, even if Text becomes "".
//   - Any other part is appended unchanged.
//
// Function-call parts are handled as follows:
//
//   - A function-call part with an empty FunctionCall.Name is dropped.
//   - The first function-call part with a non-empty Name and every directly
//     following function-call part, apart from those with an empty Name, are
//     appended as the original *genai.Part pointers. Every part from the first
//     non-function-call part after that group onward is dropped.
func (p *PlanReActPlanner) ProcessPlanningResponse(ctx agent.Context, responseParts []*genai.Part) ([]*genai.Part, error) {
	if len(responseParts) == 0 {
		return nil, nil
	}

	preservedParts := []*genai.Part{}
	firstFCPartIndex := -1
	for i, part := range responseParts {
		// Stop at the first (group of) function calls.
		if part != nil && part.FunctionCall != nil {
			if part.FunctionCall.Name == "" {
				continue
			}
			firstFCPartIndex = i
			break
		}
		preservedParts = handleNonFunctionCallPart(part, preservedParts)
	}

	if firstFCPartIndex >= 0 {
		preservedParts = append(preservedParts, responseParts[firstFCPartIndex])
		for _, part := range responseParts[firstFCPartIndex+1:] {
			if part == nil || part.FunctionCall == nil {
				break
			}
			// adk-python keeps an empty-name call here
			// (plan_re_act_planner.py:79) although it drops one before the
			// group (:66-68). Go drops it in both places.
			if part.FunctionCall.Name == "" {
				continue
			}
			preservedParts = append(preservedParts, part)
		}
	}
	return preservedParts, nil
}

// handleNonFunctionCallPart appends what part becomes to preservedParts and
// returns the result. A part whose text contains the final-answer tag is
// split by appendFinalAnswerSplit. A part whose text starts with a planning
// tag has every planning tag stripped from its text and is marked as a
// thought, in place. Any other part, nil included, is appended unchanged.
func handleNonFunctionCallPart(part *genai.Part, preservedParts []*genai.Part) []*genai.Part {
	if part == nil {
		return append(preservedParts, part)
	}
	if strings.Contains(part.Text, finalAnswerTag) {
		return appendFinalAnswerSplit(part, preservedParts)
	}
	if hasPlanningTagPrefix(part.Text) {
		part.Text = stripPlanningTags(part.Text)
		part.Thought = true
	}
	return append(preservedParts, part)
}

// appendFinalAnswerSplit splits part.Text at the last final-answer tag and
// appends the non-empty halves to preservedParts: the text before the tag,
// with planning tags stripped, as a thought, then the text after the tag,
// unchanged, as a non-thought part. part itself is not modified.
func appendFinalAnswerSplit(part *genai.Part, preservedParts []*genai.Part) []*genai.Part {
	i := strings.LastIndex(part.Text, finalAnswerTag)
	reasoningText := stripPlanningTags(part.Text[:i])
	finalAnswerText := part.Text[i+len(finalAnswerTag):]

	// adk-python builds both halves as new parts with text only
	// (plan_re_act_planner.py:143-152) and so drops the thought signature and
	// the other fields. Go keeps them on one half: the answer when there is
	// one, so the planner-made reasoning part never looks like a model-signed
	// thought, and otherwise the reasoning.
	if finalAnswerText == "" {
		if reasoningText == "" {
			return preservedParts
		}
		reasoning := *part
		reasoning.Text = reasoningText
		reasoning.Thought = true
		return append(preservedParts, &reasoning)
	}
	if reasoningText != "" {
		preservedParts = append(preservedParts, &genai.Part{Text: reasoningText, Thought: true})
	}
	answer := *part
	answer.Text = finalAnswerText
	answer.Thought = false
	return append(preservedParts, &answer)
}

// planningTags are the tags stripped from reasoning text, in the order
// adk-python strips them (PlanReActPlanner._PLANNING_TAGS).
var planningTags = []string{planningTag, reasoningTag, actionTag, replanningTag}

func hasPlanningTagPrefix(text string) bool {
	for _, tag := range planningTags {
		if strings.HasPrefix(text, tag) {
			return true
		}
	}
	return false
}

func stripPlanningTags(text string) string {
	for _, tag := range planningTags {
		text = strings.ReplaceAll(text, tag, "")
	}
	return text
}
