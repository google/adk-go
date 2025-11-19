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

package planner

import (
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
)

const (
	// PlanningTag marks the planning section of the response
	PlanningTag = "/*PLANNING*/"
	// ReplanningTag marks the replanning section of the response
	ReplanningTag = "/*REPLANNING*/"
	// ReasoningTag marks the reasoning section of the response
	ReasoningTag = "/*REASONING*/"
	// ActionTag marks the action section of the response
	ActionTag = "/*ACTION*/"
	// FinalAnswerTag marks the final answer section of the response
	FinalAnswerTag = "/*FINAL_ANSWER*/"
)

// ReActPlanner is the Plan-Re-Act planner that constrains the LLM response to generate a plan before any action/observation.
//
// Note: this planner does not require the model to support built-in thinking
// features or setting the thinking config.
type ReActPlanner struct{}

// NewReActPlanner creates a new ReActPlanner.
func NewReActPlanner() *ReActPlanner {
	return &ReActPlanner{}
}

// BuildPlanningInstruction implements BasePlanner.
func (p *ReActPlanner) BuildPlanningInstruction(readonlyContext agent.ReadonlyContext, llmRequest *model.LLMRequest) string {
	return p.buildNLPlannerInstruction()
}

// ProcessPlanningResponse implements BasePlanner.
func (p *ReActPlanner) ProcessPlanningResponse(callbackContext agent.CallbackContext, responseParts []*genai.Part) []*genai.Part {
	if len(responseParts) == 0 {
		return nil
	}

	var preservedParts []*genai.Part
	firstFCPartIndex := -1

	for i, part := range responseParts {
		if part.FunctionCall != nil {
			if part.FunctionCall.Name == "" {
				continue
			}
			preservedParts = append(preservedParts, part)
			firstFCPartIndex = i
			break
		}

		preservedParts = p.handleNonFunctionCallParts(part, preservedParts)
	}

	if firstFCPartIndex > 0 {
		for j := firstFCPartIndex + 1; j < len(responseParts); j++ {
			if responseParts[j].FunctionCall == nil {
				break
			}
			preservedParts = append(preservedParts, responseParts[j])
		}
	}

	return preservedParts
}

// splitByLastPattern splits the text by the last occurrence of the separator.
func (p *ReActPlanner) splitByLastPattern(text, separator string) (string, string) {
	index := strings.LastIndex(text, separator)
	if index == -1 {
		return text, ""
	}
	return text[:index+len(separator)], text[index+len(separator):]
}

// handleNonFunctionCallParts handles non-function-call parts of the response.
func (p *ReActPlanner) handleNonFunctionCallParts(responsePart *genai.Part, preservedParts []*genai.Part) []*genai.Part {
	if responsePart.Text != "" && strings.Contains(responsePart.Text, FinalAnswerTag) {
		reasoningText, finalAnswerText := p.splitByLastPattern(responsePart.Text, FinalAnswerTag)
		if reasoningText != "" {
			reasoningPart := &genai.Part{Text: reasoningText}
			p.markAsThought(reasoningPart)
			preservedParts = append(preservedParts, reasoningPart)
		}
		if finalAnswerText != "" {
			preservedParts = append(preservedParts, &genai.Part{Text: finalAnswerText})
		}
	} else {
		responseText := responsePart.Text
		// If the part is a text part with a planning/reasoning/action tag,
		// label it as reasoning.
		if responseText != "" && (strings.HasPrefix(responseText, PlanningTag) ||
			strings.HasPrefix(responseText, ReasoningTag) ||
			strings.HasPrefix(responseText, ActionTag) ||
			strings.HasPrefix(responseText, ReplanningTag)) {
			p.markAsThought(responsePart)
		}
		preservedParts = append(preservedParts, responsePart)
	}

	return preservedParts
}

// markAsThought marks the response part as thought.
func (p *ReActPlanner) markAsThought(responsePart *genai.Part) {
	if responsePart.Text != "" {
		responsePart.Thought = true
	}
}

// buildNLPlannerInstruction builds the NL planner instruction for the Plan-Re-Act planner.
func (p *ReActPlanner) buildNLPlannerInstruction() string {
	highLevelPreamble := `
When answering the question, try to leverage the available tools to gather the information instead of your memorized knowledge.

Follow this process when answering the question: (1) first come up with a plan in natural language text format; (2) Then use tools to execute the plan and provide reasoning between tool code snippets to make a summary of current state and next step. Tool code snippets and reasoning should be interleaved with each other. (3) In the end, return one final answer.

Follow this format when answering the question: (1) The planning part should be under ` + PlanningTag + `. (2) The tool code snippets should be under ` + ActionTag + `, and the reasoning parts should be under ` + ReasoningTag + `. (3) The final answer part should be under ` + FinalAnswerTag + `.`

	planningPreamble := `
Below are the requirements for the planning:
The plan is made to answer the user query if following the plan. The plan is coherent and covers all aspects of information from user query, and only involves the tools that are accessible by the agent. The plan contains the decomposed steps as a numbered list where each step should use one or multiple available tools. By reading the plan, you can intuitively know which tools to trigger or what actions to take.
If the initial plan cannot be successfully executed, you should learn from previous execution results and revise your plan. The revised plan should be be under ` + ReplanningTag + `. Then use tools to follow the new plan.`

	reasoningPreamble := `
Below are the requirements for the reasoning:
The reasoning makes a summary of the current trajectory based on the user query and tool outputs. Based on the tool outputs and plan, the reasoning also comes up with instructions to the next steps, making the trajectory closer to the final answer.`

	finalAnswerPreamble := `
Below are the requirements for the final answer:
The final answer should be precise and follow query formatting requirements. Some queries may not be answerable with the available tools and information. In those cases, inform the user why you cannot process their query and ask for more information.`

	// Only contains the requirements for custom tool/libraries.
	toolCodeWithoutPythonLibrariesPreamble := `
Below are the requirements for the tool code:

**Custom Tools:** The available tools are described in the context and can be directly used.
- Code must be valid self-contained Python snippets with no imports and no references to tools or Python libraries that are not in the context.
- You cannot use any parameters or fields that are not explicitly defined in the APIs in the context.
- The code snippets should be readable, efficient, and directly relevant to the user query and reasoning steps.
- When using the tools, you should use the library name together with the function name, e.g., vertex_search.search().
- If Python libraries are not provided in the context, NEVER write your own code other than the function calls using the provided tools.`

	userInputPreamble := `
VERY IMPORTANT instruction that you MUST follow in addition to the above instructions:

You should ask for clarification if you need more information to answer the question.
You should prefer using the information available in the context instead of repeated tool use.`

	return strings.Join([]string{
		highLevelPreamble,
		planningPreamble,
		reasoningPreamble,
		finalAnswerPreamble,
		toolCodeWithoutPythonLibrariesPreamble,
		userInputPreamble,
	}, "\n\n")
}
