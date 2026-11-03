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
	"fmt"
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

var planningTags = [...]string{
	planningTag,
	reasoningTag,
	actionTag,
	replanningTag,
}

// NewPlanReActPlanner creates a Plan-Re-Act [Planner] that constrains the LLM
// response to generate a plan before any action or observation.
//
// This planner does not require the model to support built-in thinking features
// or setting a ThinkingConfig.
func NewPlanReActPlanner() Planner {
	return &planReActPlanner{}
}

type planReActPlanner struct{}

func (p *planReActPlanner) BuildPlanningInstruction(agent.ReadonlyContext, *model.LLMRequest) string {
	return buildNLPlannerInstruction()
}

func (p *planReActPlanner) ProcessPlanningResponse(_ agent.Context, responseParts []*genai.Part) []*genai.Part {
	if len(responseParts) == 0 {
		return nil
	}

	var preservedParts []*genai.Part
	firstFCPartIndex := -1
	for i, responsePart := range responseParts {
		if responsePart == nil {
			continue
		}
		// Stop at the first (group of) function calls.
		if responsePart.FunctionCall != nil {
			// Ignore and filter out function calls with empty names.
			if responsePart.FunctionCall.Name == "" {
				continue
			}
			preservedParts = append(preservedParts, responsePart)
			firstFCPartIndex = i
			break
		}

		// Split the response into reasoning and final answer parts.
		preservedParts = handleNonFunctionCallParts(responsePart, preservedParts)
	}

	if firstFCPartIndex >= 0 {
		j := firstFCPartIndex + 1
		for j < len(responseParts) {
			if responseParts[j] != nil && responseParts[j].FunctionCall != nil {
				preservedParts = append(preservedParts, responseParts[j])
				j++
			} else {
				break
			}
		}
	}

	return preservedParts
}

func handleNonFunctionCallParts(responsePart *genai.Part, preservedParts []*genai.Part) []*genai.Part {
	if responsePart.Text != "" && strings.Contains(responsePart.Text, finalAnswerTag) {
		reasoningText, finalAnswerText := splitByLastPattern(responsePart.Text, finalAnswerTag)
		// splitByLastPattern includes the separator in the left part; strip it
		// so the reasoning block contains only the actual reasoning text.
		reasoningText = strings.TrimSuffix(reasoningText, finalAnswerTag)
		reasoningText = stripPlanningTags(reasoningText)
		if reasoningText != "" {
			reasoningPart := &genai.Part{Text: reasoningText}
			markAsThought(reasoningPart)
			preservedParts = append(preservedParts, reasoningPart)
		}
		if finalAnswerText != "" {
			preservedParts = append(preservedParts, &genai.Part{Text: finalAnswerText})
		}
		return preservedParts
	}

	responseText := responsePart.Text
	// If the part is a text part with a leading planning/reasoning/action tag,
	// label it as reasoning and strip all tags to produce clean text.
	if responseText != "" && hasLeadingPlanningTag(responseText) {
		partCopy := *responsePart
		partCopy.Text = stripPlanningTags(responseText)
		markAsThought(&partCopy)
		preservedParts = append(preservedParts, &partCopy)
		return preservedParts
	}

	return append(preservedParts, responsePart)
}

func splitByLastPattern(text, separator string) (string, string) {
	index := strings.LastIndex(text, separator)
	if index == -1 {
		return text, ""
	}
	return text[:index+len(separator)], text[index+len(separator):]
}

func hasLeadingPlanningTag(text string) bool {
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

func markAsThought(responsePart *genai.Part) {
	responsePart.Thought = true
}

func buildNLPlannerInstruction() string {
	highLevelPreamble := fmt.Sprintf(`
When answering the question, try to leverage the available tools to gather the information instead of your memorized knowledge.

Follow this process when answering the question: (1) first come up with a plan in natural language text format; (2) Then use tools to execute the plan and provide reasoning between tool code snippets to make a summary of current state and next step. Tool code snippets and reasoning should be interleaved with each other. (3) In the end, return one final answer.

Follow this format when answering the question: (1) The planning part should be under %s. (2) The tool code snippets should be under %s, and the reasoning parts should be under %s. (3) The final answer part should be under %s.
`, planningTag, actionTag, reasoningTag, finalAnswerTag)

	planningPreamble := fmt.Sprintf(`
Below are the requirements for the planning:
The plan is made to answer the user query if following the plan. The plan is coherent and covers all aspects of information from user query, and only involves the tools that are accessible by the agent. The plan contains the decomposed steps as a numbered list where each step should use one or multiple available tools. By reading the plan, you can intuitively know which tools to trigger or what actions to take.
If the initial plan cannot be successfully executed, you should learn from previous execution results and revise your plan. The revised plan should be under %s. Then use tools to follow the new plan.
`, replanningTag)

	reasoningPreamble := `
Below are the requirements for the reasoning:
The reasoning makes a summary of the current trajectory based on the user query and tool outputs. Based on the tool outputs and plan, the reasoning also comes up with instructions to the next steps, making the trajectory closer to the final answer.
`

	finalAnswerPreamble := `
Below are the requirements for the final answer:
The final answer should be precise and follow query formatting requirements. Some queries may not be answerable with the available tools and information. In those cases, inform the user why you cannot process their query and ask for more information.
`

	// Only contains the requirements for custom tool/libraries.
	toolCodePreamble := `
Below are the requirements for the tool code:

**Custom Tools:** The available tools are described in the context and can be directly used.
- Code must be valid self-contained Python snippets with no imports and no references to tools or Python libraries that are not in the context.
- You cannot use any parameters or fields that are not explicitly defined in the APIs in the context.
- The code snippets should be readable, efficient, and directly relevant to the user query and reasoning steps.
- When using the tools, you should use the library name together with the function name, e.g., vertex_search.search().
- If Python libraries are not provided in the context, NEVER write your own code other than the function calls using the provided tools.
`

	userInputPreamble := `
VERY IMPORTANT instruction that you MUST follow in addition to the above instructions:

You should ask for clarification if you need more information to answer the question.
You should prefer using the information available in the context instead of repeated tool use.
`

	return strings.Join([]string{
		highLevelPreamble,
		planningPreamble,
		reasoningPreamble,
		finalAnswerPreamble,
		toolCodePreamble,
		userInputPreamble,
	}, "\n\n")
}
