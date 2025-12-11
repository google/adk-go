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

package adka2a

import (
	"fmt"
	"regexp"
	"slices"
	"strings"

	"github.com/a2aproject/a2a-go/a2a"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/workflowagents/loopagent"
	iagent "google.golang.org/adk/internal/agent"
	"google.golang.org/adk/internal/llminternal"
)

// BuildAgentSkills attempts to create a list of [a2a.AgentSkill]s based on agent descriptions and types.
// This information can be used in [a2a.AgentCard] to help clients understand agent capabilities.
func BuildAgentSkills(agent agent.Agent) []a2a.AgentSkill {
	return slices.Concat(buildPrimarySkills(agent), buildSubAgentSkills(agent))
}

func buildPrimarySkills(agent agent.Agent) []a2a.AgentSkill {
	if llmAgent, ok := agent.(llminternal.Agent); ok {
		return buildLLMAgentSkills(agent, llminternal.Reveal(llmAgent))
	} else {
		return buildNonLLMAgentSkills(agent)
	}
}

func buildSubAgentSkills(agent agent.Agent) []a2a.AgentSkill {
	subAgents := agent.SubAgents()
	result := make([]a2a.AgentSkill, 0, len(agent.SubAgents()))
	for _, sub := range subAgents {
		skills := buildPrimarySkills(sub)
		for _, subSkill := range skills {
			skill := a2a.AgentSkill{
				ID:          fmt.Sprintf("%s_%s", sub.Name(), subSkill.ID),
				Name:        fmt.Sprintf("%s: %s", sub.Name(), subSkill.Name),
				Description: subSkill.Description,
				Tags:        slices.Concat([]string{fmt.Sprintf("sub_agent:%s", sub.Name())}, subSkill.Tags),
			}
			result = append(result, skill)
		}
	}
	return result
}

func buildLLMAgentSkills(agent agent.Agent, llmState *llminternal.State) []a2a.AgentSkill {
	skills := []a2a.AgentSkill{
		{
			ID:          agent.Name(),
			Name:        "model",
			Description: buildDescriptionFromInstructions(agent, llmState),
			Tags:        []string{"llm"},
		},
	}

	if len(llmState.Tools) > 0 {
		for _, tool := range llmState.Tools {
			description := tool.Description()
			if description == "" {
				description = fmt.Sprintf("Tool: %s", tool.Name())
			}
			skills = append(skills, a2a.AgentSkill{
				ID:          fmt.Sprintf("%s-%s", agent.Name(), tool.Name()),
				Name:        tool.Name(),
				Description: description,
				Tags:        []string{"llm", "tools"},
			})
		}
	}

	// TODO(yarolegovich): mention planning and code-execution skills once supported (and if configured)

	return skills
}

func buildNonLLMAgentSkills(agent agent.Agent) []a2a.AgentSkill {
	state := getInternalState(agent)
	skills := []a2a.AgentSkill{
		{
			ID:          agent.Name(),
			Name:        getAgentSkillName(state),
			Description: buildAgentDescription(agent, state),
			Tags:        []string{getAgentTypeTag(state)},
		},
	}

	subAgents := agent.SubAgents()
	if len(subAgents) > 0 {
		descriptions := make([]string, len(subAgents))
		for i, sub := range subAgents {
			if sub.Description() != "" {
				descriptions[i] = sub.Description()
			} else {
				descriptions[i] = "No description"
			}
		}
		skills = append(skills, a2a.AgentSkill{
			ID:          fmt.Sprintf("%s-sub-agents", agent.Name()),
			Name:        "sub-agents",
			Description: fmt.Sprintf("Orchestrates: %s", strings.Join(descriptions, "; ")),
			Tags:        []string{getAgentTypeTag(state), "orchestration"},
		})
	}

	return skills
}

func buildAgentDescription(agent agent.Agent, state *iagent.State) string {
	descriptionParts := []string{}

	if agent.Description() != "" {
		descriptionParts = append(descriptionParts, agent.Description())
	}

	if len(agent.SubAgents()) > 0 {
		switch state.AgentType {
		case iagent.TypeLoopAgent:
			descriptionParts = append(descriptionParts, buildLoopAgentDescription(agent, state))
		case iagent.TypeParallelAgent:
			descriptionParts = append(descriptionParts, buildParallelAgentDescription(agent))
		case iagent.TypeSequentialAgent:
			descriptionParts = append(descriptionParts, buildSequentialAgentDescription(agent))
		}
	}

	if len(descriptionParts) > 0 {
		return strings.Join(descriptionParts, " ")
	} else {
		return getDefaultAgentDescription(state)
	}
}

func buildSequentialAgentDescription(agnt agent.Agent) string {
	subAgents := agnt.SubAgents()
	descriptions := make([]string, len(subAgents))
	for i, sub := range subAgents {
		subDescription := sub.Description()
		if subDescription == "" {
			subDescription = fmt.Sprintf("execute the %s agent", sub.Name())
		}
		switch i {
		case 0:
			descriptions[i] = fmt.Sprintf("First, this agent will %s.", subDescription)
		case len(subAgents) - 1:
			descriptions[i] = fmt.Sprintf("Finally, this agent will %s.", subDescription)
		default:
			descriptions[i] = fmt.Sprintf("Then, this agent will %s.", subDescription)
		}
	}
	return strings.Join(descriptions, " ")
}

func buildParallelAgentDescription(agnt agent.Agent) string {
	subAgents := agnt.SubAgents()
	descriptions := make([]string, len(subAgents))
	for i, sub := range subAgents {
		subDescription := sub.Description()
		if subDescription == "" {
			subDescription = fmt.Sprintf("execute the %s agent", sub.Name())
		}
		switch i {
		case 0:
			descriptions[i] = fmt.Sprintf("This agent will %s", subDescription)
		case len(subAgents) - 1:
			descriptions[i] = fmt.Sprintf("and %s", subDescription)
		default:
			descriptions[i] = fmt.Sprintf(", %s", subDescription)
		}
	}
	return fmt.Sprintf("%s simultaneously.", strings.Join(descriptions, " "))
}

func buildLoopAgentDescription(agnt agent.Agent, state *iagent.State) string {
	llmConfig, ok := state.Config.(loopagent.Config)
	if !ok {
		return ""
	}
	maxIterations := "unlimited"
	if llmConfig.MaxIterations > 0 {
		maxIterations = fmt.Sprintf("%d", llmConfig.MaxIterations)
	}
	subAgents := agnt.SubAgents()
	descriptions := make([]string, len(subAgents))
	for i, sub := range subAgents {
		subDescription := sub.Description()
		if subDescription == "" {
			subDescription = fmt.Sprintf("execute the %s agent", sub.Name())
		}
		switch i {
		case 0:
			descriptions[i] = fmt.Sprintf("This agent will %s", subDescription)
		case len(subAgents) - 1:
			descriptions[i] = fmt.Sprintf("and %s", subDescription)
		default:
			descriptions[i] = fmt.Sprintf(", %s", subDescription)
		}
	}
	return fmt.Sprintf("%s in a loop (max %s iterations).", strings.Join(descriptions, " "), maxIterations)
}

func buildDescriptionFromInstructions(agent agent.Agent, llmState *llminternal.State) string {
	state := getInternalState(agent)
	descriptionParts := []string{}
	if agent.Description() != "" {
		descriptionParts = append(descriptionParts, agent.Description())
	}
	if llmState.Instruction != "" {
		descriptionParts = append(descriptionParts, replacePronouns(llmState.Instruction))
	}
	if llmState.GlobalInstruction != "" {
		descriptionParts = append(descriptionParts, replacePronouns(llmState.GlobalInstruction))
	}
	description := getDefaultAgentDescription(state)
	if len(descriptionParts) > 0 {
		description = strings.Join(descriptionParts, " ")
	}
	return description
}

// Replaces pronouns and conjugate common verbs for agent description.
// Examples: "You are" -> "I am", "your" -> "my"
func replacePronouns(instruction string) string {
	substitutions := []struct {
		original string
		target   string
	}{
		// Keep sorted by len(original) DESC to ensure longer phrases are matched first
		// which prevents "you" in "you are" from being replaced on its own.
		{"you were", "I was"},
		{"you are", "I am"},
		{"you're", "I am"},
		{"you've", "I have"},
		{"yours", "mine"},
		{"your", "my"},
		{"you", "I"},
	}
	for _, sub := range substitutions {
		pattern := regexp.MustCompile(fmt.Sprintf(`(?i)\b%s\b`, sub.original))
		instruction = pattern.ReplaceAllString(instruction, sub.target)
	}
	return instruction
}

func getDefaultAgentDescription(state *iagent.State) string {
	switch state.AgentType {
	case iagent.TypeLoopAgent:
		return "A loop workflow agent"
	case iagent.TypeSequentialAgent:
		return "A sequential workflow agent"
	case iagent.TypeParallelAgent:
		return "A parallel workflow agent"
	case iagent.TypeLLMAgent:
		return "An LLM-based agent"
	default:
		return "A custom agent"
	}
}

func getAgentTypeTag(state *iagent.State) string {
	switch state.AgentType {
	case iagent.TypeLoopAgent:
		return "loop_workflow"
	case iagent.TypeSequentialAgent:
		return "sequential_workflow"
	case iagent.TypeParallelAgent:
		return "parallel_workflow"
	case iagent.TypeLLMAgent:
		return "llm_agent"
	default:
		return "custom_agent"
	}
}

func getAgentSkillName(state *iagent.State) string {
	if state.AgentType == iagent.TypeLLMAgent {
		return "model"
	}
	if isWorkflowAgent(state) {
		return "workflow"
	}
	return "custom"
}

func getInternalState(agent agent.Agent) *iagent.State {
	if agent, ok := agent.(iagent.Agent); ok {
		return iagent.Reveal(agent)
	} else {
		return &iagent.State{AgentType: iagent.TypeCustomAgent}
	}
}

func isWorkflowAgent(state *iagent.State) bool {
	workflowAgents := []iagent.Type{iagent.TypeLoopAgent, iagent.TypeSequentialAgent, iagent.TypeParallelAgent}
	return slices.Contains(workflowAgents, state.AgentType)
}
