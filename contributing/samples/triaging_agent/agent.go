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

package main

import (
	"fmt"
	"log"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

var (
	listUnlabeledIssuesName     = "list_unlabeled_issues"
	changeIssueTypeName         = "change_issue_type"
	addLabelAndOwnerToIssueName = "add_label_and_owner_to_issue"

	labelToOwner = map[string]string{
		"agent engine":  "yeesian",
		"documentation": "polong-lin",
		"services":      "DeanChensj",
		"question":      "",
		"mcp":           "seanzhou1023",
		"tools":         "seanzhou1023",
		"eval":          "ankursharmas",
		"live":          "hangfei",
		"models":        "genquan9",
		"tracing":       "jawoszek",
		"core":          "Jacksunwei",
		"web":           "wyf7107",
		"a2a":           "seanzhou1023",
	}

	labelGuidelines = `
      Label rubric and disambiguation rules:
      - "documentation": Tutorials, README content, reference docs, or samples.
      - "services": Session and memory services, persistence layers, or storage
        integrations.
      - "web": ADK web UI, FastAPI server, dashboards, or browser-based flows.
      - "question": Usage questions without a reproducible problem.
      - "tools": Built-in tools (e.g., SQL utils, code execution) or tool APIs.
      - "mcp": Model Context Protocol features. Apply both "mcp" and "tools".
      - "eval": Evaluation framework, test harnesses, scoring, or datasets.
      - "live": Streaming, bidi, audio, or Gemini Live configuration.
      - "models": Non-Gemini model adapters (LiteLLM, Ollama, OpenAI, etc.).
      - "tracing": Telemetry, observability, structured logs, or spans.
      - "core": Core ADK runtime (Agent definitions, Runner, planners,
        thinking config, CLI commands, GlobalInstructionPlugin, CPU usage, or
        general orchestration). Default to "core" when the topic is about ADK
        behavior and no other label is a better fit.
      - "agent engine": Vertex AI Agent Engine deployment or sandbox topics
        only (e.g., .agent_engine_config.json, ae_ignore, Agent Engine
        sandbox, agent_engine_id). If the issue does not explicitly mention
        Agent Engine concepts, do not use this label—choose "core" instead.
      - "a2a": Agent-to-agent workflows, coordination logic, or A2A protocol.

      When unsure between labels, prefer the most specific match. If a label
      cannot be assigned confidently, do not call the labeling tool.
`
)

func getInstruction(repo, owner, approvalInstruction, labelGuidelines, listUnlabeledIssuesName, changeIssueTypeName string) string {
	return fmt.Sprintf(`
      You are a triaging bot for the GitHub %s repo with the owner %s. You will help get issues, and recommend a label.
      IMPORTANT: %s

      %s

      Here are the rules for labeling:
      - If the user is asking about documentation-related questions, label it with "documentation".
      - If it's about session, memory services, label it with "services".
      - If it's about UI/web, label it with "web".
      - If the user is asking about a question, label it with "question".
      - If it's related to tools, label it with "tools".
      - If it's about agent evaluation, then label it with "eval".
      - If it's about streaming/live, label it with "live".
      - If it's about model support (non-Gemini, like Litellm, Ollama, OpenAI models), label it with "models".
      - If it's about tracing, label it with "tracing".
      - If it's agent orchestration, agent definition, Runner behavior, planners, or performance, label it with "core".
      - Use "agent engine" only when the issue clearly references Vertex AI Agent Engine deployment artifacts (for example .agent_engine_config.json, ae_ignore, agent_engine_id, or Agent Engine sandbox errors).
      - If it's about Model Context Protocol (e.g. MCP tool, MCP toolset, MCP session management etc.), label it with both "mcp" and "tools".
      - If it's about A2A integrations or workflows, label it with "a2a".
      - If you can't find an appropriate labels for the issue, follow the previous instruction that starts with "IMPORTANT:".

      Call the `+"`%s`"+` tool to label the issue, which will also assign the issue to the owner of the label.

      After you label the issue, call the `+"`%s`"+` tool to change the issue type:
      - If the issue is a bug report, change the issue type to "Bug".
      - If the issue is a feature request, change the issue type to "Feature".
      - Otherwise, **do not change the issue type**.

      Response quality requirements:
      - Summarize the issue in your own words without leaving template
        placeholders (never output text like "[fill in later]").
      - Justify the chosen label with a short explanation referencing the issue
        details.
      - Mention the assigned owner when a label maps to one.
      - If no label is applied, clearly state why.

      Present the following in an easy to read format highlighting issue number and your label.
      - the issue summary in a few sentence
      - your label recommendation and justification
      - the owner of the label if you assign the issue to an owner
    `, repo, owner, approvalInstruction, labelGuidelines, listUnlabeledIssuesName, changeIssueTypeName)
}

// listUnlabeledIssues lists the most recent unlabeled issues in the repo.
func listUnlabeledIssues(_ tool.Context, args ListUnlabeledIssuesArgs) (ListUnlabeledIssuesResult, error) {
	searchURL := fmt.Sprintf("%s/search/issues", GitHubBaseURL)
	query := fmt.Sprintf("repo:%s/%s is:open is:issue no:label", Owner, Repo)

	params := map[string]string{
		"q":        query,
		"sort":     "created",
		"order":    "desc",
		"per_page": fmt.Sprintf("%d", args.IssueCount),
		"page":     "1",
	}

	response, err := getRequest(searchURL, params)
	if err != nil {
		return ListUnlabeledIssuesResult{
			Status:  "error",
			Message: fmt.Sprintf("Error: %v", err),
		}, nil
	}

	items, ok := response["items"].([]interface{})
	if !ok {
		return ListUnlabeledIssuesResult{
			Status:  "error",
			Message: "Invalid response format from GitHub API",
		}, nil
	}

	unlabeledIssues := make([]map[string]interface{}, 0)
	for _, item := range items {
		issue, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		labels, _ := issue["labels"].([]interface{})
		if len(labels) == 0 {
			unlabeledIssues = append(unlabeledIssues, issue)
		}
	}

	return ListUnlabeledIssuesResult{
		Status: "success",
		Issues: unlabeledIssues,
	}, nil
}

// addLabelAndOwnerToIssue adds the specified label and owner to the given issue.
func addLabelAndOwnerToIssue(_ tool.Context, args AddLabelAndOwnerArgs) (AddLabelAndOwnerResult, error) {
	log.Printf("Attempting to add label '%s' to issue #%d", args.Label, args.IssueNumber)

	if _, ok := labelToOwner[args.Label]; !ok {
		return AddLabelAndOwnerResult{
			Status:  "error",
			Message: fmt.Sprintf("Error: Label '%s' is not an allowed label. Will not apply.", args.Label),
		}, nil
	}

	labelURL := fmt.Sprintf("%s/repos/%s/%s/issues/%d/labels", GitHubBaseURL, Owner, Repo, args.IssueNumber)
	labelPayload := []string{args.Label}

	_, err := postRequest(labelURL, labelPayload)
	if err != nil {
		return AddLabelAndOwnerResult{
			Status:  "error",
			Message: fmt.Sprintf("Error: %v", err),
		}, nil
	}

	owner := labelToOwner[args.Label]
	if owner == "" {
		return AddLabelAndOwnerResult{
			Status:       "warning",
			Message:      fmt.Sprintf("Label '%s' does not have an owner. Will not assign.", args.Label),
			AppliedLabel: args.Label,
		}, nil
	}

	assigneeURL := fmt.Sprintf("%s/repos/%s/%s/issues/%d/assignees", GitHubBaseURL, Owner, Repo, args.IssueNumber)
	assigneePayload := map[string][]string{"assignees": {owner}}

	_, err = postRequest(assigneeURL, assigneePayload)
	if err != nil {
		return AddLabelAndOwnerResult{
			Status:  "error",
			Message: fmt.Sprintf("Error: %v", err),
		}, nil
	}

	return AddLabelAndOwnerResult{
		Status:        "success",
		AppliedLabel:  args.Label,
		AssignedOwner: owner,
	}, nil
}

// changeIssueType changes the issue type of the given issue.
func changeIssueType(_ tool.Context, args ChangeIssueTypeArgs) (ChangeIssueTypeResult, error) {
	log.Printf("Attempting to change issue type '%s' to issue #%d", args.IssueType, args.IssueNumber)

	// We currently allow for Bug and Feature issue type
	if args.IssueType != "Bug" && args.IssueType != "Feature" {
		return ChangeIssueTypeResult{
			Status:  "error",
			Message: fmt.Sprintf("Error: Invalid issue type '%s'. Only 'Bug' and 'Feature' are supported.", args.IssueType),
		}, nil
	}

	url := fmt.Sprintf("%s/repos/%s/%s/issues/%d", GitHubBaseURL, Owner, Repo, args.IssueNumber)
	payload := map[string]string{"type": args.IssueType}

	_, err := patchRequest(url, payload)
	if err != nil {
		return ChangeIssueTypeResult{
			Status:  "error",
			Message: fmt.Sprintf("Error: %v", err),
		}, nil
	}

	return ChangeIssueTypeResult{
		Status:    "success",
		IssueType: args.IssueType,
	}, nil
}

// NewRootAgent creates the root triaging agent with all the required tools.
func NewRootAgent(m model.LLM) (agent.Agent, error) {
	listUnlabeledIssuesTool, err := functiontool.New(functiontool.Config{
		Name:        listUnlabeledIssuesName,
		Description: "List most recent unlabeled issues in the repo.",
	}, listUnlabeledIssues)
	if err != nil {
		return nil, fmt.Errorf("failed to create %s tool: %w", listUnlabeledIssuesName, err)
	}

	addLabelTool, err := functiontool.New(functiontool.Config{
		Name:        addLabelAndOwnerToIssueName,
		Description: "Add the specified label and owner to the given issue number.",
	}, addLabelAndOwnerToIssue)
	if err != nil {
		return nil, fmt.Errorf("failed to create %s tool: %w", addLabelAndOwnerToIssueName, err)
	}

	changeIssueTypeTool, err := functiontool.New(functiontool.Config{
		Name:        changeIssueTypeName,
		Description: "Change the issue type of the given issue number.",
	}, changeIssueType)
	if err != nil {
		return nil, fmt.Errorf("failed to create %s tool: %w", changeIssueTypeName, err)
	}

	approvalInstruction := "Do not ask for user approval for labeling! If you can't find appropriate labels for the issue, do not label it."
	if IsInteractive {
		approvalInstruction = "Only label them when the user approves the labeling!"
	}

	instruction := getInstruction(Repo, Owner, approvalInstruction, labelGuidelines, listUnlabeledIssuesName, changeIssueTypeName)

	nAgent, err := llmagent.New(llmagent.Config{
		Name:        "adk_triaging_assistant",
		Model:       m,
		Description: "Triage ADK issues.",
		Instruction: instruction,
		Tools: []tool.Tool{
			listUnlabeledIssuesTool,
			addLabelTool,
			changeIssueTypeTool,
		},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create agent: %w", err)
	}

	return nAgent, nil
}
