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
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

const (
	appName = "adk_triage_app"
	userID  = "adk_triage_user"
)

var geminiModel = getEnvOrDefault("GEMINI_MODEL", "gemini-2.5-pro")

func fetchSpecificIssueDetails(issueNumber int) (map[string]interface{}, error) {
	url := fmt.Sprintf("%s/repos/%s/%s/issues/%d", GitHubBaseURL, Owner, Repo, issueNumber)
	log.Printf("Fetching details for specific issue: %s", url)

	issueData, err := getRequest(url, nil)
	if err != nil {
		return nil, fmt.Errorf("error fetching issue #%d: %w", issueNumber, err)
	}

	if labels, ok := issueData["labels"].([]any); ok && len(labels) > 0 {
		log.Printf("Issue #%d is already labelled. Skipping.", issueNumber)
		return nil, nil
	}

	log.Printf("Issue #%d is unlabelled. Proceeding.", issueNumber)
	return map[string]interface{}{
		"number": issueData["number"],
		"title":  issueData["title"],
		"body":   issueData["body"],
	}, nil
}

func callAgent(ctx context.Context, r *runner.Runner, userID, sessionID, prompt string) (string, error) {
	content := genai.NewContentFromText(prompt, genai.RoleUser)

	finalResponseText := ""
	for event, err := range r.Run(ctx, userID, sessionID, content, agent.RunConfig{SaveInputBlobsAsArtifacts: false}) {
		if err != nil {
			return "", fmt.Errorf("error during agent run: %w", err)
		}
		if event == nil {
			continue
		}

		if event.Content != nil && len(event.Content.Parts) > 0 {
			if textPart := event.Content.Parts[0].Text; textPart != "" {
				log.Printf("** %s (ADK): %s", event.Author, textPart)
				if event.Author == "adk_triaging_assistant" {
					finalResponseText += textPart
				}
			}
		}
	}

	return finalResponseText, nil
}

func main() {
	startTime := time.Now()
	log.Printf("Start triaging %s/%s issues at %s", Owner, Repo, startTime.Format("2006-01-02 15:04:05"))
	log.Println(strings.Repeat("-", 80))

	ctx := context.Background()

	// Create model
	model, err := gemini.NewModel(ctx, geminiModel, &genai.ClientConfig{APIKey: os.Getenv("GOOGLE_API_KEY")})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	// Create root agent
	rootAgent, err := NewRootAgent(model)
	if err != nil {
		log.Fatalf("Failed to create root agent: %v", err)
	}

	// Create session
	sessionService := session.InMemoryService()
	sessionResp, err := sessionService.Create(ctx, &session.CreateRequest{
		UserID:  userID,
		AppName: appName,
	})
	if err != nil {
		log.Fatalf("Failed to create session: %v", err)
	}
	sessionID := sessionResp.Session.ID()

	// Create runner
	r, err := runner.New(runner.Config{
		AppName:         appName,
		Agent:           rootAgent,
		SessionService:  sessionService,
		ArtifactService: artifact.InMemoryService(),
		MemoryService:   memory.InMemoryService(),
	})
	if err != nil {
		log.Fatalf("Failed to create runner: %v", err)
	}

	var prompt string
	if EventName == "issues" && IssueNumber != "" {
		log.Printf("EVENT: Processing specific issue due to '%s' event.", EventName)
		issueNumber := ParseNumberString(IssueNumber, 0)
		if issueNumber == 0 {
			log.Printf("Error: Invalid issue number received: %s.", IssueNumber)
			return
		}

		specificIssue, err := fetchSpecificIssueDetails(issueNumber)
		if err != nil {
			log.Printf("Error fetching issue #%d: %v. Skipping agent interaction.", issueNumber, err)
			return
		}
		if specificIssue == nil {
			log.Printf("No unlabelled issue details found for #%d or an error occurred. Skipping agent interaction.", issueNumber)
			return
		}

		issueTitle := IssueTitle
		if issueTitle == "" {
			if title, ok := specificIssue["title"].(string); ok {
				issueTitle = title
			}
		}

		issueBody := IssueBody
		if issueBody == "" {
			if body, ok := specificIssue["body"].(string); ok {
				issueBody = body
			}
		}

		prompt = fmt.Sprintf(
			"A new GitHub issue #%d has been opened or reopened. Title: \"%s\"\nBody: \"%s\"\n\nBased on the rules, recommend an appropriate label and its justification. Then, use the 'add_label_and_owner_to_issue' tool to apply the label directly to this issue. Only label it, do not process any other issues.",
			issueNumber, issueTitle, issueBody,
		)
	} else {
		log.Printf("EVENT: Processing batch of issues (event: %s).", EventName)
		issueCount := ParseNumberString(IssueCountToProcess, 3)
		prompt = fmt.Sprintf("Please triage the most recent %d issues.", issueCount)
	}

	response, err := callAgent(ctx, r, userID, sessionID, prompt)
	if err != nil {
		log.Fatalf("Failed to call agent: %v", err)
	}

	log.Printf("<<<< Agent Final Output: %s\n", response)

	log.Println(strings.Repeat("-", 80))
	endTime := time.Now()
	log.Printf("Triaging finished at %s", endTime.Format("2006-01-02 15:04:05"))
	log.Printf("Total script execution time: %.2f seconds", endTime.Sub(startTime).Seconds())
}
