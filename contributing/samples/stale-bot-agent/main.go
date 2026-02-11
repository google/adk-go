//    Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
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
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
	"google.golang.org/genai"
)

// --- Configuration Constants ---
const (
	AppName = "stale_bot_app"
	UserID  = "stale_bot_user"
)

var (
	rootAgent       agent.Agent
	PROMPT_TEMPLATE string
	geminiModel     = getEnv("GEMINI_MODEL", "gemini-2.5-pro")
)

// processSingleResult holds the return values for processSingleIssue
type processSingleResult struct {
	duration time.Duration
	apiCalls int
}

// ToolResult is used for tool return values
type ToolResult struct {
	Status  string `json:"status"`
	Message string `json:"message,omitempty"`
}

// processSingleIssue processes a single GitHub issue using the AI agent.
func processSingleIssue(ctx context.Context, issueNumber int) processSingleResult {
	startTime := time.Now()
	startAPICalls := GetAPICallCount()
	log.Printf("Processing Issue #%d...", issueNumber)
	res := processSingleResult{}

	// Error handling block (equivalent to try...except)
	func() {
		defer func() {
			if r := recover(); r != nil {
				log.Printf("Error processing issue #%d: %v", issueNumber, r)
			}
		}()

		// Initialize Session Service (InMemory)
		sessionService := session.InMemoryService()

		// Create Session
		sess, err := sessionService.Create(ctx, &session.CreateRequest{
			AppName: AppName,
			UserID:  UserID,
		})
		if err != nil {
			log.Printf("Error creating session for issue #%d: %v", issueNumber, err)
			return
		}

		// Create runner
		r, err := runner.New(runner.Config{
			AppName:         AppName,
			Agent:           rootAgent,
			SessionService:  sessionService,
			ArtifactService: artifact.InMemoryService(),
			MemoryService:   memory.InMemoryService(),
		})
		if err != nil {
			log.Fatalf("Failed to create runner: %v", err)
		}

		// Construct Prompt
		promptText := fmt.Sprintf("Audit Issue #%d.", issueNumber)
		promptMessage := &genai.Content{
			Role: "user",
			Parts: []*genai.Part{
				{Text: promptText},
			},
		}

		eventStream := r.Run(ctx, UserID, sess.Session.ID(), promptMessage, agent.RunConfig{})
		for event := range eventStream {
			if event.Content != nil && len(event.Content.Parts) > 0 {
				part := event.Content.Parts[0]
				if part.Text != "" {
					text := part.Text
					cleanText := strings.ReplaceAll(text, "\n", " ")
					if len(cleanText) > 150 {
						cleanText = cleanText[:150]
					}
					log.Printf("#%d Decision: %s...", issueNumber, cleanText)
				}
			}
		}
	}()

	res.duration = time.Since(startTime)
	endAPICalls := GetAPICallCount()
	res.apiCalls = endAPICalls - startAPICalls
	log.Printf("Issue #%d finished in %.2fs with ~%d API calls.", issueNumber, res.duration.Seconds(), res.apiCalls)
	return res
}

func loadPromptTemplate(filename string) (string, error) {
	_, currentFile, _, ok := runtime.Caller(0)
	if !ok {
		return "", fmt.Errorf("cannot determine caller location")
	}
	baseDir := filepath.Dir(currentFile)
	path := filepath.Join(baseDir, filename)
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func setupTools() []tool.Tool {
	t1, _ := functiontool.New(functiontool.Config{
		Name:        "add_label_to_issue",
		Description: "Adds a specific label to a GitHub issue.",
	}, addLabelToIssue)

	t2, _ := functiontool.New(functiontool.Config{
		Name:        "remove_label_from_issue",
		Description: "Remove a specific label from a GitHub issue.",
	}, removeLabelFromIssue)

	t3, _ := functiontool.New(functiontool.Config{
		Name:        "add_stale_label_and_comment",
		Description: "Marks the issue as stale with a comment and label.",
	}, addStaleLabelAndComment)

	t4, _ := functiontool.New(functiontool.Config{
		Name:        "alert_maintainer_of_edit",
		Description: "Post a comment alerting maintainers of a silent edit.",
	}, alertMaintainerOfEdit)

	t5, _ := functiontool.New(functiontool.Config{
		Name:        "close_as_stale",
		Description: "Close the issue as completed/stale.",
	}, closeAsStale)

	t6, _ := functiontool.New(functiontool.Config{
		Name:        "get_issue_state",
		Description: "Fetch and analyze the current state/history of the issue.",
	}, getIssueState)

	return []tool.Tool{t1, t2, t3, t4, t5, t6}
}

func formatPrompt(template string, values map[string]string) string {
	result := template
	for k, v := range values {
		result = strings.ReplaceAll(result, "{"+k+"}", v)
	}
	return result
}

func main() {
	startTotalTime := time.Now()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	InitConfig()

	var err error
	PROMPT_TEMPLATE, err = loadPromptTemplate("PROMPT_INSTRUCTION.txt")
	if err != nil {
		log.Fatalf("Failed to load PROMPT_INSTRUCTION.txt: %v", err)
	}

	log.Println("PROMPT_TEMPLATE loaded successfully.")
	log.Printf("--- Starting Stale Bot for %s/%s ---", Owner, Repo)
	log.Printf("Concurrency level set to %d", ConcurrencyLimit)

	model, err := gemini.NewModel(ctx, geminiModel, &genai.ClientConfig{APIKey: os.Getenv("GOOGLE_API_KEY")})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	instruction := formatPrompt(PROMPT_TEMPLATE, map[string]string{
		"OWNER":                       Owner,
		"REPO":                        Repo,
		"STALE_LABEL_NAME":            StaleLabelName,
		"REQUEST_CLARIFICATION_LABEL": RequestClarificationLabel,
		"stale_threshold_days":        fmt.Sprintf("%g", float64(STALE_HOURS_THRESHOLD)/24.0),
		"close_threshold_days":        fmt.Sprintf("%g", float64(CLOSE_HOURS_AFTER_STALE_THRESHOLD)/24.0),
	})

	toolList := setupTools()
	rootAgent, err = llmagent.New(llmagent.Config{
		Name:        "adk_repository_auditor_agent",
		Description: "Audits open issues.",
		Model:       model,
		Instruction: instruction,
		Tools:       toolList,
	})
	if err != nil {
		log.Fatalf("failed to create root agent: %v", err)
	}

	ResetAPICallCount()
	filterDays := STALE_HOURS_THRESHOLD / 24.0

	allIssues, err := GetOldOpenIssueNumbers(Owner, Repo, &filterDays)
	if err != nil {
		log.Fatalf("Failed to fetch issue list: %v", err)
	}

	totalCount := len(allIssues)
	searchAPICalls := GetAPICallCount()
	if totalCount == 0 {
		log.Println("No issues matched the criteria. Run finished.")
		return
	}

	log.Printf("Found %d issues to process. (Initial search used %d API calls).", totalCount, searchAPICalls)

	var totalProcessingTime time.Duration
	var totalIssueAPICalls int
	processedCount := 0

	for i := 0; i < totalCount; i += ConcurrencyLimit {
		end := i + ConcurrencyLimit
		if end > totalCount {
			end = totalCount
		}
		chunk := allIssues[i:end]
		currentChunkNum := (i / ConcurrencyLimit) + 1
		log.Printf("--- Starting chunk %d: Processing issues %v ---", currentChunkNum, chunk)

		var wg sync.WaitGroup
		resultsChan := make(chan processSingleResult, len(chunk))

		for _, issueNum := range chunk {
			wg.Add(1)
			go func(num int) {
				defer wg.Done()
				res := processSingleIssue(ctx, num)
				resultsChan <- res
			}(issueNum)
		}

		wg.Wait()
		close(resultsChan)

		for res := range resultsChan {
			totalProcessingTime += res.duration
			totalIssueAPICalls += res.apiCalls
		}

		processedCount += len(chunk)
		log.Printf("--- Finished chunk %d. Progress: %d/%d ---", currentChunkNum, processedCount, totalCount)

		if end < totalCount {
			time.Sleep(time.Duration(SleepBetweenChunks * float64(time.Second)))
		}
	}

	totalAPICallsForRun := searchAPICalls + totalIssueAPICalls
	avgTimePerIssue := 0.0
	if totalCount > 0 {
		avgTimePerIssue = totalProcessingTime.Seconds() / float64(totalCount)
	}

	log.Println("--- Stale Agent Run Finished ---")
	log.Printf("Successfully processed %d issues.", processedCount)
	log.Printf("Total API calls made this run: %d", totalAPICallsForRun)
	log.Printf("Average processing time per issue: %.2f seconds.", avgTimePerIssue)

	duration := time.Since(startTotalTime)
	log.Printf("Full audit finished in %.2f minutes.", duration.Minutes())
}
