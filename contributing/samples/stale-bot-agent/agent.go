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
	"fmt"
	"log"
	"math"
	"sort"
	"strings"
	"time"

	"google.golang.org/adk/tool"
)

var maintainersCache []string

var BOT_ALERT_SIGNATURE = "**Notification:** The author has updated the issue description"

var BOT_NAME = "adk-bot"

type TimelineEvent struct {
	Type  string    `json:"type"`
	Actor string    `json:"actor"`
	Time  time.Time `json:"time"`
	Data  any       `json:"data"`
}

type IssueState struct {
	LastActionRole   string    `json:"last_action_role"`
	LastActivityTime time.Time `json:"last_activity_time"`
	LastActionType   string    `json:"last_action_type"`
	LastCommentText  *string   `json:"last_comment_text"`
	LastActorName    string    `json:"last_actor_name"`
}

// Struct for tools that only need an Issue Number
// Used by: addStaleLabelAndComment, alertMaintainerOfEdit, closeAsStale, getIssueState
type IssueTargetArgs struct {
	IssueNumber int `json:"issue_number" description:"The number of the GitHub issue to act upon"`
}

// Struct for tools that need an Issue Number AND a Label Name
// Used by: addLabelToIssue, removeLabelFromIssue
type LabelTargetArgs struct {
	IssueNumber int    `json:"issue_number" description:"The number of the GitHub issue"`
	LabelName   string `json:"label_name" description:"The specific name of the label"`
}

func getCachedMaintainers() ([]string, error) {
	// if _MAINTAINERS_CACHE is not None: return it
	if maintainersCache != nil {
		return maintainersCache, nil
	}

	log.Println("Initializing Maintainers Cache...")

	url := fmt.Sprintf("%s/repos/%s/%s/collaborators", GitHubBaseURL, Owner, Repo)
	params := map[string]interface{}{
		"permission": "push",
	}

	// Uses your util-layer retry + backoff logic
	data, err := GetRequest(url, params)
	if err != nil {
		log.Printf("ERROR: Failed to verify repository maintainers. Error: %v", err)
		return nil, fmt.Errorf("maintainer verification failed: %w", err)
	}

	rawList, ok := data.([]interface{})
	if !ok {
		log.Printf(
			"Invalid API response format: Expected list, got %T",
			data,
		)
		return nil, fmt.Errorf("github API returned non-list data")
	}

	maintainers := make([]string, 0, len(rawList))

	for _, item := range rawList {
		obj, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		if login, ok := obj["login"].(string); ok {
			maintainers = append(maintainers, login)
		}
	}

	maintainersCache = maintainers
	log.Printf("Cached %d maintainers.", len(maintainersCache))

	return maintainersCache, nil
}

func FetchGraphQLData(itemNumber int) (map[string]any, error) {
	query := `
query($owner: String!, $name: String!, $number: Int!, $commentLimit: Int!, $timelineLimit: Int!, $editLimit: Int!) {
  repository(owner: $owner, name: $name) {
    issue(number: $number) {
      author { login }
      createdAt
      labels(first: 20) { nodes { name } }

      comments(last: $commentLimit) {
        nodes {
          author { login }
          body
          createdAt
          lastEditedAt
        }
      }

      userContentEdits(last: $editLimit) {
        nodes {
          editor { login }
          editedAt
        }
      }

      timelineItems(
        itemTypes: [LABELED_EVENT, RENAMED_TITLE_EVENT, REOPENED_EVENT],
        last: $timelineLimit
      ) {
        nodes {
          __typename
          ... on LabeledEvent {
            createdAt
            actor { login }
            label { name }
          }
          ... on RenamedTitleEvent {
            createdAt
            actor { login }
          }
          ... on ReopenedEvent {
            createdAt
            actor { login }
          }
        }
      }
    }
  }
}
`

	variables := map[string]any{
		"owner":         Owner,
		"name":          Repo,
		"number":        itemNumber,
		"commentLimit":  GraphQLCommentLimit,
		"editLimit":     GraphQLEditLimit,
		"timelineLimit": GraphQLTimelineLimit,
	}

	payload := map[string]any{
		"query":     query,
		"variables": variables,
	}

	respAny, err := PostRequest(GitHubBaseURL+"/graphql", payload)
	if err != nil {
		return nil, err
	}

	resp, ok := respAny.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("invalid GraphQL response format")
	}

	if errs, ok := resp["errors"]; ok {
		if errList, ok := errs.([]any); ok && len(errList) > 0 {
			if firstErr, ok := errList[0].(map[string]any); ok {
				return nil, fmt.Errorf("GraphQL Error: %v", firstErr["message"])
			}
		}
		return nil, fmt.Errorf("GraphQL Error: unknown format")
	}

	data := resp["data"].(map[string]any)
	repo := data["repository"].(map[string]any)
	issue := repo["issue"]

	if issue == nil {
		return nil, fmt.Errorf("issue #%d not found", itemNumber)
	}

	return issue.(map[string]any), nil
}

func buildHistoryTimeline(data map[string]any) ([]TimelineEvent, []time.Time, *time.Time) {
	issueAuthor := ""
	if author, ok := data["author"].(map[string]any); ok {
		issueAuthor, _ = author["login"].(string)
	}

	var history []TimelineEvent
	var labelEvents []time.Time
	var lastBotAlertTime *time.Time

	parseTime := func(val any) time.Time {
		s, _ := val.(string)
		t, _ := time.Parse(time.RFC3339, s)
		return t
	}

	isBot := func(actor string) bool {
		return actor == "" || strings.HasSuffix(actor, "[bot]") || actor == BOT_NAME
	}

	// 1. Baseline: Issue Creation
	createdAt := parseTime(data["createdAt"])
	history = append(history, TimelineEvent{
		Type:  "created",
		Actor: issueAuthor,
		Time:  createdAt,
		Data:  nil,
	})

	// 2. Process Comments
	if comments, ok := data["comments"].(map[string]any); ok {
		if nodes, ok := comments["nodes"].([]any); ok {
			for _, node := range nodes {
				c, ok := node.(map[string]any)
				if !ok || c == nil {
					continue
				}

				actor := ""
				if a, ok := c["author"].(map[string]any); ok {
					actor, _ = a["login"].(string)
				}

				cBody, _ := c["body"].(string)
				cTime := parseTime(c["createdAt"])

				// Track bot alerts for spam prevention
				if strings.Contains(cBody, BOT_ALERT_SIGNATURE) {
					if lastBotAlertTime == nil || cTime.After(*lastBotAlertTime) {
						tempTime := cTime
						lastBotAlertTime = &tempTime
					}
					continue
				}

				if !isBot(actor) {
					// Use edit time if available, otherwise creation time
					actualTime := cTime
					if eTimeStr, ok := c["lastEditedAt"].(string); ok && eTimeStr != "" {
						actualTime = parseTime(eTimeStr)
					}

					history = append(history, TimelineEvent{
						Type:  "commented",
						Actor: actor,
						Time:  actualTime,
						Data:  cBody,
					})
				}
			}
		}
	}

	// 3. Process Body Edits ("Ghost Edits")
	if edits, ok := data["userContentEdits"].(map[string]any); ok {
		if nodes, ok := edits["nodes"].([]any); ok {
			for _, node := range nodes {
				e, ok := node.(map[string]any)
				if !ok || e == nil {
					continue
				}

				actor := ""
				if ed, ok := e["editor"].(map[string]any); ok {
					actor, _ = ed["login"].(string)
				}

				if !isBot(actor) {
					history = append(history, TimelineEvent{
						Type:  "edited_description",
						Actor: actor,
						Time:  parseTime(e["editedAt"]),
						Data:  nil,
					})
				}
			}
		}
	}

	// 4. Process Timeline Events
	if timeline, ok := data["timelineItems"].(map[string]any); ok {
		if nodes, ok := timeline["nodes"].([]any); ok {
			for _, node := range nodes {
				t, ok := node.(map[string]any)
				if !ok || t == nil {
					continue
				}

				etype, _ := t["__typename"].(string)
				actor := ""
				if a, ok := t["actor"].(map[string]any); ok {
					actor, _ = a["login"].(string)
				}
				timeVal := parseTime(t["createdAt"])

				if etype == "LabeledEvent" {
					labelName := ""
					if lbl, ok := t["label"].(map[string]any); ok {
						labelName, _ = lbl["name"].(string)
					}
					if labelName == STALE_LABEL_NAME {
						labelEvents = append(labelEvents, timeVal)
					}
					continue
				}

				if !isBot(actor) {
					prettyType := "reopened"
					if etype == "RenamedTitleEvent" {
						prettyType = "renamed_title"
					}
					history = append(history, TimelineEvent{
						Type:  prettyType,
						Actor: actor,
						Time:  timeVal,
						Data:  nil,
					})
				}
			}
		}
	}

	// Sort chronologically
	sort.Slice(history, func(i, j int) bool {
		return history[i].Time.Before(history[j].Time)
	})

	return history, labelEvents, lastBotAlertTime
}

func replayHistoryToFindState(history []TimelineEvent, maintainers []string, issueAuthor string) IssueState {
	// Initialize defaults (Baseline: Issue Creation)
	// We assume history is never empty because buildHistoryTimeline adds the "created" event.
	lastActionRole := "author"
	lastActivityTime := history[0].Time
	lastActionType := "created"
	var lastCommentText *string = nil
	lastActorName := issueAuthor

	for _, event := range history {
		actor := event.Actor
		etype := event.Type

		// Determine Role
		role := "other_user"
		if actor == issueAuthor {
			role = "author"
		} else if isMaintainer(actor, maintainers) {
			role = "maintainer"
		}

		// Update State
		lastActionRole = role
		lastActivityTime = event.Time
		lastActionType = etype
		lastActorName = actor

		// Handle Comment Text Logic
		if etype == "commented" {
			// Convert any/interface{} Data to string
			if text, ok := event.Data.(string); ok {
				lastCommentText = &text
			}
		} else {
			// Resets on other events like labels/edits
			lastCommentText = nil
		}
	}

	return IssueState{
		LastActionRole:   lastActionRole,
		LastActivityTime: lastActivityTime,
		LastActionType:   lastActionType,
		LastCommentText:  lastCommentText,
		LastActorName:    lastActorName,
	}
}

// Helper to check if actor is in maintainers list
func isMaintainer(actor string, maintainers map[string]bool) bool {
	_, ok := maintainers[actor]
	return ok
}

func formatDays(hours float64) string {
	days := hours / 24.0
	// If it's a whole number (e.g., 7.0), return as integer string "7"
	if math.Mod(days, 1.0) == 0 {
		return fmt.Sprintf("%d", int(days))
	}
	// Otherwise return with 1 decimal place (e.g., "0.5")
	return fmt.Sprintf("%.1f", days)
}

func errorResponse(msg string) map[string]any {
	return map[string]any{
		"status": "error",
		"error":  msg,
	}
}

func addLabelToIssue(ctx tool.Context, args LabelTargetArgs) (ToolResult, error) {
	url := fmt.Sprintf(
		"%s/repos/%s/%s/issues/%d/labels",
		GitHubBaseURL,
		Owner,
		Repo,
		args.IssueNumber,
	)

	payload := []string{args.LabelName}

	_, err := PostRequest(url, payload)
	if err != nil {
		return ToolResult{
			Status:  "failure",
			Message: err.Error(),
		}, err
	}

	return ToolResult{
		Status: "success",
	}, nil
}

func removeLabelFromIssue(ctx tool.Context, args LabelTargetArgs) (ToolResult, error) {
	url := fmt.Sprintf(
		"%s/repos/%s/%s/issues/%d/labels/%s",
		GitHubBaseURL,
		Owner,
		Repo,
		args.IssueNumber,
		args.LabelName,
	)

	_, err := DeleteRequest(url)
	if err != nil {
		return ToolResult{
			Status:  "failure",
			Message: fmt.Sprintf("error removing label: %v", err),
		}, err
	}

	return ToolResult{
		Status: "success",
	}, nil
}

func addStaleLabelAndComment(ctx tool.Context, args IssueTargetArgs) (ToolResult, error) {
	staleDaysStr := formatDays(STALE_HOURS_THRESHOLD)
	closeDaysStr := formatDays(CLOSE_HOURS_AFTER_STALE_THRESHOLD)

	comment := fmt.Sprintf(
		"This issue has been automatically marked as stale because it has not"+
			" had recent activity for %s days after a maintainer"+
			" requested clarification. It will be closed if no further activity"+
			" occurs within %s days.",
		staleDaysStr, closeDaysStr,
	)

	// 1. Post comment
	commentURL := fmt.Sprintf(
		"%s/repos/%s/%s/issues/%d/comments",
		GitHubBaseURL, Owner, Repo, args.IssueNumber,
	)

	if _, err := PostRequest(commentURL, map[string]string{"body": comment}); err != nil {
		return ToolResult{
			Status:  "failure",
			Message: fmt.Sprintf("error posting stale comment: %v", err),
		}, err
	}

	// 2. Add label
	labelURL := fmt.Sprintf(
		"%s/repos/%s/%s/issues/%d/labels",
		GitHubBaseURL, Owner, Repo, args.IssueNumber,
	)

	if _, err := PostRequest(labelURL, []string{STALE_LABEL_NAME}); err != nil {
		return ToolResult{
			Status:  "failure",
			Message: fmt.Sprintf("error adding stale label: %v", err),
		}, err
	}

	return ToolResult{
		Status: "success",
	}, nil
}

func alertMaintainerOfEdit(ctx tool.Context, args IssueTargetArgs) (ToolResult, error) {
	comment := fmt.Sprintf("%s. Maintainers, please review.", BOT_ALERT_SIGNATURE)

	url := fmt.Sprintf(
		"%s/repos/%s/%s/issues/%d/comments",
		GitHubBaseURL, Owner, Repo, args.IssueNumber,
	)

	if _, err := PostRequest(url, map[string]string{"body": comment}); err != nil {
		return ToolResult{
			Status:  "failure",
			Message: fmt.Sprintf("error posting alert: %v", err),
		}, err
	}

	return ToolResult{
		Status: "success",
	}, nil
}

func closeAsStale(ctx tool.Context, args IssueTargetArgs) (ToolResult, error) {
	daysStr := formatDays(CLOSE_HOURS_AFTER_STALE_THRESHOLD)

	comment := fmt.Sprintf(
		"This has been automatically closed because it has been marked as stale"+
			" for over %s days.",
		daysStr,
	)

	// 1. Post comment
	commentURL := fmt.Sprintf(
		"%s/repos/%s/%s/issues/%d/comments",
		GitHubBaseURL, Owner, Repo, args.IssueNumber,
	)

	if _, err := PostRequest(commentURL, map[string]string{"body": comment}); err != nil {
		return ToolResult{
			Status:  "failure",
			Message: fmt.Sprintf("error posting close comment: %v", err),
		}, err
	}

	// 2. Close issue
	issueURL := fmt.Sprintf(
		"%s/repos/%s/%s/issues/%d",
		GitHubBaseURL, Owner, Repo, args.IssueNumber,
	)

	if _, err := PatchRequest(issueURL, map[string]string{"state": "closed"}); err != nil {
		return ToolResult{
			Status:  "failure",
			Message: fmt.Sprintf("error closing issue: %v", err),
		}, err
	}

	return ToolResult{
		Status: "success",
	}, nil
}

// getIssueState orchestrates the fetching and analysis of an issue.
func getIssueState(ctx tool.Context, args IssueTargetArgs) (map[string]any, error) {
	itemNumber := args.IssueNumber

	maintainers, err := getCachedMaintainers()
	if err != nil {
		return errorResponse(fmt.Sprintf("error getting cached maintainers: %v", err)), nil
	}

	rawData, err := FetchGraphQLData(itemNumber)
	if err != nil {
		return errorResponse(fmt.Sprintf("network error: %v", err)), nil
	}

	// Extract author
	issueAuthor := ""
	if author, ok := rawData["author"].(map[string]any); ok {
		issueAuthor, _ = author["login"].(string)
	}

	// Extract labels
	var labelsList []string
	if labels, ok := rawData["labels"].(map[string]any); ok {
		if nodes, ok := labels["nodes"].([]any); ok {
			for _, n := range nodes {
				if node, ok := n.(map[string]any); ok {
					if name, ok := node["name"].(string); ok {
						labelsList = append(labelsList, name)
					}
				}
			}
		}
	}

	history, labelEvents, lastBotAlertTime := buildHistoryTimeline(rawData)
	state := replayHistoryToFindState(history, maintainers, issueAuthor)

	now := time.Now().UTC()
	daysSinceActivity := now.Sub(state.LastActivityTime).Hours() / 24.0

	isStale := false
	for _, l := range labelsList {
		if l == STALE_LABEL_NAME {
			isStale = true
			break
		}
	}

	daysSinceStaleLabel := 0.0
	if isStale && len(labelEvents) > 0 {
		latest := labelEvents[0]
		for _, t := range labelEvents {
			if t.After(latest) {
				latest = t
			}
		}
		daysSinceStaleLabel = now.Sub(latest).Hours() / 24.0
	}

	maintainerAlertNeeded := false
	if (state.LastActionRole == "author" || state.LastActionRole == "other_user") &&
		state.LastActionType == "edited_description" {

		if lastBotAlertTime == nil || lastBotAlertTime.Before(state.LastActivityTime) {
			maintainerAlertNeeded = true
		}
	}

	return map[string]any{
		"status":                  "success",
		"last_action_role":        state.LastActionRole,
		"last_action_type":        state.LastActionType,
		"last_actor_name":         state.LastActorName,
		"maintainer_alert_needed": maintainerAlertNeeded,
		"is_stale":                isStale,
		"days_since_activity":     daysSinceActivity,
		"days_since_stale_label":  daysSinceStaleLabel,
		"last_comment_text":       state.LastCommentText,
		"current_labels":          labelsList,
		"stale_threshold_days":    STALE_HOURS_THRESHOLD / 24.0,
		"close_threshold_days":    CLOSE_HOURS_AFTER_STALE_THRESHOLD / 24.0,
		"maintainers":             maintainers,
		"issue_author":            issueAuthor,
	}, nil
}
