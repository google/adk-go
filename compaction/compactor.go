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

package compaction

import (
	"context"
	"fmt"
	"sort"

	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

// Compactor handles the selection and summarization of events for session history compaction.
// It implements the sliding window algorithm defined in ADR-010.
type Compactor struct {
	config Config
	llm    model.LLM
}

// New creates a new Compactor with the given configuration and LLM.
func New(config Config, llm model.LLM) (*Compactor, error) {
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid compaction config: %w", err)
	}

	if llm == nil {
		return nil, fmt.Errorf("LLM is required for compaction")
	}

	return &Compactor{
		config: config,
		llm:    llm,
	}, nil
}

// MaybeCompact checks if compaction should be triggered and performs it if needed.
// It returns a non-nil compaction event if compaction occurred, nil otherwise.
// Returns error if compaction fails.
func (c *Compactor) MaybeCompact(ctx context.Context, sess session.Session, newInvocationID string) (*session.Event, error) {
	if !c.config.Enabled {
		return nil, nil
	}

	// Get all events from the session
	events := sess.Events()
	allEvents := make([]*session.Event, 0, events.Len())
	for i := 0; i < events.Len(); i++ {
		allEvents = append(allEvents, events.At(i))
	}

	// Check if we should trigger compaction
	shouldCompact, lastCompactionTime := c.shouldCompact(allEvents)
	if !shouldCompact {
		return nil, nil
	}

	// Select events to compact
	eventsToCompact, startTime, endTime := c.selectEventsToCompact(allEvents, lastCompactionTime)
	if len(eventsToCompact) == 0 {
		return nil, nil
	}

	// Summarize the selected events
	compaction, err := c.summarizeEvents(ctx, eventsToCompact, startTime, endTime)
	if err != nil {
		return nil, fmt.Errorf("failed to summarize events: %w", err)
	}

	// Create the compaction event
	compactionEvent := c.createCompactionEvent(newInvocationID, compaction)

	return compactionEvent, nil
}

// shouldCompact checks if compaction should be triggered based on the current events.
// It returns true if the number of non-compaction events since the last compaction
// exceeds the CompactionInterval threshold.
// It also returns the timestamp of the last compaction (0 if none exists).
func (c *Compactor) shouldCompact(allEvents []*session.Event) (bool, float64) {
	lastCompactionTime := 0.0
	invocationsSinceCompaction := make(map[string]bool)

	// Iterate through events to find last compaction and count new invocations
	for _, event := range allEvents {
		// Check if this is a compaction event
		if event.Actions.Compaction != nil {
			lastCompactionTime = event.Actions.Compaction.EndTimestamp
		}

		// Count unique invocations that occurred after last compaction
		if event.Timestamp.Unix() > int64(lastCompactionTime) && event.Actions.Compaction == nil {
			invocationsSinceCompaction[event.InvocationID] = true
		}
	}

	// Trigger compaction if we have enough new invocations
	return len(invocationsSinceCompaction) >= c.config.CompactionInterval, lastCompactionTime
}

// selectEventsToCompact implements the sliding window algorithm to select events
// that should be compacted. It ensures an overlap of OverlapSize invocations with
// the most recent events to maintain context continuity.
//
// Returns:
// - eventsToCompact: the events to summarize
// - startTime: timestamp of the first event to compact
// - endTime: timestamp of the last event to compact
func (c *Compactor) selectEventsToCompact(allEvents []*session.Event, lastCompactionTime float64) ([]*session.Event, float64, float64) {
	if len(allEvents) == 0 {
		return nil, 0, 0
	}

	// Collect unique invocation IDs and their corresponding events, filtering non-compaction events
	invocationMap := make(map[string][]*session.Event)
	invocationOrder := []string{} // To maintain order

	for _, event := range allEvents {
		// Skip compaction events themselves
		if event.Actions.Compaction != nil {
			continue
		}

		// Only consider events after last compaction
		if event.Timestamp.Unix() <= int64(lastCompactionTime) {
			continue
		}

		if _, exists := invocationMap[event.InvocationID]; !exists {
			invocationOrder = append(invocationOrder, event.InvocationID)
		}
		invocationMap[event.InvocationID] = append(invocationMap[event.InvocationID], event)
	}

	// If we don't have enough invocations to compact, return nothing
	if len(invocationOrder) < c.config.CompactionInterval {
		return nil, 0, 0
	}

	// Calculate which invocations to exclude (keep the most recent OverlapSize invocations)
	numToCompact := len(invocationOrder) - c.config.OverlapSize
	invocationsToCompact := invocationOrder[:numToCompact]

	// Gather all events from the invocations to compact
	eventsToCompact := make([]*session.Event, 0)

	for _, invID := range invocationsToCompact {
		invEvents := invocationMap[invID]
		for _, evt := range invEvents {
			eventsToCompact = append(eventsToCompact, evt)
		}
	}

	// If no events to compact, return early
	if len(eventsToCompact) == 0 {
		return nil, 0, 0
	}

	// Sort by timestamp to ensure proper ordering before determining time range
	sort.Slice(eventsToCompact, func(i, j int) bool {
		return eventsToCompact[i].Timestamp.Before(eventsToCompact[j].Timestamp)
	})

	// Calculate time range from sorted events
	startTime := float64(eventsToCompact[0].Timestamp.Unix())
	endTime := float64(eventsToCompact[len(eventsToCompact)-1].Timestamp.Unix())

	return eventsToCompact, startTime, endTime
}

// summarizeEvents calls the LLM to generate a summary of the provided events.
// Returns an EventCompaction containing the summary and time range information.
func (c *Compactor) summarizeEvents(ctx context.Context, events []*session.Event, startTime, endTime float64) (*session.EventCompaction, error) {
	// Format events for LLM input
	contents := c.formatEventsForSummarization(events)

	// Prepare LLM request with system prompt as first content
	contents = append([]*genai.Content{
		{
			Role: "user",
			Parts: []*genai.Part{
				{
					Text: c.getSystemPrompt(),
				},
			},
		},
	}, contents...)

	req := &model.LLMRequest{
		Model:    c.config.Model,
		Contents: contents,
		Config:   &genai.GenerateContentConfig{},
	}

	// Call LLM to generate summary
	var summaryContent *genai.Content
	for resp, err := range c.llm.GenerateContent(ctx, req, false) {
		if err != nil {
			return nil, fmt.Errorf("LLM generation failed: %w", err)
		}
		if resp != nil && resp.Content != nil {
			summaryContent = resp.Content
			break // Take the first response
		}
	}

	if summaryContent == nil {
		return nil, fmt.Errorf("LLM generated no content")
	}

	return &session.EventCompaction{
		StartTimestamp:   startTime,
		EndTimestamp:     endTime,
		CompactedContent: summaryContent,
	}, nil
}

// formatEventsForSummarization converts events into LLM-friendly content format.
// It creates a conversation history that the LLM can summarize.
func (c *Compactor) formatEventsForSummarization(events []*session.Event) []*genai.Content {
	if len(events) == 0 {
		return nil
	}

	// Group events by invocation and author for coherent conversation
	invocationMessages := make(map[string][]*genai.Content)
	invocationOrder := []string{}

	for _, event := range events {
		if _, exists := invocationMessages[event.InvocationID]; !exists {
			invocationOrder = append(invocationOrder, event.InvocationID)
		}

		// Convert event to content
		if event.LLMResponse.Content != nil {
			// Map event author to valid LLM role (user or model)
			role := "model" // Default to model for agent responses
			if event.Author == "user" {
				role = "user"
			}

			content := &genai.Content{
				Role:  role,
				Parts: append([]*genai.Part{}, event.LLMResponse.Content.Parts...),
			}
			invocationMessages[event.InvocationID] = append(invocationMessages[event.InvocationID], content)
		}
	}

	// Flatten into single content sequence for LLM context
	var contents []*genai.Content
	for _, invID := range invocationOrder {
		contents = append(contents, invocationMessages[invID]...)
	}

	return contents
}

// getSystemPrompt returns the system prompt for summarization.
// If a custom prompt is configured, it uses that. Otherwise returns a default.
func (c *Compactor) getSystemPrompt() string {
	if c.config.SystemPrompt != "" {
		return c.config.SystemPrompt
	}

	return `You are summarizing a conversation to preserve important context while reducing token usage.
Create a concise summary that:
1. Preserves all important decisions, facts, and outcomes
2. Removes repetitive discussion and tangential details
3. Maintains the conversation flow for future continuity
4. Keeps key points from user and assistant perspectives

Focus on what happened, what was decided, and what the user needs to remember.`
}

// createCompactionEvent creates a new Event representing the compaction.
// It's stored like a regular event but with Actions.Compaction populated.
func (c *Compactor) createCompactionEvent(invocationID string, compaction *session.EventCompaction) *session.Event {
	event := session.NewEvent(invocationID)
	event.Author = "system"
	event.Actions.Compaction = compaction

	// Compaction events are not LLM responses, but we can add metadata about them
	event.LLMResponse = model.LLMResponse{
		Content: &genai.Content{
			Role:  "assistant",
			Parts: []*genai.Part{{Text: "Session history compacted"}},
		},
	}

	return event
}
