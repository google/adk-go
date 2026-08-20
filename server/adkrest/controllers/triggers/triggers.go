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

package triggers

import (
	"context"
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net/http"
	"strings"
	"time"

	"github.com/gorilla/mux"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/internal/compactionvalidate"
	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/server/adkrest/controllers"
	"google.golang.org/adk/v2/server/adkrest/internal/models"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

type RetriableRunner struct {
	sessionService  session.Service
	agentLoader     agent.Loader
	memoryService   memory.Service
	artifactService artifact.Service
	pluginConfig    runner.PluginConfig
	triggerConfig   TriggerConfig

	eventsCompactionConfig *compaction.Config
}

// ControllerOption configures optional behaviour shared by the trigger
// controllers.
//
// Their constructors take required dependencies positionally; anything optional
// is supplied here instead, so new capabilities do not keep widening those
// signatures or break existing callers.
type ControllerOption func(*RetriableRunner)

// WithEventsCompactionConfig enables context compaction for the runners a
// trigger controller creates, replacing older session events with summaries.
//
// The sliding window reduces prompt size by a constant factor rather than
// bounding it. Only tail retention bounds growth, and only when the sliding
// window is off: with both enabled the sliding window consumes the events tail
// retention would summarize and it never fires. Enable one. See
// [compaction.Config].
//
// Note what a trigger surface is. A delivery gets a session of its own, so
// history does not accumulate across messages and a sliding window counting
// completed invocations has little to count. Tail retention works normally,
// because it measures the prompt inside a single run.
//
// Two sliding-window configurations are worth a word, and neither is fatal, so
// both are logged rather than rejected:
//
//   - An interval of 1 fires on every delivery, including a single-turn one. It
//     spends a summarizer call to write a summary into a session that is
//     discarded when the delivery ends, so nothing ever reads it.
//   - A larger interval will not fire on a delivery handled in one attempt,
//     because that session sees a single invocation. Retries of one delivery do
//     share a session, so it can still fire on a message that was throttled.
func WithEventsCompactionConfig(cfg *compaction.Config) ControllerOption {
	return func(r *RetriableRunner) {
		switch {
		case cfg == nil:
		case cfg.CompactionInterval == 1:
			log.Printf("adk: sliding-window compaction is configured on a trigger controller with " +
				"CompactionInterval 1, so it fires on every delivery and writes a summary into a " +
				"session that is discarded when the delivery ends. Use TokenThreshold and " +
				"EventRetentionSize to compact within a single run.")
		case cfg.CompactionInterval > 1:
			log.Printf("adk: sliding-window compaction is configured on a trigger controller, but a " +
				"delivery handled in one attempt runs a single invocation, so the window will not " +
				"reach its interval. Use TokenThreshold and EventRetentionSize to compact within a " +
				"single run.")
		}
		r.eventsCompactionConfig = cfg
	}
}

// validateCompaction reports whether the compaction config can actually serve
// every app this controller knows about.
func (r *RetriableRunner) validateCompaction() error {
	return compactionvalidate.AgainstAgents(r.eventsCompactionConfig, r.agentLoader, runner.Config{
		SessionService:  r.sessionService,
		MemoryService:   r.memoryService,
		ArtifactService: r.artifactService,
		PluginConfig:    r.pluginConfig,
	})
}

func (r *RetriableRunner) RunAgent(ctx context.Context, appName, userID, messageContent string) ([]*session.Event, error) {
	// One session per delivery. Retries of that delivery reuse it, so a
	// throttled message accumulates invocations rather than starting over.
	sessReq := &session.CreateRequest{
		AppName: appName,
		UserID:  userID,
	}
	sessResp, err := r.sessionService.Create(ctx, sessReq)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %v", err)
	}

	userMessage := genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{Text: messageContent},
		},
	}

	curAgent, err := r.agentLoader.LoadAgent(appName)
	if err != nil {
		return nil, fmt.Errorf("failed to load agent: %v", err)
	}

	runR, err := runner.New(runner.Config{
		AppName:                appName,
		Agent:                  curAgent,
		SessionService:         r.sessionService,
		MemoryService:          r.memoryService,
		ArtifactService:        r.artifactService,
		PluginConfig:           r.pluginConfig,
		EventsCompactionConfig: r.eventsCompactionConfig,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create runner: %v", err)
	}

	return r.runAgentWithRetry(ctx, runR, sessResp.Session.UserID(), sessResp.Session.ID(), &userMessage)
}

// runAgentWithRetry uses exponential backoff with jitter to handle 429 rate-limit errors.
// After MaxRetries is exhausted, raises an error to signal the upstream service (Pub/Sub, Eventarc) to retry at a higher level.
func (r *RetriableRunner) runAgentWithRetry(ctx context.Context, runR *runner.Runner, userID, sessionID string, userMessage *genai.Content) ([]*session.Event, error) {
	var runErr error
	events := []*session.Event{}
	for i := 0; i <= r.triggerConfig.MaxRetries; i++ {
		resp := runR.Run(ctx, userID, sessionID, userMessage, agent.RunConfig{StreamingMode: agent.StreamingModeNone})

		isThrottled := false
		for event, err := range resp {
			if err != nil {
				// A compaction failure is bookkeeping, not the delivery. The
				// agent has already answered and its events are persisted, so
				// failing here would NACK a message that was handled, and on
				// Pub/Sub push that means redelivering work already done.
				if errors.Is(err, compaction.ErrCompaction) {
					log.Printf("triggers: %v", err)
					continue
				}
				runErr = err
				if isResourceExhausted(err) {
					isThrottled = true
				}
				break
			}
			events = append(events, event)
		}

		if !isThrottled && runErr == nil {
			return events, nil // Success
		}

		if i < r.triggerConfig.MaxRetries && isThrottled {
			delay := calculateBackoff(i, r.triggerConfig.BaseDelay, r.triggerConfig.MaxDelay)
			time.Sleep(delay)
			runErr = nil // Clear error for next attempt
			continue
		}
		break // Not throttled (but error raised) or max retries reached
	}
	return nil, runErr
}

func respondError(w http.ResponseWriter, code int, msg string) {
	resp := models.TriggerResponse{Status: msg}
	controllers.EncodeJSONResponse(resp, code, w)
}

func respondSuccess(w http.ResponseWriter) {
	resp := models.TriggerResponse{Status: "success"}
	controllers.EncodeJSONResponse(resp, http.StatusOK, w)
}

// Check if an exception represents a transient rate-limit error.
func isResourceExhausted(err error) bool {
	return err != nil && (strings.Contains(err.Error(), "429") || strings.Contains(err.Error(), "ResourceExhausted"))
}

func calculateBackoff(attempt int, base, maxDelay time.Duration) time.Duration {
	backoff := float64(base) * math.Pow(2, float64(attempt))
	delay := min(time.Duration(backoff), maxDelay)
	jitter := time.Duration(rand.Float64() * float64(delay) * 0.5)
	return delay + jitter
}

// Resolve the target app name from the request.
func appName(r *http.Request) (string, error) {
	vars := mux.Vars(r)
	appName := vars["app_name"]
	if appName == "" {
		return "", fmt.Errorf("no application name provided")
	}
	return appName, nil
}
