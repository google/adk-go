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

// Package triggers provides HTTP handlers that run an agent in response
// to external events such as Pub/Sub or Eventarc, retrying rate-limited
// runs with exponential backoff and jitter.
package triggers

import (
	"context"
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/gorilla/mux"
	"google.golang.org/api/idtoken"
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

// RetriableRunner runs an agent with retry semantics, creating a new session per retry.
type RetriableRunner struct {
	sessionService  session.Service
	agentLoader     agent.Loader
	memoryService   memory.Service
	artifactService artifact.Service
	pluginConfig    runner.PluginConfig
	triggerConfig   TriggerConfig

	eventsCompactionConfig *compaction.Config

	// oidc is a copy of TriggerConfig.OIDC taken at construction, so a caller
	// that keeps its own struct or allow-list slice and mutates it later does
	// not change what this handler enforces.
	oidc *OIDCConfig

	// validateIDToken defaults to idtoken.Validate when nil. It is a field
	// rather than a package variable so parallel tests can each install their
	// own fake without racing on the auth path.
	validateIDToken tokenValidator
}

// ControllerConfig carries everything the trigger controllers need.
//
// A config struct rather than functional options, matching the rest of this
// repository. Options would have solved only the compaction field and left the
// six positional parameters in place; a struct absorbs both, and a field added
// to it breaks no existing caller.
type ControllerConfig struct {
	SessionService  session.Service
	AgentLoader     agent.Loader
	MemoryService   memory.Service
	ArtifactService artifact.Service
	PluginConfig    runner.PluginConfig
	TriggerConfig   TriggerConfig

	// Compaction enables context compaction for the runners a trigger
	// controller creates, replacing older session events with summaries.
	//
	// The sliding window reduces prompt size by a constant factor rather than
	// bounding it. Only tail retention bounds growth, and it only fires when
	// more events accumulate between sliding-window compactions than
	// EventRetentionSize holds back, so a short interval with a large retention
	// size leaves it idle. See [compaction.Config].
	//
	// Note what a trigger surface is. A delivery gets a session of its own, so
	// history does not accumulate across messages and a sliding window counting
	// completed invocations has little to count. Tail retention works normally,
	// because it measures the prompt inside a single run.
	//
	// optional
	Compaction *compaction.Config
}

// newRetriableRunner builds the shared runner behind both trigger controllers.
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
func newRetriableRunner(cfg ControllerConfig) *RetriableRunner {
	switch {
	case cfg.Compaction == nil:
	case cfg.Compaction.CompactionInterval == 1:
		log.Printf("adk: sliding-window compaction is configured on a trigger controller with " +
			"CompactionInterval 1, so it fires on every delivery and writes a summary into a " +
			"session that is discarded when the delivery ends. Use TokenThreshold and " +
			"EventRetentionSize to compact within a single run.")
	case cfg.Compaction.CompactionInterval > 1:
		log.Printf("adk: sliding-window compaction is configured on a trigger controller, but a " +
			"delivery handled in one attempt runs a single invocation, so the window will not " +
			"reach its interval. Use TokenThreshold and EventRetentionSize to compact within a " +
			"single run.")
	}
	return &RetriableRunner{
		sessionService:         cfg.SessionService,
		agentLoader:            cfg.AgentLoader,
		memoryService:          cfg.MemoryService,
		artifactService:        cfg.ArtifactService,
		pluginConfig:           cfg.PluginConfig,
		triggerConfig:          cfg.TriggerConfig,
		eventsCompactionConfig: cfg.Compaction,
		oidc:                   cloneOIDCConfig(cfg.TriggerConfig.OIDC),
	}
}

// cloneOIDCConfig detaches the verification settings from the caller's copy,
// including the allow-list backing array.
func cloneOIDCConfig(cfg *OIDCConfig) *OIDCConfig {
	if cfg == nil {
		return nil
	}
	return &OIDCConfig{
		ExpectedAudience:       cfg.ExpectedAudience,
		AllowedServiceAccounts: slices.Clone(cfg.AllowedServiceAccounts),
	}
}

// validateOIDC rejects a non-nil OIDCConfig with no audience. Verification is
// switched off by leaving TriggerConfig.OIDC nil, so this combination is
// always a mistake: read as-is it would turn an allow-list into no
// verification at all, which is the failure mode this feature exists to close.
func (r *RetriableRunner) validateOIDC() error {
	if r.oidc != nil && r.oidc.ExpectedAudience == "" {
		return errors.New("triggers: OIDC.ExpectedAudience is required when OIDC is set; leave TriggerConfig.OIDC nil to disable verification")
	}
	return nil
}

func (r *RetriableRunner) validateCompaction() error {
	return compactionvalidate.AgainstAgents(r.eventsCompactionConfig, r.agentLoader, runner.Config{
		SessionService:  r.sessionService,
		MemoryService:   r.memoryService,
		ArtifactService: r.artifactService,
		PluginConfig:    r.pluginConfig,
	})
}

// RunAgent runs the agent for the given message and returns the resulting events.
func (r *RetriableRunner) RunAgent(ctx context.Context, appName, userID, messageContent string) ([]*session.Event, error) {
	// One session per delivery. Retries of that delivery reuse it, so a
	// throttled message accumulates invocations rather than starting over.
	sessReq := &session.CreateRequest{
		AppName: appName,
		UserID:  userID,
	}
	sessResp, err := r.sessionService.Create(ctx, sessReq)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %w", err)
	}

	userMessage := genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{Text: messageContent},
		},
	}

	curAgent, err := r.agentLoader.LoadAgent(appName)
	if err != nil {
		return nil, fmt.Errorf("failed to load agent: %w", err)
	}

	runR, err := runner.New(runner.Config{
		AppName:         appName,
		Agent:           curAgent,
		SessionService:  r.sessionService,
		MemoryService:   r.memoryService,
		ArtifactService: r.artifactService,
		PluginConfig:    r.pluginConfig,
		Compaction:      r.eventsCompactionConfig,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create runner: %w", err)
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

// tokenValidator verifies a Google-signed OIDC ID token against an expected
// audience, returning its claims. It matches the signature of
// idtoken.Validate, which is what production uses; tests substitute a fake so
// they don't make a real network call to Google's certificate endpoint.
type tokenValidator func(ctx context.Context, idToken, audience string) (*idtoken.Payload, error)

// Issuers Google uses for the OIDC ID tokens minted for Pub/Sub push
// subscriptions and Eventarc triggers. Both spellings are in circulation.
var googleIssuers = []string{"accounts.google.com", "https://accounts.google.com"}

// authFailedMessage is the entire body of every rejected trigger request.
// It is one constant for all of them: a body that varied with the reason
// would tell an anonymous caller which check it got past, and in particular
// that an allow-list exists. The reason goes to the server log instead.
const authFailedMessage = "authentication failed"

// authError distinguishes a caller that presented no usable credential (401)
// from one whose credential verified but is not a principal this endpoint
// accepts (403). The status is the only part of this the caller sees.
type authError struct {
	status int
	// err is for the server-side log only and never reaches the response.
	err error
}

func (e *authError) Error() string { return e.err.Error() }

// verifyPushRequestAuth requires a valid Google-signed OIDC bearer token when
// TriggerConfig.OIDC is set; it is a no-op when it is nil, preserving prior
// behavior for deployments that rely entirely on platform-level access control
// (for example Cloud Run configured to require IAM authentication) in front of
// this endpoint.
//
// Pub/Sub push subscriptions and Eventarc Cloud Run triggers configured with a
// service account both attach exactly this kind of token. Note that a
// subscription's audience is whatever --push-auth-token-audience was set to,
// defaulting to the full push endpoint URL, and that pinning the caller
// identity via AllowedServiceAccounts requires the token to carry an email
// claim, so unlike audience verification alone, that part does depend on how
// the trigger is configured on the Google Cloud side.
func (r *RetriableRunner) verifyPushRequestAuth(req *http.Request) error {
	if r.oidc == nil {
		return nil
	}
	// Unreachable through either WithConfig constructor, which reject this
	// pair. Deny rather than fall through to the unverified path, so a
	// controller built some other way cannot end up silently open.
	if err := r.validateOIDC(); err != nil {
		return &authError{status: http.StatusInternalServerError, err: err}
	}

	token, err := bearerToken(req.Header.Values("Authorization"))
	if err != nil {
		return &authError{status: http.StatusUnauthorized, err: err}
	}

	validate := r.validateIDToken
	if validate == nil {
		validate = idtoken.Validate
	}
	payload, err := validate(req.Context(), token, r.oidc.ExpectedAudience)
	if err != nil {
		return &authError{status: http.StatusUnauthorized, err: fmt.Errorf("invalid identity token: %w", err)}
	}
	if payload == nil {
		return &authError{status: http.StatusUnauthorized, err: errors.New("identity token verified with no payload")}
	}
	if !slices.Contains(googleIssuers, payload.Issuer) {
		return &authError{status: http.StatusUnauthorized, err: fmt.Errorf("untrusted issuer %q", payload.Issuer)}
	}

	// idtoken.Validate checks the signature, the expiry and the audience
	// string, and leaves every identity claim to the caller. Without an
	// allow-list the only property established is that somebody holds a
	// Google-signed token for this audience, which any principal able to mint
	// one can satisfy.
	if len(r.oidc.AllowedServiceAccounts) == 0 {
		return nil
	}
	email, _ := payload.Claims["email"].(string)
	// A non-boolean email_verified fails the assertion and so fails closed.
	verified, _ := payload.Claims["email_verified"].(bool)
	if email == "" || !verified || !slices.Contains(r.oidc.AllowedServiceAccounts, email) {
		return &authError{status: http.StatusForbidden, err: fmt.Errorf("principal %q (verified=%t) is not an allowed service account", email, verified)}
	}
	return nil
}

// bearerToken extracts the credential from the request's Authorization
// headers. RFC 9110 makes the scheme case-insensitive.
//
// More than one Authorization header is rejected rather than resolved: the
// first value would still have to verify, but a proxy in front that forwards
// the last one would then disagree with this handler about which credential
// the caller presented.
func bearerToken(authHeaders []string) (string, error) {
	if len(authHeaders) != 1 {
		if len(authHeaders) == 0 {
			return "", errors.New("no Authorization header")
		}
		return "", fmt.Errorf("request carries %d Authorization headers", len(authHeaders))
	}
	scheme, token, found := strings.Cut(authHeaders[0], " ")
	if !found || !strings.EqualFold(scheme, "Bearer") {
		return "", errors.New("authorization header is not a Bearer credential")
	}
	if token = strings.TrimSpace(token); token == "" {
		return "", errors.New("bearer credential is empty")
	}
	return token, nil
}

// respondAuthError rejects a request that failed verifyPushRequestAuth,
// keeping the reason server-side so an anonymous caller can't distinguish an
// expired token from an audience mismatch from a bad signature, nor a
// rejected principal from an unverifiable token.
func respondAuthError(w http.ResponseWriter, err error) {
	status := http.StatusUnauthorized
	var authErr *authError
	if errors.As(err, &authErr) {
		status = authErr.status
	}
	if status == http.StatusUnauthorized {
		w.Header().Set("WWW-Authenticate", `Bearer error="invalid_token"`)
	}
	log.Printf("adk: trigger request rejected with %d: %v", status, err)
	respondError(w, status, authFailedMessage)
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
