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

package controllers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"time"
	"unicode/utf8"

	"github.com/gorilla/websocket"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/server/adkrest/internal/models"
	"google.golang.org/adk/v2/server/authz"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

// RuntimeAPIController is the controller for the Runtime API.
type RuntimeAPIController struct {
	sseTimeout        time.Duration
	sessionService    session.Service
	memoryService     memory.Service
	artifactService   artifact.Service
	agentLoader       agent.Loader
	pluginConfig      runner.PluginConfig
	autoCreateSession bool
	authorizer        authz.Authorizer

	checkOrigin func(*http.Request) bool

	maxLiveMessageBytes  int64
	liveKeepaliveTimeout time.Duration
	// livePingInterval is how often the server pings a live connection. Zero
	// when the keepalive is off.
	livePingInterval time.Duration
	// liveSlots holds one token per live session the controller may serve at
	// once. Nil when the cap is disabled.
	liveSlots chan struct{}

	eventsCompactionConfig *compaction.Config
}

// Defaults for the /run_live limits. The message and keepalive bounds follow
// uvicorn, which serves adk-python's /run_live: ws_max_size is 16 MiB and
// ws_ping_interval/ws_ping_timeout are 20s each, so a peer that stops answering
// pings is dropped after roughly 40s.
const (
	defaultMaxLiveMessageBytes  = 16 << 20
	defaultLiveKeepaliveTimeout = 40 * time.Second

	// liveWriteWait bounds a ping write, which shares the connection with the
	// event loop's writes.
	liveWriteWait = 10 * time.Second
)

// RuntimeAPIControllerConfig carries everything [NewRuntimeAPIControllerWithConfig]
// needs.
//
// A config struct rather than functional options, which is what the rest of
// this repository does and what this constructor should have taken from the
// start. Options would have solved only the new field and left the seven
// positional parameters in place, so every later addition would need a third
// constructor; a struct absorbs both problems at once, and adding a field to it
// breaks nobody.
type RuntimeAPIControllerConfig struct {
	SessionService    session.Service
	MemoryService     memory.Service
	AgentLoader       agent.Loader
	ArtifactService   artifact.Service
	SSETimeout        time.Duration
	PluginConfig      runner.PluginConfig
	AutoCreateSession bool
	Authorizer        authz.Authorizer

	// Compaction enables context compaction for the runners this controller
	// creates, replacing older session events with summaries.
	//
	// The sliding window reduces prompt size by a constant factor rather than
	// bounding it. Only tail retention bounds growth, and it only fires when
	// more events accumulate between sliding-window compactions than
	// EventRetentionSize holds back, so a short interval with a large retention
	// size leaves it idle. See [compaction.Config].
	//
	// optional
	Compaction *compaction.Config

	// CheckOrigin reports whether a /run_live upgrade carrying this request's
	// Origin may proceed. It becomes the WebSocket upgrader's CheckOrigin hook,
	// and a false answer refuses the handshake with 403.
	//
	// [google.golang.org/adk/v2/server/adkrest.NewServer] supplies one built
	// from its AllowedOrigins, which is where the check belongs for anyone
	// using that server. Set this only when mounting this controller in a
	// router of your own.
	//
	// optional; nil keeps gorilla/websocket's default, which accepts a request
	// with no Origin and otherwise requires Origin's host to equal Host — and
	// so accepts a page that reached this server by rebinding its own DNS name,
	// since such a page controls both
	CheckOrigin func(*http.Request) bool

	// MaxLiveMessageBytes caps one client message on a /run_live connection.
	// A message larger than the cap closes the connection with close code 1009
	// instead of being buffered, so a single caller cannot make the server
	// allocate without bound.
	//
	// optional; zero means 16 MiB, negative means no cap
	MaxLiveMessageBytes int64

	// LiveKeepaliveTimeout drops a /run_live peer that has stopped responding.
	// The server pings at half this interval. While it waits on the client, a
	// pong or a client message restarts the clock, and a write the peer does
	// not accept within the timeout also ends the connection.
	//
	// It is not an idle timeout: a client that sends nothing but still answers
	// pings, as every browser does, stays connected. What it removes is a peer
	// that has gone away without closing, which as a hijacked connection sits
	// outside [net/http.Server]'s own timeouts and would otherwise hold its
	// agent session for as long as the process runs.
	//
	// optional; zero means 40s, negative turns the keepalive off
	LiveKeepaliveTimeout time.Duration

	// MaxLiveSessions caps how many /run_live connections this controller
	// serves at once. Past the cap the handler answers 503 and does not
	// upgrade, so one caller cannot hold open more agent sessions, model
	// connections and goroutines than the process can carry.
	//
	// There is no cap by default, as in adk-python. Behind an autoscaler such
	// as Cloud Run, the platform's own per-instance concurrency setting decides
	// admission, and a lower cap here would refuse sessions the platform has
	// already routed to this instance instead of letting it scale out.
	//
	// optional; zero or negative means no cap
	MaxLiveSessions int
}

// NewRuntimeAPIController creates the controller for the Runtime API.
//
// Deprecated: use [NewRuntimeAPIControllerWithConfig], which does not have to
// grow a parameter every time the controller gains a setting. This one is kept
// because it is released API and every existing call site still compiles; it is
// a candidate for removal at the next major version.
func NewRuntimeAPIController(sessionService session.Service, memoryService memory.Service, agentLoader agent.Loader, artifactService artifact.Service, sseTimeout time.Duration, pluginConfig runner.PluginConfig, autoCreateSession bool) *RuntimeAPIController {
	return NewRuntimeAPIControllerWithConfig(RuntimeAPIControllerConfig{
		SessionService:    sessionService,
		MemoryService:     memoryService,
		AgentLoader:       agentLoader,
		ArtifactService:   artifactService,
		SSETimeout:        sseTimeout,
		PluginConfig:      pluginConfig,
		AutoCreateSession: autoCreateSession,
	})
}

// WithAuthorizer sets the authorizer. Provided to be compatible with [NewRuntimeAPIController]
// Deprecated: use [NewRuntimeAPIControllerWithConfig] to set authorizer directly in RuntimeAPIControllerConfig.
func (c *RuntimeAPIController) WithAuthorizer(authorizer authz.Authorizer) {
	c.authorizer = authorizer
}

// NewRuntimeAPIControllerWithConfig creates the controller for the Runtime API.
//
// A separate constructor rather than a variadic parameter on the one above:
// adding a parameter would change that function's type, which breaks any caller
// holding it as a value even though ordinary call sites still compile, and it
// is released API.
func NewRuntimeAPIControllerWithConfig(cfg RuntimeAPIControllerConfig) *RuntimeAPIController {
	authorizer := cfg.Authorizer
	if authorizer == nil {
		authorizer = authz.NewNoop()
	}

	maxLiveMessageBytes := cfg.MaxLiveMessageBytes
	if maxLiveMessageBytes == 0 {
		maxLiveMessageBytes = defaultMaxLiveMessageBytes
	}
	liveKeepaliveTimeout := cfg.LiveKeepaliveTimeout
	if liveKeepaliveTimeout == 0 {
		liveKeepaliveTimeout = defaultLiveKeepaliveTimeout
	}
	var livePingInterval time.Duration
	if liveKeepaliveTimeout > 0 {
		// A timeout under 2ns would halve to zero, which time.NewTicker rejects.
		livePingInterval = max(liveKeepaliveTimeout/2, time.Nanosecond)
	}
	var liveSlots chan struct{}
	if cfg.MaxLiveSessions > 0 {
		liveSlots = make(chan struct{}, cfg.MaxLiveSessions)
	}

	return &RuntimeAPIController{
		sessionService:         cfg.SessionService,
		memoryService:          cfg.MemoryService,
		agentLoader:            cfg.AgentLoader,
		artifactService:        cfg.ArtifactService,
		sseTimeout:             cfg.SSETimeout,
		pluginConfig:           cfg.PluginConfig,
		autoCreateSession:      cfg.AutoCreateSession,
		checkOrigin:            cfg.CheckOrigin,
		maxLiveMessageBytes:    maxLiveMessageBytes,
		liveKeepaliveTimeout:   liveKeepaliveTimeout,
		livePingInterval:       livePingInterval,
		liveSlots:              liveSlots,
		eventsCompactionConfig: cfg.Compaction,
		authorizer:             authorizer,
	}
}

// RunHandler executes a non-streaming agent run for a given session and message.
func (c *RuntimeAPIController) RunHandler(rw http.ResponseWriter, req *http.Request) error {
	runAgentRequest, err := decodeRequestBody(req)
	if err != nil {
		return err
	}

	if c.authorizer != nil {
		if err := c.authorizer.CanActAsUser(req.Context(), runAgentRequest.UserId); err != nil {
			authz.WriteHTTPStatusForAuthError(rw, err)
			return nil
		}
	}

	sessionEvents, err := c.runAgent(req.Context(), runAgentRequest)
	if err != nil {
		return err
	}
	var events []models.Event
	for _, event := range sessionEvents {
		events = append(events, models.FromSessionEvent(*event))
	}
	EncodeJSONResponse(events, http.StatusOK, rw)
	return nil
}

// RunAgent executes a non-streaming agent run for a given session and message.
func (c *RuntimeAPIController) runAgent(ctx context.Context, runAgentRequest models.RunAgentRequest) ([]*session.Event, error) {
	err := c.validateSessionExists(ctx, runAgentRequest.AppName, runAgentRequest.UserId, runAgentRequest.SessionId)
	if err != nil {
		return nil, err
	}

	r, rCfg, err := c.getRunner(runAgentRequest)
	if err != nil {
		return nil, err
	}

	var opts []runner.RunOption
	if runAgentRequest.StateDelta != nil {
		opts = append(opts, runner.WithStateDelta(*runAgentRequest.StateDelta))
	}
	resp := r.Run(ctx, runAgentRequest.UserId, runAgentRequest.SessionId, &runAgentRequest.NewMessage, *rCfg, opts...)

	var events []*session.Event
	for event, err := range resp {
		if err != nil {
			// A compaction failure is bookkeeping, not the turn. The events are
			// already persisted and the agent has already answered, so failing
			// the request would discard work the caller asked for and paid for
			// in order to report that a later prompt will be larger.
			if errors.Is(err, compaction.ErrCompaction) {
				log.Printf("adkrest: %v", err)
				continue
			}
			return nil, newStatusError(fmt.Errorf("failed to run agent: %w", err), http.StatusInternalServerError)
		}
		events = append(events, event)
	}
	return events, nil
}

// RunSSEHandler executes an agent run and streams the resulting events using Server-Sent Events (SSE).
func (c *RuntimeAPIController) RunSSEHandler(rw http.ResponseWriter, req *http.Request) {
	// set custom deadlines for this request - it overrides server-wide timeouts
	rc := http.NewResponseController(rw)
	deadline := time.Now().Add(c.sseTimeout)
	err := rc.SetWriteDeadline(deadline)
	if err != nil {
		http.Error(rw, "failed to set write deadline: "+err.Error(), http.StatusInternalServerError)
		return
	}

	runAgentRequest, err := decodeRequestBody(req)
	if err != nil {
		http.Error(rw, "failed to decode request body: "+err.Error(), http.StatusBadRequest)
		return
	}

	if c.authorizer != nil {
		if err := c.authorizer.CanActAsUser(req.Context(), runAgentRequest.UserId); err != nil {
			authz.WriteHTTPStatusForAuthError(rw, err)
			return
		}
	}

	err = c.validateSessionExists(req.Context(), runAgentRequest.AppName, runAgentRequest.UserId, runAgentRequest.SessionId)
	if err != nil {
		http.Error(rw, "failed to find the session: "+err.Error(), http.StatusNotFound)
		return
	}

	r, rCfg, err := c.getRunner(runAgentRequest)
	if err != nil {
		http.Error(rw, "failed to get runner: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Flush as soon as possible so the client doesn't drop connection.
	// Add the headers after the error handling to avoid wrong content type.
	rw.Header().Set("Content-Type", "text/event-stream")
	rw.Header().Set("Cache-Control", "no-cache")
	rw.Header().Set("Connection", "keep-alive")
	if err := rc.Flush(); err != nil {
		http.Error(rw, "failed to flush headers", http.StatusInternalServerError)
		return
	}

	opts := []runner.RunOption{}
	if runAgentRequest.StateDelta != nil {
		opts = append(opts, runner.WithStateDelta(*runAgentRequest.StateDelta))
	}
	resp := r.Run(req.Context(), runAgentRequest.UserId, runAgentRequest.SessionId, &runAgentRequest.NewMessage, *rCfg, opts...)

	for event, err := range resp {
		if err != nil {
			// Bookkeeping, not the turn: see the RunHandler comment. Streaming
			// an error event here would tell a client its answer failed after
			// it has already received it.
			if errors.Is(err, compaction.ErrCompaction) {
				log.Printf("adkrest: %v", err)
				continue
			}
			err := flashErrorEvent(rc, rw, err)
			// The error is returned only when we cannot communicate with the client
			// Exit the handler as connection is closed.
			if err != nil {
				log.Printf("failed to flash error event: %v", err)
				return
			}
			continue
		}
		if event == nil {
			continue
		}
		// Skip reporting error if it fails to marshal to the client (to avoid recursive error reporting).
		marshalledData, err := json.Marshal(models.FromSessionEvent(*event))
		if err != nil {
			log.Printf("failed to marshal event: %v", err)
			return
		}
		err = flashEvent(rc, rw, string(marshalledData))
		if err != nil {
			log.Printf("failed to flash event: %v", err)
			return
		}
	}
}

func flashErrorEvent(rc *http.ResponseController, rw http.ResponseWriter, origError error) error {
	_, err := fmt.Fprintf(rw, "event: error\n")
	if err != nil {
		return fmt.Errorf("write error event: %w", err)
	}
	safeErrorJSON, err := json.Marshal(map[string]string{"error": origError.Error()})
	if err != nil {
		// Skip reporting error if it fails to marshal to the client (to avoid recursive error reporting).
		return fmt.Errorf("marshal error event: %w", err)
	}
	return flashEvent(rc, rw, string(safeErrorJSON))
}

func flashEvent(rc *http.ResponseController, rw http.ResponseWriter, data string) error {
	_, err := fmt.Fprintf(rw, "data: %s\n\n", data)
	if err != nil {
		return fmt.Errorf("write response: %w", err)
	}
	err = rc.Flush()
	if err != nil {
		return fmt.Errorf("flush event: %w", err)
	}
	return nil
}

func (c *RuntimeAPIController) validateSessionExists(ctx context.Context, appName, userID, sessionID string) error {
	_, err := c.sessionService.Get(ctx, &session.GetRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
	})
	if err != nil {
		return newStatusError(fmt.Errorf("failed to get session: %w", err), http.StatusNotFound)
	}
	return nil
}

func (c *RuntimeAPIController) getRunner(req models.RunAgentRequest) (*runner.Runner, *agent.RunConfig, error) {
	curAgent, err := c.agentLoader.LoadAgent(req.AppName)
	if err != nil {
		return nil, nil, newStatusError(fmt.Errorf("failed to load agent: %w", err), http.StatusInternalServerError)
	}

	r, err := runner.New(runner.Config{
		AppName:           req.AppName,
		Agent:             curAgent,
		SessionService:    c.sessionService,
		MemoryService:     c.memoryService,
		ArtifactService:   c.artifactService,
		PluginConfig:      c.pluginConfig,
		Compaction:        c.eventsCompactionConfig,
		AutoCreateSession: c.autoCreateSession,
	},
	)
	if err != nil {
		return nil, nil, newStatusError(fmt.Errorf("failed to create runner: %w", err), http.StatusInternalServerError)
	}

	streamingMode := agent.StreamingModeNone
	if req.Streaming {
		streamingMode = agent.StreamingModeSSE
	}
	return r, &agent.RunConfig{
		StreamingMode: streamingMode,
	}, nil
}

func decodeRequestBody(req *http.Request) (models.RunAgentRequest, error) {
	var runAgentRequest models.RunAgentRequest
	d := json.NewDecoder(req.Body)
	d.DisallowUnknownFields()
	if err := d.Decode(&runAgentRequest); err != nil {
		return runAgentRequest, newStatusError(fmt.Errorf("failed to decode request: %w", err), http.StatusBadRequest)
	}
	return runAgentRequest, nil
}

// RunLiveHandler upgrades the request to a WebSocket and streams a live agent
// session.
func (c *RuntimeAPIController) RunLiveHandler(rw http.ResponseWriter, req *http.Request) error {
	upgrader := websocket.Upgrader{
		ReadBufferSize:  1024,
		WriteBufferSize: 1024,
		CheckOrigin:     c.checkOrigin,
	}

	q := req.URL.Query()
	appName := q.Get("appName")
	if appName == "" {
		appName = q.Get("app_name")
	}
	userID := q.Get("userId")
	if userID == "" {
		userID = q.Get("user_id")
	}

	if c.authorizer != nil {
		if err := c.authorizer.CanActAsUser(req.Context(), userID); err != nil {
			authz.WriteHTTPStatusForAuthError(rw, err)
			return nil
		}
	}

	sessionID := q.Get("sessionId")
	if sessionID == "" {
		sessionID = q.Get("session_id")
	}

	if appName == "" || userID == "" || sessionID == "" {
		return fmt.Errorf("appName, userId, and sessionId are required")
	}

	// Before the upgrade, so a refusal is an ordinary HTTP response the client
	// can read rather than a close frame on a connection it just won.
	release, ok := c.acquireLiveSlot()
	if !ok {
		return newStatusError(fmt.Errorf("live session limit of %d reached", cap(c.liveSlots)), http.StatusServiceUnavailable)
	}
	defer release()

	ws, err := upgrader.Upgrade(rw, req, nil)
	if err != nil {
		return fmt.Errorf("failed to upgrade to websocket: %w", err)
	}
	defer func() {
		_ = ws.Close()
	}()

	stopPings := c.applyLiveConnLimits(ws)
	defer stopPings()

	sendClose := func(code int, reason string) {
		_ = ws.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(code, truncateCloseReason(reason)))
		// Only the close reply matters from here. The keepalive's pong handler
		// would push the deadline below out on every pong, so a client sending
		// pongs could hold the connection open past the drain.
		ws.SetPongHandler(nil)
		_ = ws.SetReadDeadline(time.Now().Add(time.Second))
		for {
			if _, _, err := ws.ReadMessage(); err != nil {
				break
			}
		}
	}

	r, _, err := c.getRunner(models.RunAgentRequest{AppName: appName, UserId: userID, SessionId: sessionID})
	if err != nil {
		closeReason := err.Error()
		if _, loadErr := c.agentLoader.LoadAgent(appName); loadErr != nil {
			closeReason = fmt.Sprintf("agent %s not found for original error: %v", appName, err)
		}
		log.Printf("Failed to get runner for app %s: %v", appName, err)
		sendClose(websocket.CloseInternalServerErr, closeReason)
		return nil
	}

	// Read from Runner and write back to client over the WebSocket
	liveSession, eventIter, err := r.RunLive(req.Context(), userID, sessionID, agent.LiveRunConfig{
		MaxLLMCalls:              100, // Reasonable default
		ResponseModalities:       []genai.Modality{genai.ModalityAudio},
		InputAudioTranscription:  &genai.AudioTranscriptionConfig{},
		OutputAudioTranscription: &genai.AudioTranscriptionConfig{},
	})
	if err != nil {
		log.Printf("RunLive failed for app %s: %v", appName, err)
		sendClose(websocket.CloseInternalServerErr, err.Error())
		return nil
	}
	defer func() {
		_ = liveSession.Close()
	}()

	// Spawning goroutine for reading from the client over WebSocket and pushing it to Runner
	go func() {
		defer func() {
			_ = liveSession.Close()
		}()
		for {
			// Armed only while waiting on the client, so neither session
			// setup nor a slow hand-off to the agent counts against a peer
			// that answers every ping.
			_ = ws.SetReadDeadline(c.liveDeadline())
			messageType, p, err := ws.ReadMessage()
			if err != nil {
				if !websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
					log.Printf("WebSocket read error for app %s: %v", appName, err)
				}
				break
			}

			if messageType == websocket.BinaryMessage {
				if err := liveSession.Send(agent.LiveRequest{
					RealtimeInput: &genai.Blob{
						MIMEType: "audio/pcm;rate=16000",
						Data:     p,
					},
				}); err != nil {
					log.Printf("Failed to send binary data to Gemini for app %s: %v", appName, err)
					break
				}
			} else if messageType == websocket.TextMessage {
				var apiReq models.LiveRequest
				if err := json.Unmarshal(p, &apiReq); err != nil {
					log.Printf("Failed to unmarshal client message for app %s: %v", appName, err)
					continue
				}

				if apiReq.Close {
					break
				}

				liveReq := agent.LiveRequest{
					Content: apiReq.Content,
				}

				if apiReq.ActivityStart != nil {
					liveReq.RealtimeInput = apiReq.ActivityStart
				} else if apiReq.ActivityEnd != nil {
					liveReq.RealtimeInput = apiReq.ActivityEnd
				} else if apiReq.Blob != nil {
					liveReq.RealtimeInput = &genai.Blob{
						MIMEType: apiReq.Blob.MIMEType,
						Data:     apiReq.Blob.Data,
					}
				}

				if err := liveSession.Send(liveReq); err != nil {
					log.Printf("Failed to send message to Gemini for app %s: %v", appName, err)
					break
				}
			}
		}
	}()

	for event, err := range eventIter {
		// A peer that accepts nothing for the keepalive timeout has stopped
		// responding too. Without a deadline a write would block on it for
		// good, holding the handler and its session slot, since the reader
		// timing out only closes the live session. Set on every pass, before
		// either write below, so neither inherits a deadline that expired
		// while the agent was quiet.
		_ = ws.SetWriteDeadline(c.liveDeadline())
		if err != nil {
			log.Printf("RunLive failed: %v\n", err)
			_ = ws.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseInternalServerErr, truncateCloseReason(err.Error())))
			break
		}

		err = ws.WriteJSON(models.FromSessionEvent(*event))
		if err != nil {
			if !errors.Is(err, websocket.ErrCloseSent) {
				log.Printf("WebSocket write error for app %s: %v", appName, err)
			}
			break
		}
	}

	return nil
}

// acquireLiveSlot takes one of the controller's concurrent live-session slots.
// It reports false when they are all held, and otherwise returns the function
// that gives the slot back.
func (c *RuntimeAPIController) acquireLiveSlot() (release func(), ok bool) {
	if c.liveSlots == nil {
		return func() {}, true
	}
	select {
	case c.liveSlots <- struct{}{}:
		return func() { <-c.liveSlots }, true
	default:
		return nil, false
	}
}

// applyLiveConnLimits bounds what one live connection may consume: a read limit
// so a client message cannot be buffered without end, and the pings and pong
// handler of the keepalive, so a peer that stops responding is dropped. The
// keepalive's deadlines are set around each read and write, from
// [RuntimeAPIController.liveDeadline]. It returns the function that stops the
// pinger, which the caller must call.
func (c *RuntimeAPIController) applyLiveConnLimits(ws *websocket.Conn) (stop func()) {
	if c.maxLiveMessageBytes > 0 {
		ws.SetReadLimit(c.maxLiveMessageBytes)
	}
	if c.liveKeepaliveTimeout <= 0 {
		return func() {}
	}

	ws.SetPongHandler(func(string) error {
		// A failure means the connection is already gone, which the read in
		// progress reports.
		_ = ws.SetReadDeadline(c.liveDeadline())
		return nil
	})

	// Ping from a goroutine of its own: the caller's write loop blocks on the
	// event iterator, so it cannot also keep the clock running. WriteControl is
	// the one write method gorilla allows concurrently with the others.
	done := make(chan struct{})
	ticker := time.NewTicker(c.livePingInterval)
	go func() {
		defer ticker.Stop()
		for {
			select {
			case <-done:
				return
			case <-ticker.C:
				if err := ws.WriteControl(websocket.PingMessage, nil, time.Now().Add(liveWriteWait)); err != nil {
					return
				}
			}
		}
	}()
	return func() { close(done) }
}

// liveDeadline returns the deadline for the next read or write on a live
// connection: the keepalive timeout from now, or the zero time, which sets no
// deadline, when the keepalive is off.
func (c *RuntimeAPIController) liveDeadline() time.Time {
	if c.liveKeepaliveTimeout <= 0 {
		return time.Time{}
	}
	return time.Now().Add(c.liveKeepaliveTimeout)
}

// maxCloseReason is the longest reason a websocket close frame can carry: a
// control frame payload is capped at 125 bytes and the close code takes two.
const maxCloseReason = 123

// truncateCloseReason trims reason to fit a close frame, on a rune boundary
// because the reason must be valid UTF-8. gorilla refuses to send an over-long
// control frame at all, so without this a long error reaches the browser as a
// bare abnormal closure carrying no explanation.
func truncateCloseReason(reason string) string {
	if len(reason) <= maxCloseReason {
		return reason
	}
	truncated := reason[:maxCloseReason]
	for len(truncated) > 0 && !utf8.ValidString(truncated) {
		truncated = truncated[:len(truncated)-1]
	}
	return truncated
}
