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
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/gorilla/websocket"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/server/adkrest/internal/models"
	"google.golang.org/adk/session"
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
}

// NewRuntimeAPIController creates the controller for the Runtime API.
func NewRuntimeAPIController(sessionService session.Service, memoryService memory.Service, agentLoader agent.Loader, artifactService artifact.Service, sseTimeout time.Duration, pluginConfig runner.PluginConfig, autoCreateSession bool) *RuntimeAPIController {
	return &RuntimeAPIController{sessionService: sessionService, memoryService: memoryService, agentLoader: agentLoader, artifactService: artifactService, sseTimeout: sseTimeout, pluginConfig: pluginConfig, autoCreateSession: autoCreateSession}
}

// RunAgent executes a non-streaming agent run for a given session and message.
func (c *RuntimeAPIController) RunHandler(rw http.ResponseWriter, req *http.Request) error {
	runAgentRequest, err := decodeRequestBody(req)
	if err != nil {
		return err
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

	resp := r.Run(ctx, runAgentRequest.UserId, runAgentRequest.SessionId, &runAgentRequest.NewMessage, *rCfg)

	var events []*session.Event
	for event, err := range resp {
		if err != nil {
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

func decodeRequestBody(req *http.Request) (decodedReq models.RunAgentRequest, err error) {
	var runAgentRequest models.RunAgentRequest
	defer func() {
		err = req.Body.Close()
	}()
	d := json.NewDecoder(req.Body)
	d.DisallowUnknownFields()
	if err := d.Decode(&runAgentRequest); err != nil {
		return runAgentRequest, newStatusError(fmt.Errorf("failed to decode request: %w", err), http.StatusBadRequest)
	}
	return runAgentRequest, nil
}

func (c *RuntimeAPIController) RunLiveHandler(rw http.ResponseWriter, req *http.Request) error {
	upgrader := websocket.Upgrader{
		ReadBufferSize:  1024,
		WriteBufferSize: 1024,
		CheckOrigin:     func(r *http.Request) bool { return true },
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
	sessionID := q.Get("sessionId")
	if sessionID == "" {
		sessionID = q.Get("session_id")
	}

	if appName == "" || userID == "" || sessionID == "" {
		return fmt.Errorf("appName, userId, and sessionId are required")
	}

	ws, err := upgrader.Upgrade(rw, req, nil)
	if err != nil {
		return fmt.Errorf("failed to upgrade to websocket: %w", err)
	}
	defer ws.Close()

	sendClose := func(code int, reason string) {
		ws.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(code, reason))
		ws.SetReadDeadline(time.Now().Add(time.Second))
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
			closeReason = fmt.Sprintf("agent %s not found", appName)
		}
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
		sendClose(websocket.CloseInternalServerErr, err.Error())
		return nil
	}
	defer liveSession.Close()

	// Spawning goroutine for reading from the client over WebSocket and pushing it to Runner
	go func() {
		for {
			messageType, p, err := ws.ReadMessage()
			if err != nil {
				// WebSocket is closed or error occurred
				break
			}

			if messageType == websocket.BinaryMessage {
				if err := liveSession.Send(agent.LiveRequest{
					RealtimeInput: &genai.Blob{
						MIMEType: "audio/pcm;rate=16000",
						Data:     p,
					},
				}); err != nil {
					break
				}
			} else if messageType == websocket.TextMessage {
				var apiReq models.LiveRequest
				if err := json.Unmarshal(p, &apiReq); err != nil {
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
					break
				}
			}
		}
	}()

	for event, err := range eventIter {
		if err != nil {
			fmt.Printf("RunLive failed: %v\n", err)
			ws.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseInternalServerErr, err.Error()))
			break
		}

		err = ws.WriteJSON(models.FromSessionEvent(*event))
		if err != nil {
			break
		}
	}

	return nil
}
