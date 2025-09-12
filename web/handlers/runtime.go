package handlers

import (
	"context"
	"encoding/json"
	goerr "errors"
	"fmt"
	"net/http"

	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/sessionservice"
	"google.golang.org/adk/web/errors"
	"google.golang.org/adk/web/models"
	"google.golang.org/adk/web/services"
	"google.golang.org/adk/web/utils"
)

type RuntimeApiController struct {
	sessionService sessionservice.Service
	agentLoader    services.AgentLoader
}

func NewRuntimeApiRouter(sessionService sessionservice.Service, agentLoader services.AgentLoader) *RuntimeApiController {
	return &RuntimeApiController{sessionService: sessionService, agentLoader: agentLoader}
}

func (c *RuntimeApiController) RunAgent(rw http.ResponseWriter, req *http.Request) error {
	runAgentRequest, err := decodeRequestBody(req)
	if err != nil {
		return err
	}

	err = c.validateSessionExists(req.Context(), runAgentRequest.AppName, runAgentRequest.UserId, runAgentRequest.SessionId)
	if err != nil {
		return err
	}

	r, rCfg, err := c.getRunner(runAgentRequest)
	if err != nil {
		return err
	}

	resp := r.Run(req.Context(), runAgentRequest.UserId, runAgentRequest.SessionId, &runAgentRequest.NewMessage, rCfg)

	var errs []error
	var events []models.Event
	for event, err := range resp {
		if err != nil {
			errs = append(errs, err)
			continue
		}
		events = append(events, models.FromSessionEvent(*event))
	}
	finalErr := goerr.Join(errs...)
	if finalErr != nil {
		return errors.NewStatusError(finalErr, http.StatusInternalServerError)
	}
	utils.EncodeJSONResponse(events, http.StatusOK, rw)
	return nil
}

func (c *RuntimeApiController) RunAgentSse(rw http.ResponseWriter, req *http.Request) error {
	flusher, ok := rw.(http.Flusher)
	if !ok {
		return errors.NewStatusError(fmt.Errorf("streaming not supported"), http.StatusInternalServerError)
	}

	rw.Header().Set("Content-Type", "text/event-stream")
	rw.Header().Set("Cache-Control", "no-cache")
	rw.Header().Set("Connection", "keep-alive")

	runAgentRequest, err := decodeRequestBody(req)
	if err != nil {
		return err
	}

	err = c.validateSessionExists(req.Context(), runAgentRequest.AppName, runAgentRequest.UserId, runAgentRequest.SessionId)
	if err != nil {
		return err
	}

	r, rCfg, err := c.getRunner(runAgentRequest)
	if err != nil {
		return err
	}

	resp := r.Run(req.Context(), runAgentRequest.UserId, runAgentRequest.SessionId, &runAgentRequest.NewMessage, rCfg)

	rw.WriteHeader(http.StatusOK)
	for event, err := range resp {
		if err != nil {
			fmt.Fprintf(rw, "Error while running agent: %v\n", err)
			flusher.Flush()
			continue
		}
		flashEvent(flusher, rw, *event)
	}
	return nil
}

func flashEvent(flusher http.Flusher, rw http.ResponseWriter, event session.Event) {
	fmt.Fprintf(rw, "data: ")
	json.NewEncoder(rw).Encode(models.FromSessionEvent(event))
	fmt.Fprintf(rw, "\n")
	flusher.Flush()
}

func (c *RuntimeApiController) validateSessionExists(ctx context.Context, appName, userID, sessionID string) error {
	_, err := c.sessionService.Get(ctx, &sessionservice.GetRequest{
		ID: session.ID{
			AppName:   appName,
			UserID:    userID,
			SessionID: sessionID,
		},
	})
	if err != nil {
		return errors.NewStatusError(fmt.Errorf("get session: %w", err), http.StatusNotFound)
	}
	return nil
}

func (c *RuntimeApiController) getRunner(req models.RunAgentRequest) (*runner.Runner, *runner.RunConfig, error) {
	agent, err := c.agentLoader.LoadAgent(req.AppName)
	if err != nil {
		return nil, nil, errors.NewStatusError(fmt.Errorf("load agent: %w", err), http.StatusInternalServerError)
	}

	r, err := runner.New(req.AppName, agent, c.sessionService)
	if err != nil {
		return nil, nil, errors.NewStatusError(fmt.Errorf("create runner: %w", err), http.StatusInternalServerError)
	}

	streamingMode := runner.StreamingModeNone
	if req.Streaming {
		streamingMode = runner.StreamingModeSSE
	}
	return r, &runner.RunConfig{
		StreamingMode:             streamingMode,
		SaveInputBlobsAsArtifacts: true,
	}, nil
}

func decodeRequestBody(req *http.Request) (models.RunAgentRequest, error) {
	var runAgentRequest models.RunAgentRequest
	defer req.Body.Close()
	d := json.NewDecoder(req.Body)
	d.DisallowUnknownFields()
	if err := d.Decode(&runAgentRequest); err != nil {
		return runAgentRequest, errors.NewStatusError(fmt.Errorf("decode request: %w", err), http.StatusBadRequest)
	}
	return runAgentRequest, nil
}
