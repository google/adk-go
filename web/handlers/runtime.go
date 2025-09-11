package handlers

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/sessionservice"
	weberrors "google.golang.org/adk/web/errors"
	"google.golang.org/adk/web/models"
	"google.golang.org/adk/web/services"
)

type RuntimeApiController struct {
	sessionService sessionservice.Service
	agentLoader    services.AgentLoader
}

func NewRuntimeApiRouter(sessionService sessionservice.Service, agentLoader services.AgentLoader) *RuntimeApiController {
	return &RuntimeApiController{sessionService: sessionService, agentLoader: agentLoader}
}

func (c *RuntimeApiController) RunAgent(rw http.ResponseWriter, req *http.Request) error {
	var runAgentRequest models.RunAgentRequest
	d := json.NewDecoder(req.Body)
	d.DisallowUnknownFields()
	if err := d.Decode(&runAgentRequest); err != nil {
		return weberrors.NewStatusError(fmt.Errorf("decode request: %w", err), http.StatusBadRequest)
	}
	// if err := runAgentRequest.AssertRunAgentRequestRequired(); err != nil {
	// 	return weberrors.NewStatusError(err, http.StatusBadRequest)
	// }
	if req.Method == "OPTIONS" {
		rw.WriteHeader(http.StatusOK)
		return nil
	}
	// _, err := c.sessionService.Get(req.Context(), &sessionservice.GetRequest{
	// 	ID: session.ID{
	// 		AppName:   runAgentRequest.AppName,
	// 		UserID:    runAgentRequest.UserId,
	// 		SessionID: runAgentRequest.SessionId,
	// 	},
	// })
	// if err != nil {
	// 	return weberrors.NewStatusError(err, http.StatusInternalServerError)
	// }
	agent, err := c.agentLoader.LoadAgent(runAgentRequest.AppName)
	if err != nil {
		fmt.Printf("load agent: %v", err)
		return weberrors.NewStatusError(fmt.Errorf("load agent: %w", err), http.StatusInternalServerError)
	}
	r, err := runner.New(runAgentRequest.AppName, agent, c.sessionService)
	if err != nil {
		fmt.Printf("create runner: %v", err)
		return weberrors.NewStatusError(fmt.Errorf("create runner: %w", err), http.StatusInternalServerError)
	}
	resp := r.Run(req.Context(), runAgentRequest.UserId, runAgentRequest.SessionId, &runAgentRequest.NewMessage, &runner.RunConfig{})
	events := []*session.Event{}
	errs := []error{}
	for event, err := range resp {
		errs = append(errs, err)
		events = append(events, event)
	}
	err = errors.Join(errs...)
	if err != nil {
		return weberrors.NewStatusError(fmt.Errorf("run: %w", err), http.StatusInternalServerError)
	}
	fmt.Printf("events: %v", events)
	rw.WriteHeader(http.StatusOK)
	return nil
}

func (c *RuntimeApiController) RunAgentSse(rw http.ResponseWriter, req *http.Request) {
	unimplemented(rw, req)
}
