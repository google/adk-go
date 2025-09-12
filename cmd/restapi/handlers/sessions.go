package handlers

import (
	"net/http"

	"google.golang.org/adk/sessionservice"
)

func unimplemented(rw http.ResponseWriter, req *http.Request) {
	rw.WriteHeader(http.StatusNotImplemented)
}

type SessionsApiController struct {
	service sessionservice.Service
}

func New(service sessionservice.Service) *SessionsApiController {
	return &SessionsApiController{service: service}
}

func (c *SessionsApiController) CreateSession(rw http.ResponseWriter, req *http.Request) {
	unimplemented(rw, req)
}

func (*SessionsApiController) DeleteSession(rw http.ResponseWriter, req *http.Request) {
	unimplemented(rw, req)
}

// GetSession handles receiving a sesion from the system.
func (*SessionsApiController) GetSession(rw http.ResponseWriter, req *http.Request) {
	unimplemented(rw, req)
}

func (*SessionsApiController) ListSessions(rw http.ResponseWriter, req *http.Request) {
	unimplemented(rw, req)
}
