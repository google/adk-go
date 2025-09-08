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
	// swagger:operation POST /apps/{app_name}/users/{user_id}/sessions/{session_id} sessions getSession
	//
	// Returns a session.
	//
	// ---
	//
	//	parameters:
	//	- name: app_name
	//	  in: path
	//	  description: The app name.
	//	  required: true
	//	  type: string
	//	- name: user_id
	//	  in: path
	//	  description: The user id.
	//	  required: true
	//	  type: string
	//	- name: session_id
	//	  in: path
	//	  description: The session id.
	//	  required: true
	//	  type: string
	//
	//	responses:
	//	  '200':
	//	    description: The session was found.
	//	  '400':
	//	    description: Bad request.
	unimplemented(rw, req)
}

func (*SessionsApiController) ListSessions(rw http.ResponseWriter, req *http.Request) {
	unimplemented(rw, req)
}
