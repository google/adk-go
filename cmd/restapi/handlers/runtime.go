package handlers

import (
	"net/http"
)

type RuntimeApiController struct{}

func (*RuntimeApiController) RunAgent(rw http.ResponseWriter, req *http.Request) {
	unimplemented(rw, req)
}

func (*RuntimeApiController) RunAgentSse(rw http.ResponseWriter, req *http.Request) {
	unimplemented(rw, req)
}
