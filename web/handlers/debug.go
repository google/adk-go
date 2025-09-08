package handlers

import (
	"net/http"
)

type DebugApiController struct{}

func (*DebugApiController) TraceDict(rw http.ResponseWriter, req *http.Request) {
	unimplemented(rw, req)
}

func (*DebugApiController) EventGraph(rw http.ResponseWriter, req *http.Request) {
	unimplemented(rw, req)
}
