package handlers

import (
	"net/http"
)

type AppsApiController struct{}

func (*AppsApiController) ListApps(rw http.ResponseWriter, req *http.Request) {
	unimplemented(rw, req)
}
