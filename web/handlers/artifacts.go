package handlers

import (
	"net/http"
)

type ArtifactsApiController struct{}

func (*ArtifactsApiController) ListArtifacts(rw http.ResponseWriter, req *http.Request) {
	unimplemented(rw, req)
}

func (*ArtifactsApiController) LoadArtifact(rw http.ResponseWriter, req *http.Request) {
	unimplemented(rw, req)
}

func (*ArtifactsApiController) DeleteArtifact(rw http.ResponseWriter, req *http.Request) {
	unimplemented(rw, req)
}
