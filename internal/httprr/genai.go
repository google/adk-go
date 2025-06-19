package httprr

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// NewGeminiTransportForTesting returns the genai.ClientConfig configured for record and replay.
func NewGeminiTransportForTesting(rrfile string) (http.RoundTripper, error) {
	rr, err := Open(rrfile, http.DefaultTransport)
	if err != nil {
		return nil, fmt.Errorf("httprr.Open(%q) failed: %w", rrfile, err)
	}
	rr.ScrubReq(scrubGeminiRequest)
	return rr, nil
}

func scrubGeminiRequest(req *http.Request) error {
	delete(req.Header, "x-goog-api-key")    // genai does not canonicalize
	req.Header.Del("X-Goog-Api-Key")        // in case it starts
	delete(req.Header, "x-goog-api-client") // contains version numbers
	req.Header.Del("X-Goog-Api-Client")
	delete(req.Header, "user-agent") // contains google-genai-sdk and gl-go version numbers
	req.Header.Del("User-Agent")

	if ctype := req.Header.Get("Content-Type"); ctype == "application/json" || strings.HasPrefix(ctype, "application/json;") {
		// Canonicalize JSON body.
		// google.golang.org/protobuf/internal/encoding.json
		// goes out of its way to randomize the JSON encodings
		// of protobuf messages by adding or not adding spaces
		// after commas. Derandomize by compacting the JSON.
		b := req.Body.(*Body)
		var buf bytes.Buffer
		if err := json.Compact(&buf, b.Data); err == nil {
			b.Data = buf.Bytes()
		}
	}
	return nil
}
