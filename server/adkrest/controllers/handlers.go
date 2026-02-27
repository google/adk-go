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

// Package controllers contains the controllers for the ADK REST API.
package controllers

import (
	"encoding/json"
	"log"
	"net/http"
)

// TODO: Move to an internal package, controllers doesn't have to be public API.

// trackingResponseWriter wraps http.ResponseWriter to track if headers have been written.
// This prevents the "superfluous WriteHeader" error when errors occur after streaming starts.
type trackingResponseWriter struct {
	http.ResponseWriter
	headerWritten bool
}

// WriteHeader tracks that headers have been written and delegates to the underlying writer.
func (w *trackingResponseWriter) WriteHeader(statusCode int) {
	if w.headerWritten {
		// Headers already written, log and skip to avoid superfluous WriteHeader
		log.Printf("ADK: Skipping duplicate WriteHeader call (status %d) - headers already sent", statusCode)
		return
	}
	w.headerWritten = true
	w.ResponseWriter.WriteHeader(statusCode)
}

// Write delegates to the underlying writer and marks headers as written
// (Go's http.ResponseWriter implicitly calls WriteHeader(200) on first Write if not called)
func (w *trackingResponseWriter) Write(data []byte) (int, error) {
	w.headerWritten = true
	return w.ResponseWriter.Write(data)
}

// Unwrap returns the underlying ResponseWriter for http.ResponseController compatibility
func (w *trackingResponseWriter) Unwrap() http.ResponseWriter {
	return w.ResponseWriter
}

// EncodeJSONResponse uses the json encoder to write an interface to the http response with an optional status code
func EncodeJSONResponse(i any, status int, w http.ResponseWriter) {
	wHeader := w.Header()
	wHeader.Set("Content-Type", "application/json; charset=UTF-8")

	w.WriteHeader(status)

	if i != nil {
		err := json.NewEncoder(w).Encode(i)
		if err != nil {
			// Only attempt error response if headers haven't been written yet
			if tw, ok := w.(*trackingResponseWriter); ok && tw.headerWritten {
				log.Printf("ADK: Failed to encode JSON response after headers written: %v", err)
				return
			}
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	}
}

type errorHandler func(http.ResponseWriter, *http.Request) error

// NewErrorHandler writes the error code returned from the http handler.
// It uses trackingResponseWriter to prevent "superfluous WriteHeader" errors
// when handlers return errors after already starting to write a response (e.g., SSE streaming).
func NewErrorHandler(fn errorHandler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Wrap the response writer to track if headers have been written
		tw := &trackingResponseWriter{ResponseWriter: w}

		err := fn(tw, r)
		if err != nil {
			// Only write error response if headers haven't been sent yet
			if tw.headerWritten {
				// Headers already written (e.g., during SSE streaming), just log the error
				log.Printf("ADK: Error occurred after response started: %v", err)
				return
			}

			if statusErr, ok := err.(statusError); ok {
				http.Error(tw, statusErr.Error(), statusErr.Status())
			} else {
				http.Error(tw, err.Error(), http.StatusInternalServerError)
			}
		}
	}
}

// Unimplemented returns 501 - Status Not Implemented error
func Unimplemented(rw http.ResponseWriter, req *http.Request) {
	rw.WriteHeader(http.StatusNotImplemented)
}
