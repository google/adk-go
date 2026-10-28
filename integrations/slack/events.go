// Copyright 2026 Google LLC
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

package slack

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strconv"
)

// maxBodyBytes is far above any Events API payload, which Slack keeps small.
const maxBodyBytes = 1 << 20

// serveEvents is the Events API transport. It answers as soon as the event
// is queued, because Slack retries a request not answered within three
// seconds.
func (s *Service) serveEvents(w http.ResponseWriter, r *http.Request) {
	if s.secret == "" {
		http.Error(w, "slack: signing secret not configured", http.StatusInternalServerError)
		return
	}
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxBodyBytes))
	if err != nil {
		if _, ok := errors.AsType[*http.MaxBytesError](err); ok {
			http.Error(w, "request body too large", http.StatusRequestEntityTooLarge)
			return
		}
		http.Error(w, "cannot read request body", http.StatusBadRequest)
		return
	}
	ctx := r.Context()
	if err := verifySignature(s.secret, r.Header.Get("X-Slack-Request-Timestamp"), body, r.Header.Get("X-Slack-Signature"), s.now()); err != nil {
		s.logger.WarnContext(ctx, "slack: request rejected", "reason", "auth_failed", "error", err)
		http.Error(w, "invalid signature", http.StatusUnauthorized)
		return
	}
	var env envelope
	if err := json.Unmarshal(body, &env); err != nil {
		http.Error(w, "malformed body", http.StatusBadRequest)
		return
	}
	switch env.Type {
	case "url_verification":
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		_, _ = io.WriteString(w, env.Challenge)
		return
	case "event_callback":
	default:
		s.ignored(r, env.EventID)
		return
	}
	s.bot.record(env.Authorizations)
	// Only the cached ID: the handler itself makes no outbound network call.
	// The only calls it makes that can block are to user-supplied code, the
	// Deduper (through Submit) and the Logger. The worker resolves the ID and
	// redoes mention detection.
	m, err := parseEventCallback(&env, s.bot.cached())
	if err != nil {
		// Slack would redeliver the same bytes, so answering 200 avoids
		// three pointless retries.
		s.logger.WarnContext(ctx, "slack: event not decoded", "event_id", env.EventID, "error", err)
		return
	}
	if m == nil {
		s.ignored(r, env.EventID)
		return
	}
	m.Retry, _ = strconv.Atoi(r.Header.Get("X-Slack-Retry-Num"))
	if err := s.disp.Submit(ctx, m); errors.Is(err, errStopped) {
		http.Error(w, "shutting down", http.StatusServiceUnavailable)
	}
}

func (s *Service) ignored(r *http.Request, eventID string) {
	s.logger.DebugContext(r.Context(), "slack: event ignored", "event_id", eventID, "reason", "unsupported_event")
}
