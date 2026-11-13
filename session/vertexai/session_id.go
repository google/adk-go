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

package vertexai

import (
	"fmt"
	"regexp"
	"strings"
)

// sessionIDPattern is what a short session id must match. It keeps the id
// inside one segment of the session's resource name, as adk-python's
// _SESSION_ID_PATTERN does (vertex_ai_session_service.py).
var sessionIDPattern = regexp.MustCompile(`^[A-Za-z0-9_-]+$`)

// shortSessionID returns the last segment of a ".../sessions/<id>" resource
// name, and id unchanged when it is not one. When the name has a
// ".../reasoningEngines/<engine>/sessions/<id>" tail and engineID is not
// empty, it returns an error if <engine> is not engineID. It ports adk-python's
// _extract_short_session_id.
func shortSessionID(id, engineID string) (string, error) {
	parts := strings.Split(id, "/")
	if len(parts) < 2 || parts[len(parts)-2] != "sessions" {
		return id, nil
	}
	if len(parts) >= 4 && parts[len(parts)-4] == "reasoningEngines" && engineID != "" &&
		parts[len(parts)-3] != engineID {
		return "", fmt.Errorf("session resource name mismatch: session belongs to reasoningEngine %q, but service is configured for %q",
			parts[len(parts)-3], engineID)
	}
	return parts[len(parts)-1], nil
}

// checkSessionID returns an error unless id matches sessionIDPattern. It ports
// adk-python's _validate_session_id.
func checkSessionID(id string) error {
	if !sessionIDPattern.MatchString(id) {
		return fmt.Errorf("invalid session_id %q: must match %s", id, sessionIDPattern)
	}
	return nil
}

// resolveSessionID returns the short session id that raw names for appName's
// reasoning engine, or an error, before any RPC, when raw is a resource name
// of another engine or its short id does not match sessionIDPattern.
func (c *vertexAiClient) resolveSessionID(appName, raw string) (string, error) {
	engineID, err := c.getReasoningEngineID(appName)
	if err != nil {
		return "", err
	}
	id, err := shortSessionID(raw, engineID)
	if err != nil {
		return "", err
	}
	if err := checkSessionID(id); err != nil {
		return "", err
	}
	return id, nil
}
