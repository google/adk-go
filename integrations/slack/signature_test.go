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
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"strconv"
	"strings"
	"testing"
	"time"
)

func sign(secret, ts string, body []byte) string {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte("v0:" + ts + ":"))
	mac.Write(body)
	return "v0=" + hex.EncodeToString(mac.Sum(nil))
}

// Slack's worked example from "Verifying requests from Slack"
// (https://docs.slack.dev/authentication/verifying-requests-from-slack).
func TestVerifySignatureSlackExample(t *testing.T) {
	const (
		secret    = "8f742231b10e8888abcd99yyyzzz85a5"
		timestamp = "1531420618"
		body      = "token=xyzz0WbapA4vBCDEFasx0q6G&team_id=T1DC2JH3J&team_domain=testteamnow&channel_id=G8PSS9T3V&channel_name=foobar&user_id=U2CERLKJA&user_name=roadrunner&command=%2Fwebhook-collect&text=&response_url=https%3A%2F%2Fhooks.slack.com%2Fcommands%2FT1DC2JH3J%2F397700885554%2F96rGlfmibIGlgcZRskXaIFfN&trigger_id=398738663015.47445629121.803a0bc887a14d10d2c447fce8b6703c"
		signature = "v0=a2114d57b48eac39b9ad189dd8316235a7b4a8d21a10bd27519666489c69b503"
	)
	if err := verifySignature(secret, timestamp, []byte(body), signature, time.Unix(1531420618, 0)); err != nil {
		t.Errorf("verifySignature() = %v, want nil", err)
	}
}

func TestVerifySignature(t *testing.T) {
	const secret = "s3cret"
	now := time.Unix(1_800_000_000, 0)
	body := []byte(`{"type":"event_callback"}`)
	tsAt := func(offset time.Duration) string { return strconv.FormatInt(now.Add(offset).Unix(), 10) }
	ts := tsAt(0)

	tests := []struct {
		name      string
		timestamp string
		body      []byte
		signature string
		wantErr   string // substring of the error, empty for success
	}{
		{"valid", ts, body, sign(secret, ts, body), ""},
		{"wrong secret", ts, body, sign("other", ts, body), "mismatch"},
		{"body tampered", ts, []byte(`{"type":"event_callbacK"}`), sign(secret, ts, body), "mismatch"},
		{"uppercase hex", ts, body, "v0=" + strings.ToUpper(strings.TrimPrefix(sign(secret, ts, body), "v0=")), "mismatch"},
		{"299s old", tsAt(-299 * time.Second), body, sign(secret, tsAt(-299*time.Second), body), ""},
		{"300s old", tsAt(-300 * time.Second), body, sign(secret, tsAt(-300*time.Second), body), ""},
		{"301s old", tsAt(-301 * time.Second), body, sign(secret, tsAt(-301*time.Second), body), "window"},
		{"299s ahead", tsAt(299 * time.Second), body, sign(secret, tsAt(299*time.Second), body), ""},
		{"300s ahead", tsAt(300 * time.Second), body, sign(secret, tsAt(300*time.Second), body), ""},
		{"301s ahead", tsAt(301 * time.Second), body, sign(secret, tsAt(301*time.Second), body), "window"},
		{"non-numeric timestamp", "abc", body, sign(secret, "abc", body), "bad X-Slack-Request-Timestamp"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := verifySignature(secret, tt.timestamp, tt.body, tt.signature, now)
			switch {
			case tt.wantErr == "" && err != nil:
				t.Errorf("verifySignature() = %v, want nil", err)
			case tt.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tt.wantErr)):
				t.Errorf("verifySignature() = %v, want an error containing %q", err, tt.wantErr)
			}
		})
	}
}
