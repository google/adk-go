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
	"errors"
	"strconv"
	"time"
)

// verifySignature checks Slack's request signature: hex HMAC-SHA256 over
// "v0:<timestamp>:<body>" with the signing secret, prefixed "v0=". Requests
// older than five minutes are rejected to blunt replay.
func verifySignature(secret, timestamp string, body []byte, signature string, now time.Time) error {
	ts, err := strconv.ParseInt(timestamp, 10, 64)
	if err != nil {
		return errors.New("slack: bad X-Slack-Request-Timestamp")
	}
	if d := now.Unix() - ts; d > 300 || d < -300 {
		return errors.New("slack: request timestamp outside the five minute window")
	}
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte("v0:" + timestamp + ":"))
	mac.Write(body)
	want := "v0=" + hex.EncodeToString(mac.Sum(nil))
	if !hmac.Equal([]byte(want), []byte(signature)) {
		return errors.New("slack: signature mismatch")
	}
	return nil
}
