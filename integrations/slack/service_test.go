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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/session"

	"google.golang.org/adk/integrations/slack/internal/slackapitest"
)

func TestNewValidates(t *testing.T) {
	for _, tt := range []struct {
		field string
		cfg   Config
	}{
		{"BotToken", Config{Backend: newFakeBackend()}},
		{"Backend", Config{BotToken: testBotToken}},
		{"MaxConcurrentRuns", Config{BotToken: testBotToken, Backend: newFakeBackend(), MaxConcurrentRuns: -1}},
		{"MaxQueued", Config{BotToken: testBotToken, Backend: newFakeBackend(), MaxQueued: -1}},
	} {
		if _, err := New(t.Context(), tt.cfg); err == nil || !strings.Contains(err.Error(), tt.field) {
			t.Errorf("New() with a bad %s = %v, want an error naming it", tt.field, err)
		}
	}
}

func TestGatesRefuse(t *testing.T) {
	const root = "1700000000.000100"
	quiet := func(context.Context, *Message) (Decision, error) { return Decision{Reason: "quiet_hours"}, nil }
	broken := func(context.Context, *Message) (Decision, error) { return Decision{}, errors.New("policy store down") }
	tests := []struct {
		name   string
		mut    func(*Config)
		body   []byte
		reason string
	}{
		{"kill switch", func(c *Config) { c.Disabled = true }, mention("E1", root, gateText), ReasonKillSwitch},
		{"channel allowlist", func(c *Config) { c.Channels = []string{"C2"} }, mention("E1", root, gateText), ReasonChannel},
		{"unmentioned channel message", func(*Config) {}, callback("E1", slackEvent("message", "C1", root, gateText)), ReasonNoMention},
		{"thread started by someone else", func(*Config) {}, callback("E1", slackEvent("message", "C1", "1700000000.000200", gateText, "thread_ts", root, "parent_user_id", "U2")), ReasonNoMention},
		{"bot thread with RequireMentionInThreads", func(c *Config) { c.RequireMentionInThreads = true }, callback("E1", slackEvent("message", "C1", "1700000000.000200", gateText, "thread_ts", root, "parent_user_id", "UBOT")), ReasonNoMention},
		{"policy refusal", func(c *Config) { c.Policy = quiet }, mention("E1", root, gateText), "quiet_hours"},
		{"policy error", func(c *Config) { c.Policy = broken }, mention("E1", root, gateText), ReasonPolicyError},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := newTestService(t, tt.mut)
			if rec := e.post(tt.body); rec.Code != http.StatusOK {
				t.Fatalf("code = %d, want 200", rec.Code)
			}
			e.waitLog(t, `msg="slack: message not answered" reason=`+tt.reason+" channel=C1 event_id=E1")
			e.noCall(t, 50*time.Millisecond)
		})
	}
}

func TestPolicyNotCalledAfterBuiltinRefusal(t *testing.T) {
	var calls atomic.Int32
	e := newTestService(t, func(c *Config) {
		c.Disabled = true
		c.Policy = func(context.Context, *Message) (Decision, error) { calls.Add(1); return Decision{Allow: true}, nil }
	})
	e.post(mention("E1", "1700000000.000100", "hi"))
	e.waitLog(t, "reason="+ReasonKillSwitch)
	if n := calls.Load(); n != 0 {
		t.Errorf("Policy called %d times, want 0", n)
	}
}

func TestPolicyAllows(t *testing.T) {
	e := newTestService(t, func(c *Config) {
		c.Policy = func(_ context.Context, m *Message) (Decision, error) { return Decision{Allow: m.UserID == "U1"}, nil }
	})
	e.post(mention("E1", "1700000000.000100", "hi"))
	close(e.waitCall(t).release)
}

func TestUnmentionedTurnsThatRun(t *testing.T) {
	const root = "1700000000.000100"
	tests := map[string][]byte{
		"DM":                     dm("E1", root, "hi"),
		"DM mentioning the bot":  dm("E1", root, "<@UBOT> hi"),
		"thread the bot started": callback("E1", slackEvent("message", "C1", "1700000000.000200", "hi", "thread_ts", root, "parent_user_id", "UBOT")),
	}
	for name, body := range tests {
		t.Run(name, func(t *testing.T) {
			e := newTestService(t)
			e.post(body)
			close(e.waitCall(t).release)
		})
	}
}

// eventNoAuth is an event_callback without authorizations, so the service
// must learn its own user ID from auth.test.

func TestBotUserIDFromAuthTest(t *testing.T) {
	e := newTestService(t)
	const root = "1700000000.000100"
	for i, id := range []string{"E1", "E2"} {
		e.post(eventNoAuth(id, slackEvent("message", "C1", fmt.Sprintf("1700000000.00020%d", i), "hi", "thread_ts", root, "parent_user_id", "UBOT")))
		close(e.waitCall(t).release)
	}
	if n := countMethod(e.api.Methods(), "auth.test"); n != 1 {
		t.Errorf("auth.test called %d times, want 1", n)
	}
}

func TestAuthTestFailureBacksOff(t *testing.T) {
	e := newTestService(t)
	e.api.Fail = map[string][]slackapitest.Response{"auth.test": {{APIError: "fatal_error"}, {APIError: "fatal_error"}, {APIError: "fatal_error"}}}
	for i := range 3 {
		e.post(eventNoAuth(fmt.Sprintf("E%d", i), slackEvent("message", "D1", fmt.Sprintf("1700000000.00010%d", i), "hi")))
		close(e.waitCall(t).release)
	}
	e.drain(t)
	if n := countMethod(e.api.Methods(), "auth.test"); n != 1 {
		t.Errorf("auth.test called %d times, want 1", n)
	}
}

// Without the bot's ID, an unmentioned channel message cannot be shown to be
// in the bot's thread, so it is refused.

func TestUnknownBotUserIDRefusesThreadReply(t *testing.T) {
	e := newTestService(t)
	e.api.Fail = map[string][]slackapitest.Response{"auth.test": {{APIError: "fatal_error"}}}
	e.post(eventNoAuth("E1", slackEvent("message", "C1", "1700000000.000200", gateText, "thread_ts", "1700000000.000100", "parent_user_id", "UBOT")))
	e.waitLog(t, "reason="+ReasonNoMention+" channel=C1 event_id=E1")
	e.noCall(t, 50*time.Millisecond)
}

// The handler has no bot ID when the plain-message copy arrives first, so
// the worker must find the mention. In the bot's thread the copy would
// otherwise pass the gate and be answered twice.

func TestMentionEchoWithoutAuthorizations(t *testing.T) {
	e := newTestService(t)
	const root, ts = "1700000000.000100", "1700000000.000200"
	thread := []string{"thread_ts", root, "parent_user_id", "UBOT"}
	e.post(eventNoAuth("E1", slackEvent("message", "C1", ts, "<@UBOT> hi", thread...)))
	e.post(eventNoAuth("E2", slackEvent("app_mention", "C1", ts, "<@UBOT> hi", thread...)))
	close(e.waitCall(t).release)
	e.flush(t)
}

// Authorizations are what Slack says about this very event, so they replace
// an ID learned earlier.

func TestAuthorizationsReplaceCachedBotUserID(t *testing.T) {
	e := newTestService(t)
	e.api.BotUserID = "UOLD"
	const root = "1700000000.000100"
	e.post(eventNoAuth("E1", slackEvent("message", "D1", root, "hi")))
	close(e.waitCall(t).release)
	e.post(mention("E2", "1700000000.000200", "hi"))
	close(e.waitCall(t).release)
	e.post(eventNoAuth("E3", slackEvent("message", "C1", "1700000000.000300", "hi", "thread_ts", root, "parent_user_id", "UBOT")))
	close(e.waitCall(t).release)
}

// Only a bot authorization with a user ID names the bot.

func TestBotUserIDFromAuthorizationList(t *testing.T) {
	e := newTestService(t)
	e.api.BotUserID = "UOLD"
	b, err := json.Marshal(map[string]any{
		"type": "event_callback", "event_id": "E1", "team_id": "T1",
		"authorizations": []any{
			map[string]any{"user_id": "", "is_bot": true},
			map[string]any{"user_id": "UHUMAN", "is_bot": false},
			map[string]any{"user_id": "UBOT", "is_bot": true},
		},
		"event": slackEvent("message", "C1", "1700000000.000200", "hi", "thread_ts", "1700000000.000100", "parent_user_id", "UBOT"),
	})
	if err != nil {
		t.Fatal(err)
	}
	e.post(b)
	close(e.waitCall(t).release)
}

func TestNoAuthTestWithAuthorizations(t *testing.T) {
	e := newTestService(t)
	e.post(callback("E1", slackEvent("message", "C1", "1700000000.000200", "hi", "thread_ts", "1700000000.000100", "parent_user_id", "UBOT")))
	close(e.waitCall(t).release)
	e.drain(t)
	if slices.Contains(e.api.Methods(), "auth.test") {
		t.Errorf("auth.test called although the event carried authorizations: %q", e.api.Methods())
	}
}

func TestDelivery(t *testing.T) {
	partial := textEvent("draft")
	partial.Partial = true
	final := textEvent("**one**", " \n ", "two")
	final.Content.Parts = append(final.Content.Parts[:1], append([]*genai.Part{nil, {Text: "pondering", Thought: true}}, final.Content.Parts[1:]...)...)
	tests := []struct {
		name string
		mut  func(*Config)
		want []slackapitest.Call
	}{
		{"default", func(*Config) {}, []slackapitest.Call{post(1, "_Thinking…_"), update(1, "*one*"), post(2, "two")}},
		{"custom placeholder", func(c *Config) { c.Placeholder = "wait" }, []slackapitest.Call{post(1, "wait"), update(1, "*one*"), post(2, "two")}},
		{"no placeholder", func(c *Config) { c.NoPlaceholder = true }, []slackapitest.Call{post(1, "*one*"), post(2, "two")}},
		{"markdown conversion disabled", func(c *Config) { c.DisableMarkdownConversion = true }, []slackapitest.Call{post(1, "_Thinking…_"), update(1, "**one**"), post(2, "two")}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := newTestService(t, tt.mut)
			e.backend.events = []*session.Event{partial, nil, {}, final}
			e.post(mention("E1", testThread, "hi"))
			e.waitCall(t)
			e.drain(t)
			if got := e.api.Snapshot(); !slices.Equal(got, tt.want) {
				t.Errorf("calls =\n%+v\nwant\n%+v", got, tt.want)
			}
		})
	}
}

func TestBackendReceivesTextWithoutOwnMention(t *testing.T) {
	e := newTestService(t)
	e.post(callback("E1", slackEvent("app_mention", "C1", "1700000000.000100", "<@UBOT> hello <@UOTHER>")))
	c := e.waitCall(t)
	close(c.release)
	if c.text != "hello <@UOTHER>" {
		t.Errorf("backend text = %q, want %q", c.text, "hello <@UOTHER>")
	}
}

func TestUserAndSessionIDs(t *testing.T) {
	const root, reply = "1700000000.000100", "1700000000.000200"
	tests := []struct {
		name        string
		mut         func(*Config)
		body        []byte
		wantUser    string
		wantSession string
	}{
		{"default user, root", func(*Config) {}, mention("E1", root, "hi"), "slack:T1:C1", SessionID("C1", root)},
		{"thread reply shares the root's session", func(*Config) {}, mention("E1", reply, "hi", "thread_ts", root), "slack:T1:C1", SessionID("C1", root)},
		{"UserIDFunc", func(c *Config) { c.UserIDFunc = func(m *Message) string { return "u-" + m.UserID } }, mention("E1", root, "hi"), "u-U1", SessionID("C1", root)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := newTestService(t, tt.mut)
			e.post(tt.body)
			c := e.waitCall(t)
			close(c.release)
			if c.userID != tt.wantUser || c.sessionID != tt.wantSession {
				t.Errorf("Run(userID=%q, sessionID=%q), want (%q, %q)", c.userID, c.sessionID, tt.wantUser, tt.wantSession)
			}
		})
	}
}

func TestTurnFailure(t *testing.T) {
	e := newTestService(t, func(c *Config) { c.ErrorMessage = "turn failed" })
	e.backend.events = []*session.Event{textEvent("partial answer")}
	e.backend.err = errors.New("model unavailable")
	e.post(mention("E1", testThread, "hi"))
	e.waitCall(t)
	e.drain(t)
	want := []slackapitest.Call{post(1, "_Thinking…_"), update(1, "partial answer"), post(2, "turn failed")}
	if got := e.api.Snapshot(); !slices.Equal(got, want) {
		t.Errorf("calls =\n%+v\nwant\n%+v", got, want)
	}
	for _, want := range []string{
		"level=ERROR", `msg="slack: turn failed"`, "event_id=E1", "session_id=" + SessionID("C1", testThread),
		"backend=*slack.fakeBackend", "duration=", `error="model unavailable"`,
	} {
		if !strings.Contains(e.log.String(), want) {
			t.Errorf("log has no %q:\n%s", want, e.log.String())
		}
	}
}

func TestBotMessagesIgnored(t *testing.T) {
	e := newTestService(t)
	e.post(callback("E1", slackEvent("message", "D1", "1700000000.000100", "hi", "bot_id", "B1")))
	want := `level=DEBUG msg="slack: message not answered" reason=` + ReasonFromBot + " channel=D1 event_id=E1"
	if !strings.Contains(e.log.String(), want) {
		t.Errorf("log has no %q:\n%s", want, e.log.String())
	}
	e.flush(t)
}

func TestPolicyRefusalWithoutReason(t *testing.T) {
	e := newTestService(t, func(c *Config) {
		c.Policy = func(context.Context, *Message) (Decision, error) { return Decision{}, nil }
	})
	e.post(mention("E1", "1700000000.000100", gateText))
	e.waitLog(t, "reason="+ReasonPolicyRefused+" channel=C1 event_id=E1")
	e.noCall(t, 50*time.Millisecond)
}

func TestBotIDLookupPanic(t *testing.T) {
	e := newTestService(t, func(c *Config) { c.HTTPClient = &http.Client{Transport: panicTransport{}} })
	e.post(eventNoAuth("E1", slackEvent("message", "D1", "1700000000.000100", gateText)))
	e.waitLog(t, `msg="slack: turn panicked" stage=bot_id event_id=E1`)
	e.noCall(t, 50*time.Millisecond)
}

func TestDeduperErrorAnswers(t *testing.T) {
	e := newTestService(t, func(c *Config) { c.Deduper = failingDeduper{} })
	e.post(mention("E1", "1700000000.000100", "hi"))
	close(e.waitCall(t).release)
	for _, want := range []string{"level=WARN", `msg="slack: dedup failed"`, "event_id=E1", `error="store down"`} {
		if !strings.Contains(e.log.String(), want) {
			t.Errorf("log has no %q:\n%s", want, e.log.String())
		}
	}
}

// A bare mention strips to empty text, which the session would store and
// llmagent would resend with each later turn of the thread.

func TestBareMentionSendsOriginalText(t *testing.T) {
	e := newTestService(t)
	e.post(callback("E1", slackEvent("app_mention", "C1", "1700000000.000100", "<@UBOT>")))
	c := e.waitCall(t)
	close(c.release)
	if c.text != "<@UBOT>" {
		t.Errorf("backend text = %q, want %q", c.text, "<@UBOT>")
	}
}

func TestBareMentionStoresNonEmptyUserContent(t *testing.T) {
	sessions := session.InMemoryService()
	b, err := NewLocalBackend(LocalBackendConfig{AppName: "app", Agent: newPongAgent(t, &pongModel{}), SessionService: sessions})
	if err != nil {
		t.Fatalf("NewLocalBackend: %v", err)
	}
	e := newTestService(t, func(c *Config) { c.Backend = b })
	const ts = "1700000000.000100"
	e.post(callback("E1", slackEvent("app_mention", "C1", ts, "<@UBOT>")))
	e.waitAPI(t, hasText("pong"))
	e.drain(t)

	resp, err := sessions.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "slack:T1:C1", SessionID: SessionID("C1", ts)})
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	users := 0
	for ev := range resp.Session.Events().All() {
		if ev.Author != "user" {
			continue
		}
		users++
		ok := false
		if ev.Content != nil {
			for _, p := range ev.Content.Parts {
				if p != nil && p.Text != "" {
					ok = true
				}
			}
		}
		if !ok {
			t.Errorf("stored user event %s has no part with text", ev.ID)
		}
	}
	if users != 1 {
		t.Errorf("session has %d user events, want 1", users)
	}
}

func TestAllowUnmentionedAnswersChannelMessage(t *testing.T) {
	e := newTestService(t, func(c *Config) { c.AllowUnmentioned = true })
	e.post(callback("E1", slackEvent("message", "C1", "1700000000.000100", "hi")))
	close(e.waitCall(t).release)
}

func countMethod(methods []string, method string) int {
	n := 0
	for _, m := range methods {
		if m == method {
			n++
		}
	}
	return n
}

type panicTransport struct{}

func (panicTransport) RoundTrip(*http.Request) (*http.Response, error) { panic("transport exploded") }

type failingDeduper struct{}

func (failingDeduper) Seen(context.Context, string) (bool, error) {
	return true, errors.New("store down")
}
