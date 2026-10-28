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
	"math/rand/v2"
	"regexp"
	"testing"
)

func TestSessionID(t *testing.T) {
	tests := []struct {
		channel, threadTS, want string
	}{
		{"C0BARCEJJ1K", "1784622590.251599", "C0BARCEJJ1K-1784622590_251599"},
		{"D123", "1.2", "D123-1_2"},
		{"C1", "", "C1-"},
		{"c_lower", "a-b", "c_lower-a-b"},
	}
	for _, tt := range tests {
		if got := SessionID(tt.channel, tt.threadTS); got != tt.want {
			t.Errorf("SessionID(%q, %q) = %q, want %q", tt.channel, tt.threadTS, got, tt.want)
		}
	}
}

func TestSessionIDProperty(t *testing.T) {
	const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-:/ "
	valid := regexp.MustCompile(`^[A-Za-z0-9_-]+$`)
	rng := rand.New(rand.NewPCG(1, 2))
	randString := func() string {
		b := make([]byte, rng.IntN(20))
		for i := range b {
			b[i] = alphabet[rng.IntN(len(alphabet))]
		}
		return string(b)
	}
	for range 1000 {
		a, b := randString(), randString()
		got := SessionID(a, b)
		if !valid.MatchString(got) {
			t.Fatalf("SessionID(%q, %q) = %q, does not match %s", a, b, got, valid)
		}
		if again := SessionID(a, b); again != got {
			t.Fatalf("SessionID(%q, %q) not deterministic: %q then %q", a, b, got, again)
		}
	}
}

func TestThreadKey(t *testing.T) {
	if got := (&Message{TS: "1.1"}).threadKey(); got != "1.1" {
		t.Errorf("root threadKey() = %q, want 1.1", got)
	}
	if got := (&Message{TS: "2.2", ThreadTS: "1.1"}).threadKey(); got != "1.1" {
		t.Errorf("reply threadKey() = %q, want 1.1", got)
	}
}

func TestParseEventCallback(t *testing.T) {
	wrap := func(event string) string {
		return `{"type":"event_callback","event_id":"Ev1","team_id":"T1","event":` + event + `}`
	}
	tests := []struct {
		name      string
		body      string
		botUserID string
		wantNil   bool
		check     func(t *testing.T, m *Message)
	}{
		{
			name:      "app_mention in channel",
			body:      wrap(`{"type":"app_mention","user":"U1","text":"<@UBOT> hi","channel":"C1","channel_type":"channel","ts":"1.1"}`),
			botUserID: "UBOT",
			check: func(t *testing.T, m *Message) {
				if !m.IsMention || m.ThreadTS != "" || m.ParentUser != "" {
					t.Errorf("IsMention=%v ThreadTS=%q ParentUser=%q, want true, empty, empty", m.IsMention, m.ThreadTS, m.ParentUser)
				}
				if m.kind != "app_mention" {
					t.Errorf("kind = %q, want app_mention", m.kind)
				}
			},
		},
		{
			name: "file share with text",
			body: wrap(`{"type":"message","subtype":"file_share","user":"U1","text":"what is in this file?","channel":"D1","channel_type":"im","ts":"1.1"}`),
			check: func(t *testing.T, m *Message) {
				if m.Text != "what is in this file?" || m.kind != "message" {
					t.Errorf("Text=%q kind=%q, want the text and message", m.Text, m.kind)
				}
			},
		},
		{
			name:    "file share without text",
			body:    wrap(`{"type":"message","subtype":"file_share","user":"U1","text":"","channel":"D1","channel_type":"im","ts":"1.1"}`),
			wantNil: true,
		},
		{
			name:      "message in im",
			body:      wrap(`{"type":"message","user":"U1","text":"hello","channel":"D1","channel_type":"im","ts":"1.1"}`),
			botUserID: "UBOT",
			check: func(t *testing.T, m *Message) {
				if m.ChannelType != "im" || m.IsMention {
					t.Errorf("ChannelType=%q IsMention=%v, want im, false", m.ChannelType, m.IsMention)
				}
				if m.kind != "message" {
					t.Errorf("kind = %q, want message", m.kind)
				}
			},
		},
		{
			name:      "message mentioning bot",
			body:      wrap(`{"type":"message","user":"U1","text":"hey <@UBOT> look","channel":"C1","channel_type":"channel","ts":"1.1"}`),
			botUserID: "UBOT",
			check: func(t *testing.T, m *Message) {
				if !m.IsMention {
					t.Error("IsMention = false, want true")
				}
				if m.kind != "message" {
					t.Errorf("kind = %q, want message", m.kind)
				}
			},
		},
		{
			name:      "message mentioning bot with unknown bot user",
			body:      wrap(`{"type":"message","user":"U1","text":"hey <@UBOT> look","channel":"C1","channel_type":"channel","ts":"1.1"}`),
			botUserID: "",
			check: func(t *testing.T, m *Message) {
				if m.IsMention {
					t.Error("IsMention = true, want false")
				}
			},
		},
		{
			name:      "message mentioning a user whose id extends the bot id",
			body:      wrap(`{"type":"message","user":"U1","text":"hey <@UBOT2> look","channel":"C1","channel_type":"channel","ts":"1.1"}`),
			botUserID: "UBOT",
			check: func(t *testing.T, m *Message) {
				if m.IsMention {
					t.Error("IsMention = true, want false")
				}
			},
		},
		{
			name:      "empty mention token with unknown bot user",
			body:      wrap(`{"type":"message","user":"U1","text":"stray <@> token","channel":"C1","channel_type":"channel","ts":"1.1"}`),
			botUserID: "",
			check: func(t *testing.T, m *Message) {
				if m.IsMention {
					t.Error("IsMention = true, want false")
				}
			},
		},
		{
			name: "message with bot_id",
			body: wrap(`{"type":"message","bot_id":"B1","text":"x","channel":"C1","channel_type":"channel","ts":"1.1"}`),
			check: func(t *testing.T, m *Message) {
				if !m.FromBot {
					t.Error("FromBot = false, want true")
				}
			},
		},
		{
			name: "message with bot_profile",
			body: wrap(`{"type":"message","user":"U2","bot_profile":{"id":"B1"},"text":"x","channel":"C1","channel_type":"channel","ts":"1.1"}`),
			check: func(t *testing.T, m *Message) {
				if !m.FromBot {
					t.Error("FromBot = false, want true")
				}
			},
		},
		{
			name: "human message is not from bot",
			body: wrap(`{"type":"message","user":"U1","text":"x","channel":"C1","channel_type":"channel","ts":"1.1"}`),
			check: func(t *testing.T, m *Message) {
				if m.FromBot {
					t.Error("FromBot = true, want false")
				}
			},
		},
		{
			name:    "message_changed ignored",
			body:    wrap(`{"type":"message","subtype":"message_changed","channel":"C1","channel_type":"channel","ts":"1.1"}`),
			wantNil: true,
		},
		{
			name: "thread_broadcast kept",
			body: wrap(`{"type":"message","subtype":"thread_broadcast","user":"U1","text":"x","channel":"C1","channel_type":"channel","ts":"2.2","thread_ts":"1.1"}`),
			check: func(t *testing.T, m *Message) {
				if m.ThreadTS != "1.1" {
					t.Errorf("ThreadTS = %q, want 1.1", m.ThreadTS)
				}
			},
		},
		{
			name:    "reaction_added ignored",
			body:    wrap(`{"type":"reaction_added","user":"U1","reaction":"thumbsup"}`),
			wantNil: true,
		},
		{
			name:      "thread reply copies every field",
			body:      wrap(`{"type":"message","user":"U1","text":"reply","channel":"C1","channel_type":"channel","ts":"2.2","thread_ts":"1.1","parent_user_id":"UBOT"}`),
			botUserID: "UBOT",
			check: func(t *testing.T, m *Message) {
				want := Message{
					EventID: "Ev1", TeamID: "T1", Channel: "C1", ChannelType: "channel",
					UserID: "U1", Text: "reply", TS: "2.2", ThreadTS: "1.1", ParentUser: "UBOT",
					kind: "message",
				}
				if *m != want {
					t.Errorf("got %+v, want %+v", *m, want)
				}
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var env envelope
			if err := json.Unmarshal([]byte(tt.body), &env); err != nil {
				t.Fatal(err)
			}
			m, err := parseEventCallback(&env, tt.botUserID)
			if err != nil {
				t.Fatalf("parseEventCallback: %v", err)
			}
			if tt.wantNil {
				if m != nil {
					t.Fatalf("got %+v, want nil", *m)
				}
				return
			}
			if m == nil {
				t.Fatal("got nil, want a message")
			}
			if m.EventID != "Ev1" || m.TeamID != "T1" {
				t.Errorf("EventID=%q TeamID=%q, want Ev1, T1", m.EventID, m.TeamID)
			}
			tt.check(t, m)
		})
	}
}

func TestParseEventCallbackBadEvent(t *testing.T) {
	env := envelope{Type: "event_callback", Event: json.RawMessage(`"not an object"`)}
	if _, err := parseEventCallback(&env, ""); err == nil {
		t.Error("parseEventCallback on a non-object event: got nil error")
	}
}

// Slack sends app_mention without channel_type, so the type comes from the
// channel ID's prefix.
func TestParseEventCallbackInfersChannelType(t *testing.T) {
	for channel, want := range map[string]string{"C1": "channel", "D1": "im", "G1": "group", "X1": ""} {
		env := envelope{EventID: "Ev1", Event: json.RawMessage(`{"type":"app_mention","user":"U1","text":"<@UBOT> hi","channel":"` + channel + `","ts":"1.1"}`)}
		m, err := parseEventCallback(&env, "UBOT")
		if err != nil || m == nil {
			t.Fatalf("parseEventCallback(%s) = %v, %v", channel, m, err)
		}
		if m.ChannelType != want {
			t.Errorf("channel %s: ChannelType = %q, want %q", channel, m.ChannelType, want)
		}
	}
	// Private channels have C IDs, so a channel_type Slack sends wins.
	env := envelope{EventID: "Ev1", Event: json.RawMessage(`{"type":"message","user":"U1","text":"hi","channel":"C1","channel_type":"group","ts":"1.1"}`)}
	if m, err := parseEventCallback(&env, "UBOT"); err != nil || m == nil || m.ChannelType != "group" {
		t.Errorf("parseEventCallback with channel_type group = %+v, %v, want ChannelType group", m, err)
	}
}
