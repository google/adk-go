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

import "testing"

func TestBuiltinGate(t *testing.T) {
	const bot = "UBOT"
	botID := func() string { return bot }
	tests := []struct {
		name string
		cfg  gateConfig
		msg  Message
		want Decision
	}{
		{
			name: "disabled refuses even a DM",
			cfg:  gateConfig{disabled: true, botUserID: botID},
			msg:  Message{Channel: "D1", ChannelType: "im"},
			want: Decision{Reason: ReasonKillSwitch},
		},
		{
			name: "channel not in allowlist",
			cfg:  gateConfig{channels: map[string]bool{"C1": true}, botUserID: botID},
			msg:  Message{Channel: "C2", ChannelType: "channel", IsMention: true},
			want: Decision{Reason: ReasonChannel},
		},
		{
			name: "channel in allowlist",
			cfg:  gateConfig{channels: map[string]bool{"C1": true}, botUserID: botID},
			msg:  Message{Channel: "C1", ChannelType: "channel", IsMention: true},
			want: Decision{Allow: true},
		},
		{
			name: "empty allowlist allows any channel",
			cfg:  gateConfig{botUserID: botID},
			msg:  Message{Channel: "C2", ChannelType: "channel", IsMention: true},
			want: Decision{Allow: true},
		},
		{
			name: "channel message without mention",
			cfg:  gateConfig{botUserID: botID},
			msg:  Message{Channel: "C1", ChannelType: "channel"},
			want: Decision{Reason: ReasonNoMention},
		},
		{
			name: "allowUnmentioned allows channel message without mention",
			cfg:  gateConfig{allowUnmentioned: true, botUserID: botID},
			msg:  Message{Channel: "C1", ChannelType: "channel"},
			want: Decision{Allow: true},
		},
		{
			name: "DM without mention",
			cfg:  gateConfig{botUserID: botID},
			msg:  Message{Channel: "D1", ChannelType: "im"},
			want: Decision{Allow: true},
		},
		{
			name: "thread reply to the bot without mention",
			cfg:  gateConfig{botUserID: botID},
			msg:  Message{Channel: "C1", ChannelType: "channel", ThreadTS: "1.1", ParentUser: bot},
			want: Decision{Allow: true},
		},
		{
			name: "requireMentionInThreads refuses thread reply to the bot",
			cfg:  gateConfig{requireMentionInThreads: true, botUserID: botID},
			msg:  Message{Channel: "C1", ChannelType: "channel", ThreadTS: "1.1", ParentUser: bot},
			want: Decision{Reason: ReasonNoMention},
		},
		{
			name: "thread reply to another user without mention",
			cfg:  gateConfig{botUserID: botID},
			msg:  Message{Channel: "C1", ChannelType: "channel", ThreadTS: "1.1", ParentUser: "UOTHER"},
			want: Decision{Reason: ReasonNoMention},
		},
		{
			// The bot ID is not known yet, so an empty ParentUser must not match it.
			name: "root message with unknown bot ID",
			cfg:  gateConfig{botUserID: func() string { return "" }},
			msg:  Message{Channel: "C1", ChannelType: "channel", ThreadTS: "1.1"},
			want: Decision{Reason: ReasonNoMention},
		},
		{
			name: "parent user set but not in a thread",
			cfg:  gateConfig{botUserID: botID},
			msg:  Message{Channel: "C1", ChannelType: "channel", ParentUser: bot},
			want: Decision{Reason: ReasonNoMention},
		},
		{
			name: "kill switch wins over channel allowlist",
			cfg:  gateConfig{disabled: true, channels: map[string]bool{"C1": true}, botUserID: botID},
			msg:  Message{Channel: "C2", ChannelType: "channel"},
			want: Decision{Reason: ReasonKillSwitch},
		},
		{
			name: "channel allowlist wins over mention rule",
			cfg:  gateConfig{channels: map[string]bool{"C1": true}, botUserID: botID},
			msg:  Message{Channel: "C2", ChannelType: "channel"},
			want: Decision{Reason: ReasonChannel},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := builtinGate(tt.cfg, &tt.msg); got != tt.want {
				t.Errorf("builtinGate() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

// builtinGate asks for the bot ID only when the thread exemption is
// evaluated. The dispatcher resolves the ID before calling the gate, so this
// matters only when builtinGate is used on its own.
func TestBuiltinGateBotUserIDLazy(t *testing.T) {
	tests := []struct {
		name      string
		cfg       gateConfig
		msg       Message
		wantCalls int
	}{
		{"DM", gateConfig{}, Message{ChannelType: "im", ThreadTS: "1.1", ParentUser: "UBOT"}, 0},
		{"mention in allowed channel", gateConfig{}, Message{ChannelType: "channel", IsMention: true, ThreadTS: "1.1", ParentUser: "UBOT"}, 0},
		{"allowUnmentioned", gateConfig{allowUnmentioned: true}, Message{ChannelType: "channel", ThreadTS: "1.1", ParentUser: "UBOT"}, 0},
		{"kill switch", gateConfig{disabled: true}, Message{ChannelType: "channel", ThreadTS: "1.1", ParentUser: "UBOT"}, 0},
		{"requireMentionInThreads", gateConfig{requireMentionInThreads: true}, Message{ChannelType: "channel", ThreadTS: "1.1", ParentUser: "UBOT"}, 0},
		{"no parent user", gateConfig{}, Message{ChannelType: "channel", ThreadTS: "1.1"}, 0},
		{"unmentioned thread reply", gateConfig{}, Message{ChannelType: "channel", ThreadTS: "1.1", ParentUser: "UBOT"}, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls := 0
			tt.cfg.botUserID = func() string { calls++; return "UBOT" }
			builtinGate(tt.cfg, &tt.msg)
			if calls != tt.wantCalls {
				t.Errorf("botUserID called %d times, want %d", calls, tt.wantCalls)
			}
		})
	}
}
