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

package mcptoolset

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func TestIsReservedMetaKey(t *testing.T) {
	tests := []struct {
		key  string
		want bool
	}{
		{key: mcp.MetaKeyServerInfo, want: true},
		{key: mcp.MetaKeyClientInfo, want: true},
		{key: mcp.MetaKeyProtocolVersion, want: true},
		{key: mcp.MetaKeyClientCapabilities, want: true},
		{key: mcp.MetaKeySubscriptionID, want: true},
		{key: "io.modelcontextprotocol/anythingElse", want: true},
		{key: "dev.mcp/key", want: true},
		{key: "org.modelcontextprotocol.api/key", want: true},
		{key: "com.mcp.tools/key", want: true},
		{key: "com.example.mcp/key", want: false},
		{key: "com.example/auth", want: false},
		{key: "com.giantswarm.muster/authChallenge", want: false},
		{key: "mcp/key", want: false},
		{key: "bare", want: false},
		// Unprefixed keys the protocol reserves as an exception to the prefix rule.
		{key: "progressToken", want: true},
		{key: "traceparent", want: true},
		{key: "tracestate", want: true},
		{key: "baggage", want: true},
		// A key name holds no slash, so the prefix ends at the first one.
		{key: "io.modelcontextprotocol/a/b", want: true},
	}

	for _, tc := range tests {
		if got := isReservedMetaKey(tc.key); got != tc.want {
			t.Errorf("isReservedMetaKey(%q) = %v, want %v", tc.key, got, tc.want)
		}
	}
}

func TestServerMeta(t *testing.T) {
	tests := []struct {
		name string
		meta mcp.Meta
		want map[string]any
	}{
		{
			name: "nil meta",
			meta: nil,
			want: nil,
		},
		{
			name: "only reserved keys",
			meta: mcp.Meta{mcp.MetaKeyServerInfo: &mcp.Implementation{Name: "s"}},
			want: nil,
		},
		{
			name: "reserved keys dropped alongside server keys",
			meta: mcp.Meta{
				mcp.MetaKeyServerInfo:  &mcp.Implementation{Name: "s"},
				"progressToken":        "p1",
				"traceparent":          "00-trace-span-01",
				"com.example/auth":     map[string]any{"url": "https://idp.example.com/login"},
				"com.example.mcp/note": "kept",
			},
			want: map[string]any{
				"com.example/auth":     map[string]any{"url": "https://idp.example.com/login"},
				"com.example.mcp/note": "kept",
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if diff := cmp.Diff(tc.want, serverMeta(tc.meta)); diff != "" {
				t.Errorf("serverMeta() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
