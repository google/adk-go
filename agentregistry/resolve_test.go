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

package agentregistry

import (
	"testing"

	"github.com/a2aproject/a2a-go/v2/a2a"
)

func TestCleanName(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{name: "already valid", in: "my_agent", want: "my_agent"},
		{name: "spaces and dashes", in: "My Cool-Agent", want: "My_Cool_Agent"},
		{name: "collapse underscores", in: "a  b--c", want: "a_b_c"},
		{name: "trim underscores", in: "__agent__", want: "agent"},
		{name: "leading digit", in: "123agent", want: "_123agent"},
		{name: "resource path", in: "projects/p/locations/l/agents/foo", want: "projects_p_locations_l_agents_foo"},
		{name: "empty", in: "", want: ""},
		{name: "only specials", in: "!!!", want: ""},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := cleanName(tc.in); got != tc.want {
				t.Errorf("cleanName(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

func TestConnectionURI(t *testing.T) {
	agentProtocols := []Protocol{
		{
			Type:            "A2A_AGENT",
			ProtocolVersion: "0.3.0",
			Interfaces: []Interface{
				{URL: "https://a.example/jsonrpc", ProtocolBinding: bindingJSONRPC},
				{URL: "https://a.example/http", ProtocolBinding: bindingHTTPJSON},
			},
		},
		{
			Type:       "CUSTOM",
			Interfaces: []Interface{{URL: "https://custom.example", ProtocolBinding: bindingGRPC}},
		},
	}

	tests := []struct {
		name         string
		protocols    []Protocol
		ifaces       []Interface
		protocolType string
		transport    a2a.TransportProtocol
		wantURL      string
		wantVersion  string
		wantBinding  a2a.TransportProtocol
		wantOK       bool
	}{
		{
			name:         "type filter picks first interface",
			protocols:    agentProtocols,
			protocolType: "A2A_AGENT",
			wantURL:      "https://a.example/jsonrpc",
			wantVersion:  "0.3.0",
			wantBinding:  a2a.TransportProtocolJSONRPC,
			wantOK:       true,
		},
		{
			// Filter and result are A2A transports: wire HTTP_JSON -> a2a HTTP+JSON.
			name:        "transport filter picks matching interface (wire->a2a mapping)",
			protocols:   agentProtocols,
			transport:   a2a.TransportProtocolHTTPJSON,
			wantURL:     "https://a.example/http",
			wantVersion: "0.3.0",
			wantBinding: a2a.TransportProtocolHTTPJSON,
			wantOK:      true,
		},
		{
			name:         "type and transport filter",
			protocols:    agentProtocols,
			protocolType: "A2A_AGENT",
			transport:    a2a.TransportProtocolHTTPJSON,
			wantURL:      "https://a.example/http",
			wantVersion:  "0.3.0",
			wantBinding:  a2a.TransportProtocolHTTPJSON,
			wantOK:       true,
		},
		{
			name:        "no filters returns first",
			protocols:   agentProtocols,
			wantURL:     "https://a.example/jsonrpc",
			wantVersion: "0.3.0",
			wantBinding: a2a.TransportProtocolJSONRPC,
			wantOK:      true,
		},
		{
			name:        "top-level interfaces treated as typeless protocol",
			ifaces:      []Interface{{URL: "https://top.example", ProtocolBinding: bindingJSONRPC}},
			transport:   a2a.TransportProtocolJSONRPC,
			wantURL:     "https://top.example",
			wantBinding: a2a.TransportProtocolJSONRPC,
			wantOK:      true,
		},
		{
			name:         "type filter excludes typeless top-level interfaces",
			ifaces:       []Interface{{URL: "https://top.example", ProtocolBinding: bindingJSONRPC}},
			protocolType: "A2A_AGENT",
			wantOK:       false,
		},
		{
			// Unrecognized binding -> empty transport (Python's None), still returned unfiltered.
			name:        "unknown binding returns empty transport",
			protocols:   []Protocol{{ProtocolVersion: "1.0", Interfaces: []Interface{{URL: "https://x.example", ProtocolBinding: "WEIRD"}}}},
			wantURL:     "https://x.example",
			wantVersion: "1.0",
			wantBinding: "",
			wantOK:      true,
		},
		{
			name:      "no match returns not ok",
			protocols: agentProtocols,
			transport: a2a.TransportProtocol("UNKNOWN"),
			wantOK:    false,
		},
		{
			name:      "empty returns not ok",
			protocols: nil,
			wantOK:    false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			gotURL, gotVer, gotBind, gotOK := connectionURI(tc.protocols, tc.ifaces, tc.protocolType, tc.transport)
			if gotOK != tc.wantOK || gotURL != tc.wantURL || gotVer != tc.wantVersion || gotBind != tc.wantBinding {
				t.Errorf("connectionURI() = (%q, %q, %q, %t), want (%q, %q, %q, %t)",
					gotURL, gotVer, gotBind, gotOK, tc.wantURL, tc.wantVersion, tc.wantBinding, tc.wantOK)
			}
		})
	}
}

func TestTransportBinding(t *testing.T) {
	tests := []struct {
		in     string
		want   a2a.TransportProtocol
		wantOK bool
	}{
		{in: "JSONRPC", want: a2a.TransportProtocolJSONRPC, wantOK: true},
		{in: "HTTP_JSON", want: a2a.TransportProtocolHTTPJSON, wantOK: true},
		{in: "GRPC", want: a2a.TransportProtocolGRPC, wantOK: true},
		{in: "http+json", wantOK: false},
		{in: "", wantOK: false},
	}
	for _, tc := range tests {
		t.Run(tc.in, func(t *testing.T) {
			got, ok := transportBinding(tc.in)
			if ok != tc.wantOK || got != tc.want {
				t.Errorf("transportBinding(%q) = (%q, %t), want (%q, %t)", tc.in, got, ok, tc.want, tc.wantOK)
			}
		})
	}
}
