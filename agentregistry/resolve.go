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
	"regexp"
	"strings"

	"github.com/a2aproject/a2a-go/v2/a2a"
)

// Registry protocol-binding wire values.
const (
	bindingJSONRPC  = "JSONRPC"
	bindingHTTPJSON = "HTTP_JSON"
	bindingGRPC     = "GRPC"
)

var (
	reNonIdentifier = regexp.MustCompile(`[^a-zA-Z0-9_]`)
	reUnderscores   = regexp.MustCompile(`_+`)
)

// cleanName converts an arbitrary display name into a valid identifier suitable
// for use as an agent or tool name. It mirrors adk-python's _clean_name:
// non-identifier characters become underscores, runs of underscores collapse,
// leading/trailing underscores are trimmed, and a leading digit is prefixed
// with an underscore.
func cleanName(name string) string {
	s := reNonIdentifier.ReplaceAllString(name, "_")
	s = reUnderscores.ReplaceAllString(s, "_")
	s = strings.Trim(s, "_")
	// A valid identifier cannot start with a digit; prefix one with "_".
	// (After trimming, s can never start with "_".)
	if s != "" && s[0] >= '0' && s[0] <= '9' {
		s = "_" + s
	}
	return s
}

// connectionURI returns the first interface URL matching the optional type and
// transport filters (empty filters match anything), plus the protocol version
// and the interface's A2A transport (mapped from its registry binding via
// [transportBinding]; empty if unrecognized).
//
// Like adk-python's _get_connection_uri, top-level interfaces are treated as a
// trailing type-less protocol.
func connectionURI(protocols []Protocol, ifaces []Interface, protocolType string, transport a2a.TransportProtocol) (uri, version string, binding a2a.TransportProtocol, ok bool) {
	all := protocols
	if len(ifaces) > 0 {
		all = append(append([]Protocol(nil), protocols...), Protocol{Interfaces: ifaces})
	}
	for _, p := range all {
		if protocolType != "" && p.Type != protocolType {
			continue
		}
		for _, i := range p.Interfaces {
			mapped, _ := transportBinding(i.ProtocolBinding)
			if transport != "" && mapped != transport {
				continue
			}
			if i.URL != "" {
				return i.URL, p.ProtocolVersion, mapped, true
			}
		}
	}
	return "", "", "", false
}

// transportBinding maps a registry protocol-binding wire value to the
// corresponding A2A transport protocol.
func transportBinding(wire string) (a2a.TransportProtocol, bool) {
	switch wire {
	case bindingJSONRPC:
		return a2a.TransportProtocolJSONRPC, true
	case bindingHTTPJSON:
		return a2a.TransportProtocolHTTPJSON, true
	case bindingGRPC:
		return a2a.TransportProtocolGRPC, true
	default:
		return "", false
	}
}
