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

package remoteagent

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/a2aproject/a2a-go/v2/a2a"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/auth"
	"google.golang.org/adk/v2/internal/googlemtls/googlemtlstest"
	"google.golang.org/adk/v2/session"
)

// offline makes http.DefaultTransport fail every request for the rest of the
// test, so a request that misses the mTLS path fails here instead of going to
// the network. The mTLS transport then starts from the settings net/http
// creates http.DefaultTransport with.
func offline(t *testing.T) {
	orig := http.DefaultTransport
	http.DefaultTransport = offlineTransport{}
	t.Cleanup(func() { http.DefaultTransport = orig })
}

type offlineTransport struct{}

func (offlineTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return nil, fmt.Errorf("offline: request to %s reached http.DefaultTransport", req.URL.Host)
}

// The clients NewA2A builds present the environment's client certificate to a
// Google API host, on its mTLS name, with Auth and without it. The stand-in
// server is not an A2A server, so the run fails after the request arrives.
func TestNewA2APresentsClientCertificateToGoogleAPIs(t *testing.T) {
	const rpcURL = "https://agent.googleapis.com/rpc"
	card := &a2a.AgentCard{
		Name: "remote",
		SupportedInterfaces: []*a2a.AgentInterface{
			a2a.NewAgentInterface(rpcURL, a2a.TransportProtocolJSONRPC),
		},
	}
	for _, tc := range []struct {
		name string
		cfg  A2AConfig
	}{
		{name: "no Auth", cfg: A2AConfig{Name: "a2a", AgentCard: card}},
		{name: "Auth", cfg: A2AConfig{Name: "a2a", AgentCard: card, Auth: auth.StaticToken("tok")}},
		// The card fetch shares the Auth client.
		{name: "Auth, fetched card", cfg: A2AConfig{Name: "a2a", AgentCardProvider: NewAgentCardProvider("https://agent.googleapis.com"), Auth: auth.StaticToken("tok")}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("GOOGLE_API_USE_MTLS_ENDPOINT", "")
			srv := googlemtlstest.NewServer(t)
			srv.Install(t)
			offline(t)
			a, err := NewA2A(tc.cfg)
			if err != nil {
				t.Fatalf("NewA2A() error = %v", err)
			}
			ictx := newInvocationContextWithStreamingMode(t, []*session.Event{newUserHello()}, agent.StreamingModeNone)
			events, runErr := runAndCollect(ictx, a)
			select {
			case got := <-srv.Requests:
				want := googlemtlstest.Request{Host: "agent.mtls.googleapis.com", ServerName: "agent.mtls.googleapis.com", PeerCerts: 1, Leaf: srv.Leaf()}
				if got != want {
					t.Errorf("server saw %+v, want %+v", got, want)
				}
			default:
				// The server records a request before it answers, so one that
				// arrived is in the channel by the time the run returns.
				t.Fatalf("no request reached the mTLS server; run error = %v, events = %v", runErr, firstErrorEvent(events))
			}
		})
	}
}
