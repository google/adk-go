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

package redis

import (
	"strings"
	"testing"
	"time"
)

func TestNewService_InvalidURI_RedactsCredentials(t *testing.T) {
	svc, err := NewService(Config{URI: "redis://:secret_pw@localhost:notaport/0?token=secret_tok"})
	if err == nil {
		t.Fatal("NewService() error = nil, want an error for an unparsable URI")
	}
	if svc != nil {
		t.Errorf("NewService() service = %v, want nil on error", svc)
	}
	msg := err.Error()
	if !strings.Contains(msg, "redis: invalid connection URI") {
		t.Errorf("NewService() error does not name the problem: got %d bytes without %q", len(msg), "redis: invalid connection URI")
	}
	for _, secret := range []string{"secret_pw", "secret_tok"} {
		if strings.Contains(msg, secret) {
			t.Errorf("NewService() error contains the credential %q", secret)
		}
	}
}

func TestNewService_NegativePortOrDB(t *testing.T) {
	tests := []struct {
		name     string
		cfg      Config
		wantText string
	}{
		{name: "negative port", cfg: Config{Port: -1}, wantText: "Port"},
		{name: "negative db", cfg: Config{DB: -1}, wantText: "DB"},
		{name: "ttl not whole seconds", cfg: Config{TTL: 1500 * time.Millisecond}, wantText: "TTL"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			svc, err := NewService(tc.cfg)
			if err == nil {
				t.Fatalf("NewService(%+v) error = nil, want an error", tc.cfg)
			}
			if svc != nil {
				t.Errorf("NewService(%+v) service = %v, want nil on error", tc.cfg, svc)
			}
			if !strings.Contains(err.Error(), tc.wantText) {
				t.Errorf("NewService(%+v) error = %q, want it to mention %q", tc.cfg, err, tc.wantText)
			}
		})
	}

	t.Run("whole-second ttl", func(t *testing.T) {
		svc, err := NewService(Config{TTL: 2 * time.Second})
		if err != nil {
			t.Fatalf("NewService(TTL: 2s) error = %v, want nil", err)
		}
		if err := svc.Close(); err != nil {
			t.Errorf("Close() error = %v", err)
		}
	})
}
