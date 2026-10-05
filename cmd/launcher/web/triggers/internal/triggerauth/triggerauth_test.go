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

package triggerauth

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/server/authn"
)

func TestConfigCarriesFlagValues(t *testing.T) {
	f := &Flags{audience: "https://svc.run.app", serviceAccounts: "a@p.iam.gserviceaccount.com, b@p.iam.gserviceaccount.com"}

	got, err := f.config()
	if err != nil {
		t.Fatalf("config() = %v, want no error", err)
	}
	want := &authn.GoogleOIDCConfig{
		Audience:               "https://svc.run.app",
		AllowedServiceAccounts: []string{"a@p.iam.gserviceaccount.com", "b@p.iam.gserviceaccount.com"},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("config() mismatch (-want +got):\n%s", diff)
	}
}

func TestConfigNilWithoutFlags(t *testing.T) {
	got, err := (&Flags{}).config()
	if err != nil || got != nil {
		t.Errorf("config() = %v, %v, want nil, nil", got, err)
	}
}

func TestConfigRejectsPaddedAudience(t *testing.T) {
	for _, aud := range []string{" https://svc.run.app", "https://svc.run.app ", "https://svc.run.app\t", " "} {
		f := &Flags{audience: aud, serviceAccounts: "a@p.iam.gserviceaccount.com"}
		if _, err := f.config(); err == nil {
			t.Errorf("config() with audience %q succeeded, want an error", aud)
		}
	}
}
