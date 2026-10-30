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

package credredirect

import (
	"net/url"
	"testing"
)

// TestSameTarget covers which redirects may carry the credential, across the
// scheme and port combinations which the end-to-end tests in mcptoolset and
// remoteagent/v2 do not reach: an upgrade to https, default ports written out
// or left implicit, and schemes that are neither http nor https.
func TestSameTarget(t *testing.T) {
	tests := []struct {
		name     string
		from, to string
		want     bool
	}{
		{name: "identical", from: "https://h/rpc", to: "https://h/other", want: true},
		{name: "default port made explicit", from: "https://h/rpc", to: "https://h:443/rpc", want: true},
		{name: "default http port made explicit", from: "http://h/rpc", to: "http://h:80/rpc", want: true},
		{name: "host case differs", from: "https://Host/rpc", to: "https://host/rpc", want: true},
		{name: "scheme upgraded to https, same explicit port", from: "http://h:8080/rpc", to: "https://h:8080/rpc", want: true},
		{name: "scheme upgraded to https, both ports implicit", from: "http://h/rpc", to: "https://h/rpc", want: true},
		{name: "scheme upgraded to https, explicit default ports", from: "http://h:80/rpc", to: "https://h:443/rpc", want: true},
		{name: "scheme upgraded to https onto a different port", from: "http://h/rpc", to: "https://h:9443/rpc", want: false},
		{name: "scheme downgraded with implicit ports", from: "https://h/rpc", to: "http://h/rpc", want: false},
		{name: "unrelated scheme", from: "https://h/rpc", to: "ftp://h/rpc", want: false},
		// Only http to https counts as an upgrade: either half alone is not one.
		{name: "http to an unrelated scheme", from: "http://h/rpc", to: "ftp://h/rpc", want: false},
		{name: "an unrelated scheme to https", from: "ftp://h/rpc", to: "https://h/rpc", want: false},
		// An explicit default port on one side and an implicit one on the other
		// is still each scheme's own default.
		{name: "scheme upgraded to https, explicit http default to implicit", from: "http://h:80/rpc", to: "https://h/rpc", want: true},
		{name: "scheme downgraded to http", from: "https://h/rpc", to: "http://h/rpc", want: false},
		{name: "different host", from: "https://h/rpc", to: "https://other/rpc", want: false},
		{name: "different port", from: "https://h:8443/rpc", to: "https://h:9443/rpc", want: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			from, err := url.Parse(tc.from)
			if err != nil {
				t.Fatalf("url.Parse(%q) error = %v", tc.from, err)
			}
			to, err := url.Parse(tc.to)
			if err != nil {
				t.Fatalf("url.Parse(%q) error = %v", tc.to, err)
			}
			if got := sameTarget(from, to); got != tc.want {
				t.Errorf("sameTarget(%q, %q) = %v, want %v", tc.from, tc.to, got, tc.want)
			}
		})
	}
}
