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

package version

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"os"
	"regexp"
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/internal/version/versiontest"
)

// The MCP tests compare against TrackingHeader and the Gemini test checks only
// substrings, so only this test pins its form.
func TestTrackingHeader(t *testing.T) {
	versiontest.ClearRuntimeEnv(t)
	re := regexp.MustCompile(`^google-adk/` + regexp.QuoteMeta(Version) + ` gl-go/\S+( \S+)*$`)
	for _, dest := range []*url.URL{nil, googleAPI, otherHost} {
		if got := TrackingHeader(dest); !re.MatchString(got) {
			t.Errorf("TrackingHeader(%v) = %q, want a match for %s", dest, got, re)
		}
	}
}

func TestLanguageTokens(t *testing.T) {
	tests := []struct {
		version string
		want    []string
	}{
		{version: "go1.26.5", want: []string{"gl-go/1.26.5"}},
		{version: "go1.27.3-X:boringcrypto", want: []string{"gl-go/1.27.3-X:boringcrypto"}},
		// A development toolchain.
		{
			version: "devel go1.27-0a1b2c3 Mon Jan 2 15:04:05 2006 +0000",
			want:    []string{"gl-go/devel", "go1.27-0a1b2c3", "Mon", "Jan", "2", "15:04:05", "2006", "+0000"},
		},
	}
	for _, tt := range tests {
		if got := languageTokens(tt.version); !slices.Equal(got, tt.want) {
			t.Errorf("languageTokens(%q) = %q, want %q", tt.version, got, tt.want)
		}
	}
}

func TestTrackingHeaderRuntimeSuffix(t *testing.T) {
	tests := []struct {
		name string
		set  []string
		want string
	}{
		{name: "Agent Engine", set: []string{"GOOGLE_CLOUD_AGENT_ENGINE_ID"}, want: "+remote_reasoning_engine"},
		{name: "Cloud Run service", set: []string{"K_SERVICE"}, want: "+cloud_run"},
		{name: "Cloud Run job", set: []string{"CLOUD_RUN_JOB"}, want: "+cloud_run"},
		{name: "Cloud Run service and job", set: []string{"K_SERVICE", "CLOUD_RUN_JOB"}, want: "+cloud_run"},
		{name: "Kubernetes", set: []string{"KUBERNETES_SERVICE_HOST"}, want: "+k8s"},
		{name: "Cloud Run wins over Kubernetes", set: []string{"K_SERVICE", "KUBERNETES_SERVICE_HOST"}, want: "+cloud_run"},
		{name: "Cloud Run job wins over Kubernetes", set: []string{"CLOUD_RUN_JOB", "KUBERNETES_SERVICE_HOST"}, want: "+cloud_run"},
		{name: "Agent Engine wins over Cloud Run", set: []string{"GOOGLE_CLOUD_AGENT_ENGINE_ID", "K_SERVICE"}, want: "+remote_reasoning_engine"},
		{name: "Agent Engine wins over Kubernetes", set: []string{"GOOGLE_CLOUD_AGENT_ENGINE_ID", "KUBERNETES_SERVICE_HOST"}, want: "+remote_reasoning_engine"},
		{name: "Agent Engine wins over all", set: versiontest.RuntimeEnvVars(), want: "+remote_reasoning_engine"},
		{name: "no runtime"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			versiontest.ClearRuntimeEnv(t)
			for _, name := range tt.set {
				t.Setenv(name, "some-value")
			}
			want := "google-adk/" + Version + tt.want
			if got := adkToken(TrackingHeader(googleAPI)); got != want {
				t.Errorf("TrackingHeader(%v) starts with %q, want %q", googleAPI, got, want)
			}
			// The value of a variable is never sent, only the suffix.
			if got := TrackingHeader(googleAPI); strings.Contains(got, "some-value") {
				t.Errorf("TrackingHeader(%v) = %q, which holds the value of an environment variable", googleAPI, got)
			}
			// A destination that is not a Google API never gets a suffix.
			for _, dest := range []*url.URL{nil, otherHost} {
				if got, want := adkToken(TrackingHeader(dest)), "google-adk/"+Version; got != want {
					t.Errorf("TrackingHeader(%v) starts with %q, want %q", dest, got, want)
				}
			}
		})
	}
}

// Detection is not cached, so a variable set after the first call counts.
func TestTrackingHeaderReadsEnvironmentOnEveryCall(t *testing.T) {
	versiontest.ClearRuntimeEnv(t)
	if got, want := adkToken(TrackingHeader(googleAPI)), "google-adk/"+Version; got != want {
		t.Fatalf("TrackingHeader(%v) starts with %q, want %q", googleAPI, got, want)
	}
	t.Setenv("CLOUD_RUN_JOB", "my-job")
	if got, want := adkToken(TrackingHeader(googleAPI)), "google-adk/"+Version+"+cloud_run"; got != want {
		t.Errorf("TrackingHeader(%v) starts with %q, want %q", googleAPI, got, want)
	}
}

// The table holds the rows of adk-python's test_is_googleapis_url, and more.
// A destination that does not parse is no destination, as it is to a caller.
func TestTrackingHeaderDestination(t *testing.T) {
	versiontest.ClearRuntimeEnv(t)
	t.Setenv("K_SERVICE", "svc")
	tests := []struct {
		dest       string
		wantSuffix bool
	}{
		{dest: "https://aiplatform.googleapis.com/v1", wantSuffix: true},
		{dest: "https://aiplatform.mtls.googleapis.com/v1", wantSuffix: true},
		{dest: "https://us-central1-aiplatform.googleapis.com/", wantSuffix: true},
		{dest: "https://aiplatform.us.rep.googleapis.com/", wantSuffix: true},
		{dest: "https://BigQuery.GoogleAPIs.com/v1", wantSuffix: true},
		{dest: "HTTPS://bigquery.googleapis.com/v1", wantSuffix: true},
		{dest: "https://bigquery.googleapis.com:443/v1", wantSuffix: true},
		{dest: "https://bigquery.googleapis.com", wantSuffix: true},
		{dest: "https://1.2.3.4.googleapis.com/v1", wantSuffix: true},
		{dest: "https://AZaz09.googleapis.com/v1", wantSuffix: true},
		// No such host exists, but it has the suffix, as in adk-python.
		{dest: "https://.googleapis.com/v1", wantSuffix: true},
		// A backslash outside the authority is fine.
		{dest: `https://bigquery.googleapis.com/v1?q=a\b`, wantSuffix: true},
		{dest: "http://aiplatform.googleapis.com/v1"},
		{dest: "wss://generativelanguage.googleapis.com/"},
		{dest: "https://notgoogleapis.com/v1"},
		{dest: "https://googleapis.com.evil.com/v1"},
		{dest: "https://example.com/v1"},
		{dest: "https://example.com/bigquery.googleapis.com"},
		{dest: "https://example.com/?u=https://bigquery.googleapis.com"},
		// The apex is not a service endpoint.
		{dest: "https://googleapis.com/v1"},
		{dest: "https://googleapis.com@evil.com/v1"},
		{dest: "https://user@bigquery.googleapis.com/v1"},
		{dest: "https://user:pw@bigquery.googleapis.com/v1"},
		{dest: `https://evil.com\.googleapis.com/v1`},
		{dest: `https://evil.com\@bigquery.googleapis.com/v1`},
		// The zone of an IPv6 address can end in the suffix, and the address
		// is anyone's.
		{dest: "https://[::1%25.googleapis.com]/v1"},
		{dest: "https://[::1%25.googleapis.com]:8443/v1"},
		{dest: "https://[::ffff:127.0.0.1%25.googleapis.com]/v1"},
		{dest: "https://[2001:db8::1%25.googleapis.com]/v1"},
		{dest: "https://[::1]/v1"},
		{dest: "https://127.0.0.1/v1"},
		// Not the characters of a DNS name.
		{dest: "https://evil.com,foo.googleapis.com/v1"},
		{dest: "https://evil.com%25.googleapis.com/v1"},
		{dest: "https://my_api.googleapis.com/v1"},
		{dest: "https://\u0130.googleapis.com/v1"},
		{dest: "bigquery.googleapis.com/v1"},
		{dest: "//bigquery.googleapis.com/v1"},
		{dest: "https://bigquery.googleapis.com./v1"},
		{dest: "https://%zz.googleapis.com/v1"},
		// "İ" and "ſ" are not the ASCII letters that case mapping turns them
		// into, so these are names of other domains.
		{dest: "https://x.googleap\u0130s.com/v1"},
		{dest: "https://x.googleapi\u017f.com/v1"},
		{dest: ""},
	}
	for _, tt := range tests {
		want := "google-adk/" + Version
		if tt.wantSuffix {
			want += "+cloud_run"
		}
		dest, _ := url.Parse(tt.dest)
		if got := adkToken(TrackingHeader(dest)); got != want {
			t.Errorf("TrackingHeader(%q) starts with %q, want %q", tt.dest, got, want)
		}
	}
}

// The host a client connects to decides, which is the URL's Host: a URL built
// by hand can render as another one.
func TestTrackingHeaderDestinationBuiltByHand(t *testing.T) {
	versiontest.ClearRuntimeEnv(t)
	t.Setenv("K_SERVICE", "svc")
	tests := []struct {
		dest       *url.URL
		wantSuffix bool
	}{
		{dest: &url.URL{Scheme: "https", Host: "bigquery.googleapis.com"}, wantSuffix: true},
		{dest: &url.URL{Scheme: "https", Host: "bigquery.googleapis.com", Opaque: "//evil.com/mcp"}, wantSuffix: true},
		{dest: &url.URL{Scheme: "https", Host: "evil.com", Opaque: "//bigquery.googleapis.com/mcp"}},
		{dest: &url.URL{Scheme: "https", Host: "evil.com", Path: "//bigquery.googleapis.com/mcp"}},
		// http.Transport refuses a scheme spelled "HTTPS".
		{dest: &url.URL{Scheme: "HTTPS", Host: "bigquery.googleapis.com"}},
		{dest: &url.URL{Host: "bigquery.googleapis.com"}},
		{dest: &url.URL{}},
		{dest: nil},
	}
	for _, tt := range tests {
		want := "google-adk/" + Version
		if tt.wantSuffix {
			want += "+cloud_run"
		}
		if got := adkToken(TrackingHeader(tt.dest)); got != want {
			t.Errorf("TrackingHeader(%#v) starts with %q, want %q", tt.dest, got, want)
		}
	}
}

// TestTrackingHeaderDestinationIsDialed checks the rule against what it is
// for: a destination that gets the runtime suffix is one where net/http
// connects, over TLS, to a DNS name under googleapis.com. The address is the
// one http.Transport asks to dial. ".googleapis.com" itself counts, though no
// such host exists, because adk-python accepts it.
//
// The destinations are every sequence of up to three of the pieces below, put
// where a host, a zone, userinfo, a port or a path would go. Left out on
// purpose: schemes other than https, which TestTrackingHeaderDestination
// covers, and non-ASCII letters other than "İ".
func TestTrackingHeaderDestinationIsDialed(t *testing.T) {
	pieces := []string{"a", "A", "1", ".", "-", "_", ":", "%", "%25", "[", "]", "@", "/", "\\", ",", "#", "?", " ", "\u0130", "::1", "1.2.3.4"}
	templates := []string{
		"https://%s.googleapis.com/v1",
		"https://%s.googleapis.com:443/v1",
		"https://%sgoogleapis.com/v1",
		"https://[%s.googleapis.com]/v1",
		"https://[::1%s.googleapis.com]/v1",
		"https://[::1%s.googleapis.com]:443/v1",
		"https://%sx.googleapis.com/v1",
		"https://x.googleapis.com%s",
	}
	dnsName := regexp.MustCompile(`^[a-z0-9.-]*\.googleapis\.com$`)

	var dialed string
	dial := func(_ context.Context, _, addr string) (net.Conn, error) {
		dialed = addr
		return nil, errDialed
	}
	client := &http.Client{Transport: &http.Transport{DialContext: dial, DialTLSContext: dial, DisableKeepAlives: true}}
	tagged := "google-adk/" + Version + "+cloud_run"

	// check reports whether dest gets the suffix, after checking where a
	// request to it would connect.
	var connects int
	check := func(t *testing.T, dest string) bool {
		u, err := url.Parse(dest)
		if err != nil || adkToken(TrackingHeader(u)) != tagged {
			return false
		}
		req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, dest, http.NoBody)
		if err != nil {
			return true // No request can be sent there.
		}
		dialed = ""
		resp, err := client.Do(req)
		if err == nil {
			_ = resp.Body.Close()
			t.Fatalf("request to %q was sent", dest)
		}
		if !errors.Is(err, errDialed) {
			return true // Refused before any connection.
		}
		connects++
		host, _, err := net.SplitHostPort(dialed)
		if err != nil {
			t.Errorf("%q gets the runtime suffix and connects to %q, which is not host:port", dest, dialed)
			return true
		}
		if _, err := netip.ParseAddr(host); err == nil {
			t.Errorf("%q gets the runtime suffix and connects to the IP address %q", dest, dialed)
		}
		if !dnsName.MatchString(strings.ToLower(host)) {
			t.Errorf("%q gets the runtime suffix and connects to %q, which is not a DNS name under googleapis.com", dest, dialed)
		}
		return true
	}

	for _, mode := range []struct {
		name, godebug string
		// only is a destination that url.Parse takes only with godebug.
		only string
	}{
		{name: "default"},
		// Go 1.26 added the setting: without it, url.Parse takes a host that
		// holds more than one colon.
		{name: "colons in the host", godebug: "urlstrictcolons=0", only: "https://evil.com:443,foo.googleapis.com:443/v1"},
	} {
		t.Run(mode.name, func(t *testing.T) {
			versiontest.ClearRuntimeEnv(t)
			t.Setenv("K_SERVICE", "svc")
			if mode.godebug != "" {
				// After what the environment holds, because the last setting
				// of a name wins.
				t.Setenv("GODEBUG", os.Getenv("GODEBUG")+","+mode.godebug)
				if _, err := url.Parse(mode.only); err != nil {
					t.Skipf("GODEBUG=%s has no effect: %v", mode.godebug, err)
				}
				if check(t, mode.only) {
					t.Errorf("%q gets the runtime suffix", mode.only)
				}
			}
			connects = 0
			var total, suffixed int
			var walk func(prefix string, depth int)
			walk = func(prefix string, depth int) {
				for _, tmpl := range templates {
					total++
					if check(t, fmt.Sprintf(tmpl, prefix)) {
						suffixed++
					}
				}
				if depth == 0 {
					return
				}
				for _, piece := range pieces {
					walk(prefix+piece, depth-1)
				}
			}
			walk("", 3)
			// Guards against a generator, or a dialer, that reaches nothing.
			if total < 50000 || suffixed < 500 || connects < suffixed/2 {
				t.Errorf("checked %d destinations, %d with the suffix, %d of those connecting: too few to mean anything", total, suffixed, connects)
			}
			t.Logf("%d destinations, %d with the suffix, %d of those connecting", total, suffixed, connects)
		})
	}
}

func TestWithTrackingHeaders(t *testing.T) {
	adk := TrackingHeader(nil)
	tests := []struct {
		name string
		in   http.Header
		want http.Header
	}{
		{
			name: "nil",
			want: http.Header{"User-Agent": {adk}, "X-Goog-Api-Client": {adk}},
		},
		{
			name: "caller tokens follow ADK labels",
			in:   http.Header{"User-Agent": {"my-app/1.0"}, "X-Goog-Api-Client": {"my-lib/2.0"}},
			want: http.Header{"User-Agent": {adk + " my-app/1.0"}, "X-Goog-Api-Client": {adk + " my-lib/2.0"}},
		},
		{
			name: "non-canonical keys are merged into one in sorted key order",
			in:   http.Header{"user-agent": {"a/1"}, "User-Agent": {"b/2"}, "USER-AGENT": {"c/3"}, "uSer-agEnt": {"d/4"}, "x-goog-api-client": {"e/5"}},
			want: http.Header{"User-Agent": {adk + " c/3 b/2 d/4 a/1"}, "X-Goog-Api-Client": {adk + " e/5"}},
		},
		{
			name: "duplicate tokens are dropped",
			in:   http.Header{"User-Agent": {"my-app/1.0 " + adk, "my-app/1.0"}},
			want: http.Header{"User-Agent": {adk + " my-app/1.0"}, "X-Goog-Api-Client": {adk}},
		},
		{
			name: "a stale ADK label is kept after the current one",
			in:   http.Header{"X-Goog-Api-Client": {"my-lib/2.0 google-adk/0.0.1"}},
			want: http.Header{"User-Agent": {adk}, "X-Goog-Api-Client": {adk + " my-lib/2.0 google-adk/0.0.1"}},
		},
		{
			name: "empty values add nothing",
			in:   http.Header{"User-Agent": {"", "  "}},
			want: http.Header{"User-Agent": {adk}, "X-Goog-Api-Client": {adk}},
		},
		{
			name: "other headers are kept",
			in:   http.Header{"X-Custom": {"v"}},
			want: http.Header{"User-Agent": {adk}, "X-Custom": {"v"}, "X-Goog-Api-Client": {adk}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			orig := tt.in.Clone()
			// Repeated, since map order would make an unsorted merge pass by chance.
			for range 20 {
				got := WithTrackingHeaders(tt.in, nil)
				if diff := cmp.Diff(tt.want, got); diff != "" {
					t.Fatalf("WithTrackingHeaders() mismatch (-want +got):\n%s", diff)
				}
			}
			if diff := cmp.Diff(orig, tt.in); diff != "" {
				t.Errorf("WithTrackingHeaders() modified its input (-want +got):\n%s", diff)
			}
		})
	}
}

func TestWithTrackingHeadersRuntimeSuffix(t *testing.T) {
	versiontest.ClearRuntimeEnv(t)
	t.Setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
	plain, tagged := TrackingHeader(nil), TrackingHeader(googleAPI)
	if want := "google-adk/" + Version + "+k8s"; adkToken(tagged) != want {
		t.Fatalf("TrackingHeader(%v) = %q, want it to start with %q", googleAPI, tagged, want)
	}
	in := http.Header{"User-Agent": {"my-app/1.0"}}
	tests := []struct {
		dest *url.URL
		want http.Header
	}{
		{dest: googleAPI, want: http.Header{"User-Agent": {tagged + " my-app/1.0"}, "X-Goog-Api-Client": {tagged}}},
		{dest: otherHost, want: http.Header{"User-Agent": {plain + " my-app/1.0"}, "X-Goog-Api-Client": {plain}}},
		{dest: nil, want: http.Header{"User-Agent": {plain + " my-app/1.0"}, "X-Goog-Api-Client": {plain}}},
	}
	for _, tt := range tests {
		if diff := cmp.Diff(tt.want, WithTrackingHeaders(in, tt.dest)); diff != "" {
			t.Errorf("WithTrackingHeaders(_, %v) mismatch (-want +got):\n%s", tt.dest, diff)
		}
	}
}

// googleAPI gets the hosting runtime suffix, and otherHost does not.
var (
	googleAPI = &url.URL{Scheme: "https", Host: "aiplatform.googleapis.com", Path: "/v1"}
	otherHost = &url.URL{Scheme: "https", Host: "mcp.example", Path: "/mcp"}
)

// errDialed stops a request once the address it would connect to is known.
var errDialed = errors.New("dialed")

// adkToken returns the first token of a tracking header value.
func adkToken(header string) string {
	token, _, _ := strings.Cut(header, " ")
	return token
}
