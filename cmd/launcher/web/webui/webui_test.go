// Copyright 2025 Google LLC
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

package webui

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"regexp"
	"strings"
	"testing"

	"github.com/gorilla/mux"

	weblauncher "google.golang.org/adk/v2/cmd/launcher/web"
)

const testBackendAddress = "http://localhost:8080/api"

// assetRefPattern finds the src/href of every <script> and <link> in a
// document. The tests use it to follow what a browser would actually fetch.
var assetRefPattern = regexp.MustCompile(`(?is)<(?:script|link)\s[^>]*?(?:src|href)\s*=\s*"([^"]+)"`)

// docBaseHrefPattern captures the value of the <base href> tag.
var docBaseHrefPattern = regexp.MustCompile(`(?is)<base\s[^>]*?href\s*=\s*"([^"]*)"`)

// newTestRouter builds the UI routes the way the web launcher does, including
// StrictSlash(true), so the tests exercise the production routing behavior.
func newTestRouter(t *testing.T, pathPrefix string) *mux.Router {
	t.Helper()
	router := mux.NewRouter().StrictSlash(true)
	launcher := &webUILauncher{config: &webUIConfig{
		pathPrefix:     pathPrefix,
		backendAddress: testBackendAddress,
	}}
	if err := launcher.SetupSubrouters(router, nil); err != nil {
		t.Fatalf("SetupSubrouters(%q) failed: %v", pathPrefix, err)
	}
	return router
}

// request performs an in-process request against the router.
func request(t *testing.T, router http.Handler, method, target string) (*http.Response, string) {
	t.Helper()
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, httptest.NewRequest(method, target, nil))
	res := rec.Result()
	body, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatalf("%s %s: reading body failed: %v", method, target, err)
	}
	if err := res.Body.Close(); err != nil {
		t.Fatalf("%s %s: closing body failed: %v", method, target, err)
	}
	return res, string(body)
}

// assertAssetsResolve replays what a browser does with the returned document:
// resolve every script and stylesheet reference against the <base href> (itself
// resolved against the document URL) and fetch it. A document whose assets all
// 404 renders as a blank page, so checking the status code and the shape of the
// body is not enough - the references have to be followed.
func assertAssetsResolve(t *testing.T, router http.Handler, docURL, body string) {
	t.Helper()
	doc, err := url.Parse(docURL)
	if err != nil {
		t.Fatalf("cannot parse document URL %q: %v", docURL, err)
	}
	base := doc
	if m := docBaseHrefPattern.FindStringSubmatch(body); m != nil {
		if base, err = doc.Parse(m[1]); err != nil {
			t.Fatalf("cannot resolve <base href=%q> against %q: %v", m[1], docURL, err)
		}
	}

	local := 0
	for _, m := range assetRefPattern.FindAllStringSubmatch(body, -1) {
		ref := m[1]
		if strings.HasPrefix(ref, "//") || strings.HasPrefix(ref, "data:") {
			continue
		}
		resolved, err := base.Parse(ref)
		if err != nil {
			t.Errorf("%s: cannot resolve asset reference %q: %v", docURL, ref, err)
			continue
		}
		if resolved.Host != doc.Host {
			// A third-party asset (fonts.gstatic.com) is not ours to serve.
			continue
		}
		local++
		res, _ := request(t, router, http.MethodGet, resolved.RequestURI())
		if res.StatusCode != http.StatusOK {
			t.Errorf("%s: asset %q resolved to %s, GET returned %d, want %d",
				docURL, ref, resolved.RequestURI(), res.StatusCode, http.StatusOK)
		}
	}
	// Guard against a document whose references the pattern failed to find:
	// zero assets checked would pass every assertion above vacuously.
	if local < 10 {
		t.Errorf("%s: found only %d local asset references in the served document, want at least 10", docURL, local)
	}
}

func TestDeepLinkServesAppWithWorkingAssets(t *testing.T) {
	router := newTestRouter(t, "/ui/")

	for _, target := range []string{
		"/ui/",
		"/ui/eval",
		"/ui/a/b/c",
	} {
		t.Run(target, func(t *testing.T) {
			res, body := request(t, router, http.MethodGet, target)
			if res.StatusCode != http.StatusOK {
				t.Fatalf("GET %s status = %d, want %d", target, res.StatusCode, http.StatusOK)
			}
			if ct := res.Header.Get("Content-Type"); !strings.Contains(ct, "text/html") {
				t.Errorf("GET %s Content-Type = %q, want text/html", target, ct)
			}
			assertAssetsResolve(t, router, "http://example.com"+target, body)
		})
	}
}

func TestServedIndexRewritesBaseHref(t *testing.T) {
	router := newTestRouter(t, "/ui/")

	res, body := request(t, router, http.MethodGet, "/ui/a/b/c")
	if res.StatusCode != http.StatusOK {
		t.Fatalf("GET /ui/a/b/c status = %d, want %d", res.StatusCode, http.StatusOK)
	}
	if strings.Contains(body, `<base href="./">`) {
		t.Errorf(`served document still contains the bundled <base href="./">, which resolves assets against the request directory`)
	}
	if !strings.Contains(body, `<base href="/ui/">`) {
		t.Errorf(`served document does not contain <base href="/ui/">`)
	}
}

func TestMissingAssetStill404s(t *testing.T) {
	router := newTestRouter(t, "/ui/")

	// A missing asset must not be answered with index.html: HTML arriving where
	// a script was expected fails much later and much less clearly.
	for _, target := range []string{
		"/ui/does-not-exist.js",
		"/ui/missing.css",
		"/ui/missing.svg",
		"/ui/missing.map",
		"/ui/missing.json",
		"/ui/missing.wasm",
		"/ui/missing.woff2",
		"/ui/deep/link/missing.js",
	} {
		t.Run(target, func(t *testing.T) {
			res, body := request(t, router, http.MethodGet, target)
			if res.StatusCode != http.StatusNotFound {
				t.Errorf("GET %s status = %d, want %d", target, res.StatusCode, http.StatusNotFound)
			}
			if strings.Contains(body, "<html") {
				t.Errorf("GET %s returned an HTML document instead of a plain 404", target)
			}
		})
	}
}

func TestBarePrefixRedirectsToPrefix(t *testing.T) {
	router := newTestRouter(t, "/ui/")

	res, _ := request(t, router, http.MethodGet, "/ui")
	if res.StatusCode != http.StatusMovedPermanently {
		t.Fatalf("GET /ui status = %d, want %d", res.StatusCode, http.StatusMovedPermanently)
	}
	if got := res.Header.Get("Location"); got != "/ui/" {
		t.Errorf("GET /ui Location = %q, want %q", got, "/ui/")
	}
}

// TestBarePrefixDoesNotLoop uses the real base router, whose StrictSlash(true)
// would turn a Path("/ui") redirect route into an endless /ui -> /ui/ -> /ui
// bounce, and follows the redirects the way a browser does.
func TestBarePrefixDoesNotLoop(t *testing.T) {
	router := weblauncher.BuildBaseRouter()
	launcher := &webUILauncher{config: &webUIConfig{
		pathPrefix:     "/ui/",
		backendAddress: testBackendAddress,
	}}
	if err := launcher.SetupSubrouters(router, nil); err != nil {
		t.Fatalf("SetupSubrouters failed: %v", err)
	}

	server := httptest.NewServer(router)
	defer server.Close()

	var hops []string
	client := &http.Client{
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			hops = append(hops, req.URL.Path)
			if len(via) >= 5 {
				return http.ErrUseLastResponse
			}
			return nil
		},
	}

	for _, target := range []string{"/ui", "/"} {
		t.Run(target, func(t *testing.T) {
			hops = nil
			res, err := client.Get(server.URL + target)
			if err != nil {
				t.Fatalf("GET %s failed: %v", target, err)
			}
			defer func() {
				if err := res.Body.Close(); err != nil {
					t.Errorf("closing body failed: %v", err)
				}
			}()
			if res.StatusCode != http.StatusOK {
				t.Errorf("GET %s ended at %s with status %d after hops %v, want %d",
					target, res.Request.URL.Path, res.StatusCode, hops, http.StatusOK)
			}
			if res.Request.URL.Path != "/ui/" {
				t.Errorf("GET %s ended at %q after hops %v, want %q", target, res.Request.URL.Path, hops, "/ui/")
			}
			if len(hops) > 2 {
				t.Errorf("GET %s took %d redirects (%v), which looks like a loop", target, len(hops), hops)
			}
		})
	}
}

// TestHeadIsAllowed goes through a real server rather than a recorder, because
// only net/http strips the response body of a HEAD.
func TestHeadIsAllowed(t *testing.T) {
	server := httptest.NewServer(newTestRouter(t, "/ui/"))
	defer server.Close()

	for _, target := range []string{
		"/ui/",
		"/ui/eval",
		"/ui/adk_favicon.svg",
		"/ui/assets/config/runtime-config.json",
	} {
		t.Run(target, func(t *testing.T) {
			res, err := server.Client().Head(server.URL + target)
			if err != nil {
				t.Fatalf("HEAD %s failed: %v", target, err)
			}
			body, err := io.ReadAll(res.Body)
			if err != nil {
				t.Fatalf("HEAD %s: reading body failed: %v", target, err)
			}
			if err := res.Body.Close(); err != nil {
				t.Fatalf("HEAD %s: closing body failed: %v", target, err)
			}
			if res.StatusCode != http.StatusOK {
				t.Errorf("HEAD %s status = %d, want %d", target, res.StatusCode, http.StatusOK)
			}
			if len(body) != 0 {
				t.Errorf("HEAD %s returned a %d-byte body, want none", target, len(body))
			}
		})
	}
}

func TestDirectoryListingIsSuppressed(t *testing.T) {
	router := newTestRouter(t, "/ui/")

	res, body := request(t, router, http.MethodGet, "/ui/assets/config/")
	if res.StatusCode != http.StatusNotFound {
		t.Errorf("GET /ui/assets/config/ status = %d, want %d", res.StatusCode, http.StatusNotFound)
	}
	if strings.Contains(body, "runtime-config.json") {
		t.Errorf("GET /ui/assets/config/ body links to runtime-config.json, so it is a directory listing:\n%s", body)
	}
}

func TestRuntimeConfigIsGenerated(t *testing.T) {
	router := newTestRouter(t, "/ui/")

	res, body := request(t, router, http.MethodGet, "/ui/assets/config/runtime-config.json")
	if res.StatusCode != http.StatusOK {
		t.Fatalf("GET runtime-config.json status = %d, want %d", res.StatusCode, http.StatusOK)
	}
	var got struct {
		BackendURL string `json:"backendUrl"`
	}
	if err := json.Unmarshal([]byte(body), &got); err != nil {
		t.Fatalf("runtime-config.json is not valid JSON (%v): %s", err, body)
	}
	if got.BackendURL != testBackendAddress {
		t.Errorf("runtime-config.json backendUrl = %q, want %q", got.BackendURL, testBackendAddress)
	}
}

func TestRootRedirectsToPrefix(t *testing.T) {
	router := newTestRouter(t, "/ui/")

	res, _ := request(t, router, http.MethodGet, "/")
	if res.StatusCode != http.StatusFound {
		t.Errorf("GET / status = %d, want %d", res.StatusCode, http.StatusFound)
	}
	if got := res.Header.Get("Location"); got != "/ui/" {
		t.Errorf("GET / Location = %q, want %q", got, "/ui/")
	}
}

// TestCustomPathPrefix checks that nothing hardcodes "/ui/".
func TestCustomPathPrefix(t *testing.T) {
	const prefix = "/adk-console/"
	router := newTestRouter(t, prefix)

	res, body := request(t, router, http.MethodGet, prefix+"a/b/c")
	if res.StatusCode != http.StatusOK {
		t.Fatalf("GET %sa/b/c status = %d, want %d", prefix, res.StatusCode, http.StatusOK)
	}
	if !strings.Contains(body, `<base href="`+prefix+`">`) {
		t.Errorf("served document does not contain <base href=%q>", prefix)
	}
	assertAssetsResolve(t, router, "http://example.com"+prefix+"a/b/c", body)

	res, _ = request(t, router, http.MethodGet, "/adk-console")
	if res.StatusCode != http.StatusMovedPermanently {
		t.Errorf("GET /adk-console status = %d, want %d", res.StatusCode, http.StatusMovedPermanently)
	}
	if got := res.Header.Get("Location"); got != prefix {
		t.Errorf("GET /adk-console Location = %q, want %q", got, prefix)
	}

	res, _ = request(t, router, http.MethodGet, "/")
	if got := res.Header.Get("Location"); got != prefix {
		t.Errorf("GET / Location = %q, want %q", got, prefix)
	}

	res, body = request(t, router, http.MethodGet, prefix+"assets/config/runtime-config.json")
	if res.StatusCode != http.StatusOK || !strings.Contains(body, "backendUrl") {
		t.Errorf("GET %sassets/config/runtime-config.json = %d %q, want 200 with backendUrl", prefix, res.StatusCode, body)
	}

	res, _ = request(t, router, http.MethodGet, prefix+"missing.js")
	if res.StatusCode != http.StatusNotFound {
		t.Errorf("GET %smissing.js status = %d, want %d", prefix, res.StatusCode, http.StatusNotFound)
	}

	// The default prefix must not be served at all under a custom one.
	res, _ = request(t, router, http.MethodGet, "/ui/")
	if res.StatusCode != http.StatusNotFound {
		t.Errorf("GET /ui/ status = %d with prefix %q, want %d", res.StatusCode, prefix, http.StatusNotFound)
	}
}
