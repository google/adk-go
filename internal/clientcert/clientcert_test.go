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

package clientcert

import (
	"bytes"
	"crypto/tls"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"google.golang.org/adk/v2/internal/clientcert/clientcerttest"
)

// Outside an Isolate child, only the tests that stand in for the auth library
// turn the certificate on.
func TestMain(m *testing.M) {
	clientcerttest.PinOff()
	os.Exit(m.Run())
}

// libraryDER is the certificate of the stand-in auth library's own search.
var libraryDER = []byte("found by the auth library")

// fakeAuthLibrary stands in for the auth library, so that a test of the rule
// cannot start the library's search of the machine. searches reports how
// often the rule asked it to search, and setting gateOff makes it ignore a
// certificate it is handed, as the library does while its gate is off.
type fakeAuthLibrary struct {
	searches int
	gateOff  bool
}

func newFakeAuthLibrary(t *testing.T) *fakeAuthLibrary {
	t.Helper()
	library := authLibrary
	t.Cleanup(func() { authLibrary = library })
	f := &fakeAuthLibrary{}
	authLibrary = func(given source) (source, error) {
		if f.gateOff {
			return nil, nil
		}
		if given != nil {
			return given, nil
		}
		f.searches++
		return func(*tls.CertificateRequestInfo) (*tls.Certificate, error) {
			return &tls.Certificate{Certificate: [][]byte{libraryDER}}, nil
		}, nil
	}
	return f
}

// setHome makes dir the home directory gcloud's default path is under.
func setHome(t *testing.T, dir string) (defaultConfig string) {
	t.Helper()
	t.Setenv("HOME", dir)
	t.Setenv("APPDATA", "")
	if runtime.GOOS == "windows" {
		t.Setenv("APPDATA", filepath.Join(dir, ".config"))
	}
	return filepath.Join(dir, ".config", "gcloud", "certificate_config.json")
}

// TestSourceRule covers the rule. The auth library, which has the last word
// on a certificate it is handed, is the stand-in here and real in TestSource.
func TestSourceRule(t *testing.T) {
	library := newFakeAuthLibrary(t)
	named, namedDER := clientcerttest.Workload(t)
	other, homeDER := clientcerttest.Workload(t)
	// The config at gcloud's default path names its certificate by full path.
	atHome, err := os.ReadFile(other)
	if err != nil {
		t.Fatal(err)
	}
	defaultConfig := setHome(t, t.TempDir())
	if err := os.MkdirAll(filepath.Dir(defaultConfig), 0o700); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	file := func(name, content string) string {
		path := filepath.Join(dir, name)
		clientcerttest.WriteFile(t, path, []byte(content))
		return path
	}
	var (
		unset   = clientcerttest.Unset
		notJSON = file("bad.json", "{")
		proxy   = file("proxy.json", `{"cert_configs":{"pkcs11":{}},"libs":{}}`)
		null    = file("null.json", `{"cert_configs":{"workload":null}}`)
		missing = filepath.Join(dir, "missing.json")
		optOut  = "GOOGLE_API_USE_CLIENT_CERTIFICATE=false"
	)
	for _, tc := range []struct {
		name    string
		useCert string // GOOGLE_API_USE_CLIENT_CERTIFICATE
		gcloud  string // CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE
		config  string // GOOGLE_API_CERTIFICATE_CONFIG
		home    string // the config at gcloud's default path, "" for none
		gateOff bool   // the auth library ignores a certificate
		want    []byte // the certificate, nil for none
		wantErr []string
	}{
		{name: "true", useCert: "true", gcloud: unset, config: unset, want: libraryDER},
		{name: "upper case", useCert: "TRUE", gcloud: unset, config: unset, want: libraryDER},
		{name: "mixed case", useCert: "tRuE", gcloud: unset, config: unset, want: libraryDER},
		{name: "true with a workload config", useCert: "true", gcloud: unset, config: named, want: namedDER},
		{name: "true wins over gcloud's variable", useCert: "true", gcloud: "false", config: unset, want: libraryDER},
		{name: "false", useCert: "false", gcloud: unset, config: named},
		{name: "false wins over gcloud's variable", useCert: "false", gcloud: "true", config: named},
		// strconv.ParseBool reads these as true. adk-python does not.
		{name: "1", useCert: "1", gcloud: unset, config: named},
		{name: "t", useCert: "t", gcloud: unset, config: named},
		{name: "not a boolean", useCert: "yes", gcloud: unset, config: named},
		// adk-python goes on to gcloud's variable. The auth library reads a
		// set variable as false and would ignore the certificate.
		{name: "set but empty", useCert: "", gcloud: "true", config: named},

		{name: "gcloud's variable", useCert: unset, gcloud: "true", config: unset, want: libraryDER},
		{name: "gcloud's variable in upper case", useCert: unset, gcloud: "TRUE", config: unset, want: libraryDER},
		{name: "gcloud's variable with a config at the default path", useCert: unset, gcloud: "true", config: unset, home: string(atHome), want: homeDER},
		{name: "gcloud's variable false", useCert: unset, gcloud: "false", config: named},
		{name: "gcloud's variable 1", useCert: unset, gcloud: "1", config: named},
		{name: "gcloud's variable empty", useCert: unset, gcloud: "", config: named, want: namedDER},

		{name: "workload config", useCert: unset, gcloud: unset, config: named, want: namedDER},
		{name: "workload config at the default path", useCert: unset, gcloud: unset, config: unset, home: string(atHome), want: homeDER},
		{name: "empty config variable", useCert: unset, gcloud: unset, config: "", home: string(atHome), want: homeDER},
		{name: "the named config wins over the default path", useCert: unset, gcloud: unset, config: named, home: string(atHome), want: namedDER},
		// A named config is the only one read, as in adk-python.
		{name: "named config without a workload section", useCert: unset, gcloud: unset, config: proxy, home: string(atHome)},
		{name: "named config missing", useCert: unset, gcloud: unset, config: missing, home: string(atHome)},
		{name: "named config is a directory", useCert: unset, gcloud: unset, config: dir, home: string(atHome)},
		{name: "no config", useCert: unset, gcloud: unset, config: unset},
		{name: "null workload section", useCert: unset, gcloud: unset, config: null},
		{name: "true with a null workload section", useCert: "true", gcloud: unset, config: null, want: libraryDER},

		{name: "not JSON", useCert: unset, gcloud: unset, config: notJSON, wantErr: []string{notJSON, optOut}},
		{name: "not JSON at the default path", useCert: unset, gcloud: unset, config: unset, home: "{", wantErr: []string{defaultConfig, optOut}},
		{name: "true and not JSON", useCert: "true", gcloud: unset, config: notJSON, wantErr: []string{notJSON}},
		{name: "false and not JSON", useCert: "false", gcloud: unset, config: notJSON},
		{name: "gcloud's variable false and not JSON", useCert: unset, gcloud: "false", config: notJSON},
		{name: "cert_configs is not an object", useCert: unset, gcloud: unset, config: file("list.json", `{"cert_configs":[]}`), wantErr: []string{"list.json", optOut}},
		{name: "workload section without paths", useCert: unset, gcloud: unset, config: file("empty.json", `{"cert_configs":{"workload":{}}}`), wantErr: []string{"empty.json", "cert_path", optOut}},
		{name: "workload section without a key", useCert: unset, gcloud: unset, config: file("nokey.json", `{"cert_configs":{"workload":{"cert_path":"cert.pem"}}}`), wantErr: []string{"nokey.json", "key_path"}},
		{name: "workload section without a certificate", useCert: unset, gcloud: unset, config: file("nocert.json", `{"cert_configs":{"workload":{"key_path":"key.pem"}}}`), wantErr: []string{"nocert.json", "cert_path"}},

		// With the auth library's gate off nothing is on, and no config is read.
		{name: "gate off, workload config", useCert: unset, gcloud: unset, config: named, gateOff: true},
		{name: "gate off, true", useCert: "true", gcloud: unset, config: unset, gateOff: true},
		{name: "gate off, not JSON", useCert: unset, gcloud: unset, config: notJSON, gateOff: true},
		{name: "gate off, true and not JSON", useCert: "true", gcloud: unset, config: notJSON, gateOff: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			clientcerttest.Setenv(t, envUseClientCert, tc.useCert)
			clientcerttest.Setenv(t, envGcloudUseClientCert, tc.gcloud)
			clientcerttest.Setenv(t, envCertConfig, tc.config)
			if tc.home != "" {
				clientcerttest.WriteFile(t, defaultConfig, []byte(tc.home))
				t.Cleanup(func() { _ = os.Remove(defaultConfig) })
			}
			library.gateOff = tc.gateOff
			before := library.searches

			src, err := Source()
			if len(tc.wantErr) > 0 {
				if err == nil {
					t.Fatalf("Source() = (non-nil: %v), nil error; want an error", src != nil)
				}
				for _, want := range tc.wantErr {
					if !strings.Contains(err.Error(), want) {
						t.Errorf("Source() error = %q, want it to contain %q", err, want)
					}
				}
				return
			}
			if err != nil {
				t.Fatalf("Source() error = %v", err)
			}
			if tc.want == nil {
				if src != nil {
					t.Error("Source() returned a certificate source, want nil")
				}
				// With the rule off the auth library must not search: that is
				// what runs a certificate helper.
				if got := library.searches - before; got != 0 {
					t.Errorf("Source() had the auth library search %d times with the rule off, want 0", got)
				}
				return
			}
			// A certificate the rule loaded itself needs no search either.
			if got, want := library.searches-before, bytes.Equal(tc.want, libraryDER); (got != 0) != want {
				t.Errorf("Source() had the auth library search %d times, want a search: %v", got, want)
			}
			if src == nil {
				t.Fatal("Source() = nil, want a certificate source")
			}
			cert, err := src(nil)
			if err != nil {
				t.Fatalf("certificate source: %v", err)
			}
			if len(cert.Certificate) != 1 || !bytes.Equal(cert.Certificate[0], tc.want) {
				t.Error("the source returned a certificate other than the expected one")
			}
		})
	}
}

// TestSourceWithoutAHomeDirectory covers a process with no HOME, for which
// gcloud's default path is unknown. A config under the working directory,
// where a path built from an empty home would point, is not read.
func TestSourceWithoutAHomeDirectory(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("gcloud's default path is under %APPDATA% on Windows")
	}
	library := newFakeAuthLibrary(t)
	config, _ := clientcerttest.Workload(t)
	content, err := os.ReadFile(config)
	if err != nil {
		t.Fatal(err)
	}
	cwd := t.TempDir()
	if err := os.MkdirAll(filepath.Join(cwd, ".config", "gcloud"), 0o700); err != nil {
		t.Fatal(err)
	}
	clientcerttest.WriteFile(t, filepath.Join(cwd, ".config", "gcloud", "certificate_config.json"), content)
	t.Chdir(cwd)
	t.Setenv("HOME", "")
	clientcerttest.Setenv(t, envUseClientCert, clientcerttest.Unset)
	clientcerttest.Setenv(t, envGcloudUseClientCert, clientcerttest.Unset)
	clientcerttest.Setenv(t, envCertConfig, clientcerttest.Unset)

	if src, err := Source(); src != nil || err != nil || library.searches != 0 {
		t.Errorf("Source() = (non-nil: %v), %v after %d searches by the auth library; want nil, nil and 0", src != nil, err, library.searches)
	}
}

// TestDefaultConfigPath covers the system the tests do not run on.
func TestDefaultConfigPath(t *testing.T) {
	t.Setenv("HOME", filepath.Join("home", "user"))
	t.Setenv("APPDATA", filepath.Join("appdata", "roaming"))
	if got, want := defaultConfigPath("windows"), filepath.Join("appdata", "roaming", "gcloud", "certificate_config.json"); got != want {
		t.Errorf("defaultConfigPath(windows) = %q, want %q", got, want)
	}
	if runtime.GOOS != "windows" {
		if got, want := defaultConfigPath("linux"), filepath.Join("home", "user", ".config", "gcloud", "certificate_config.json"); got != want {
			t.Errorf("defaultConfigPath(linux) = %q, want %q", got, want)
		}
	}
	// A path built from an empty directory would be relative to the working
	// directory.
	t.Setenv("APPDATA", "")
	if got := defaultConfigPath("windows"); got != "" {
		t.Errorf("defaultConfigPath(windows) = %q without APPDATA, want none", got)
	}
}

// TestSourceFollowsARotatedCertificate pins that the workload certificate is
// read for every handshake and not once when Source returns.
func TestSourceFollowsARotatedCertificate(t *testing.T) {
	newFakeAuthLibrary(t)
	config, der := clientcerttest.Workload(t)
	clientcerttest.Setenv(t, envUseClientCert, clientcerttest.Unset)
	clientcerttest.Setenv(t, envGcloudUseClientCert, clientcerttest.Unset)
	t.Setenv(envCertConfig, config)

	src, err := Source()
	if err != nil || src == nil {
		t.Fatalf("Source() = (non-nil: %v), %v; want a certificate source", src != nil, err)
	}
	for _, rotate := range []bool{false, true} {
		if rotate {
			der = clientcerttest.Rotate(t, config)
		}
		cert, err := src(nil)
		if err != nil {
			t.Fatalf("certificate source: %v", err)
		}
		if len(cert.Certificate) != 1 || !bytes.Equal(cert.Certificate[0], der) {
			t.Fatalf("the source returned a certificate other than the one in the files (rotated: %v)", rotate)
		}
	}
	if err := os.Remove(filepath.Join(filepath.Dir(config), "key.pem")); err != nil {
		t.Fatal(err)
	}
	if cert, err := src(nil); err == nil {
		t.Errorf("certificate source = %v, nil error with the key file gone; want an error", cert)
	}
}

// TestSource covers what the rule and the real auth library decide together.
// See [clientcerttest.Isolate] for why it runs in a child process.
func TestSource(t *testing.T) {
	if !clientcerttest.Isolate(t) {
		return
	}
	workload, der := clientcerttest.Workload(t)
	home := clientcerttest.HomeCertificate(t)
	// The certificate does not depend on credentials, so Source must not look
	// for any. These cannot be loaded.
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", filepath.Join(t.TempDir(), "absent.json"))
	unset := clientcerttest.Unset
	for _, tc := range []struct {
		name     string
		useCert  string
		gcloud   string
		config   string
		universe string
		want     []byte
	}{
		{name: "workload config", useCert: unset, gcloud: unset, config: workload, want: der},
		{name: "workload config at the default path", useCert: unset, gcloud: unset, config: unset, want: home},
		{name: "true", useCert: "true", gcloud: unset, config: workload, want: der},
		{name: "upper case", useCert: "TRUE", gcloud: unset, config: workload, want: der},
		{name: "capitalized", useCert: "True", gcloud: unset, config: workload, want: der},
		{name: "true in another universe", useCert: "true", gcloud: unset, config: workload, universe: "example.com", want: der},
		{name: "gcloud's variable", useCert: unset, gcloud: "true", config: workload, want: der},

		{name: "false", useCert: "false", gcloud: unset, config: workload},
		{name: "gcloud's variable false", useCert: unset, gcloud: "false", config: workload},
		{name: "1", useCert: "1", gcloud: unset, config: workload},
		// The rule is on for these three, and the auth library's gate is off.
		{name: "mixed case", useCert: "tRuE", gcloud: unset, config: workload},
		{name: "workload config in another universe", useCert: unset, gcloud: unset, config: workload, universe: "example.com"},
		{name: "gcloud's variable in another universe", useCert: unset, gcloud: "true", config: workload, universe: "example.com"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			clientcerttest.Setenv(t, envUseClientCert, tc.useCert)
			clientcerttest.Setenv(t, envGcloudUseClientCert, tc.gcloud)
			clientcerttest.Setenv(t, envCertConfig, tc.config)
			t.Setenv(envUniverseDomain, tc.universe)
			src, err := Source()
			if err != nil {
				t.Fatalf("Source() error = %v", err)
			}
			if tc.want == nil {
				if src != nil {
					t.Fatal("Source() returned a certificate source, want nil")
				}
				return
			}
			if src == nil {
				t.Fatal("Source() = nil, want the workload certificate")
			}
			cert, err := src(nil)
			if err != nil {
				t.Fatalf("certificate source: %v", err)
			}
			if len(cert.Certificate) != 1 || !bytes.Equal(cert.Certificate[0], tc.want) {
				t.Error("the source returned a certificate other than the configured one")
			}
		})
	}
}

// TestSourceWithSecureConnect covers a certificate source that runs a helper
// program, with the real auth library. Where [clientcerttest.SecureConnect]
// skips it, TestSourceRule still pins when the library is asked.
func TestSourceWithSecureConnect(t *testing.T) {
	if !clientcerttest.Isolate(t) {
		return
	}
	der, ran := clientcerttest.SecureConnect(t)

	// With no variable set, the auth library by itself runs the helper.
	for _, gcloud := range []string{clientcerttest.Unset, "false"} {
		clientcerttest.Setenv(t, envGcloudUseClientCert, gcloud)
		if src, err := Source(); src != nil || err != nil {
			t.Fatalf("Source() = (non-nil: %v), %v with the certificate off; want nil, nil", src != nil, err)
		}
	}
	if ran() {
		t.Fatal("the helper ran with the certificate off")
	}

	t.Setenv(envGcloudUseClientCert, "true")
	src, err := Source()
	if err != nil || src == nil {
		t.Fatalf("Source() = (non-nil: %v), %v with gcloud's variable true; want the helper's certificate", src != nil, err)
	}
	cert, err := src(nil)
	if err != nil {
		t.Fatalf("certificate source: %v", err)
	}
	if len(cert.Certificate) != 1 || !bytes.Equal(cert.Certificate[0], der) || !ran() {
		t.Error("the source returned a certificate other than the helper's")
	}
}

// TestSourceReportsAFailedSearch covers an error from the auth library's own
// search. A directory where the certificate config should be is one the rule
// reads as no config and the library fails on, without going on to another
// source. It runs apart from the tests above because the library keeps the
// failure for the rest of the process.
func TestSourceReportsAFailedSearch(t *testing.T) {
	if !clientcerttest.Isolate(t) {
		return
	}
	dir := t.TempDir()
	t.Setenv(envCertConfig, dir)
	// With the rule off the library must not search, or it would fail here.
	if src, err := Source(); src != nil || err != nil {
		t.Fatalf("Source() = (non-nil: %v), %v with no variable set; want nil, nil", src != nil, err)
	}

	t.Setenv(envGcloudUseClientCert, "true")
	src, err := Source()
	if err == nil {
		t.Fatalf("Source() = (non-nil: %v), nil error; want the auth library's error", src != nil)
	}
	for _, want := range []string{dir, "GOOGLE_API_USE_CLIENT_CERTIFICATE=false"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("Source() error = %q, want it to contain %q", err, want)
		}
	}
}
