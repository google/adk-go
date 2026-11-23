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
	"context"
	"net/http"
	"os"
	"strconv"
	"strings"

	"google.golang.org/api/option"
	"google.golang.org/api/option/internaloption"
	htransport "google.golang.org/api/transport/http"
)

const (
	defaultUniverse = "googleapis.com"

	envMTLSEndpoint = "GOOGLE_API_USE_MTLS_ENDPOINT"
	// The name envMTLSEndpoint replaced. The client libraries still read it
	// when the new one is unset.
	envMTLSEndpointOld = "GOOGLE_API_USE_MTLS"
	// Makes google.golang.org/api ignore internaloption.EnableNewAuthLibrary.
	envNewAuthLibOff = "GOOGLE_API_GO_EXPERIMENTAL_DISABLE_NEW_AUTH_LIB"
)

// APIOptions returns the options that make an HTTP client built by
// google.golang.org/api present [Source]'s certificate. mtlsEndpoint reports
// whether the client should then use its mTLS endpoint where it would use its
// default one.
//
// caller holds the options the client was given by its own caller, which go
// after the returned ones. With an HTTP client among them the client is the
// caller's to configure, and outside the googleapis.com universe the api
// library refuses a client certificate. In both cases the options are empty,
// and so they are when the library fails on caller's options.
//
// That library has a rule of its own: the certificate is off unless
// GOOGLE_API_USE_CLIENT_CERTIFICATE is true, and its certificate sources
// include no workload certificate. So the options hand it the certificate and
// switch the client to the library's newer auth library, which presents a
// certificate it is handed under the gate [Source] describes. The library
// still picks the endpoint by its own rule, which is why the caller has to
// move it.
//
// With no certificate to present the options are empty, and the library
// builds the client by its own rule.
func APIOptions(ctx context.Context, caller ...option.ClientOption) (opts []option.ClientOption, mtlsEndpoint bool, err error) {
	// With the newer auth library switched off, the api library presents a
	// certificate it is handed only under its own rule.
	if off, _ := strconv.ParseBool(os.Getenv(envNewAuthLibOff)); off {
		return nil, false, nil
	}
	if httpClient, universe := callerSettings(ctx, caller); httpClient || universe != defaultUniverse {
		return nil, false, nil
	}
	source, err := Source()
	if err != nil || source == nil {
		return nil, false, err
	}
	// Read as the api library reads it, on purpose: this answer has to agree
	// with the endpoint the library picks. With always it picks the mTLS
	// endpoint itself, any other value but auto keeps the default one, and
	// the older variable is a fallback. auth/gcp.NewClient builds its own
	// client and differs: it rejects an unknown value and does not read the
	// older variable.
	mode := os.Getenv(envMTLSEndpoint)
	if mode == "" {
		mode = os.Getenv(envMTLSEndpointOld)
	}
	opts = []option.ClientOption{internaloption.EnableNewAuthLibrary(), option.WithClientCertSource(source)}
	return opts, mode == "" || strings.EqualFold(mode, "auto"), nil
}

// callerSettings reports whether opts hold an [option.WithHTTPClient], and
// the universe a client built with them is in: the one an option names, or
// else the environment's. The universe is empty when the api library fails
// on opts.
//
// Options cannot be inspected, so it asks the api library. Given an HTTP
// client the library builds nothing: it returns the last client among its
// options, and a default endpoint with the universe in it.
func callerSettings(ctx context.Context, opts []option.ClientOption) (httpClient bool, universe string) {
	universe = os.Getenv(envUniverseDomain)
	if universe == "" {
		universe = defaultUniverse
	}
	if len(opts) == 0 {
		return false, universe
	}
	const prefix = "https://universe.probe."
	probe := &http.Client{}
	probeOpts := append([]option.ClientOption{
		option.WithHTTPClient(probe),
		internaloption.WithDefaultEndpointTemplate(prefix + "UNIVERSE_DOMAIN"),
		// The library has an mTLS endpoint in the googleapis.com universe
		// alone.
		internaloption.WithDefaultMTLSEndpoint(prefix + defaultUniverse),
	}, opts...)
	// The probe reads two settings and leaves the rest to the real call. So
	// it skips validation, which rejects options that only clash with its
	// HTTP client, a quota project for one, and ones the client completes
	// after it asks, a scope for option.ImpersonateCredentials. It leaves out
	// the caller's endpoint, which the library would return in place of the
	// default one. And it has a certificate in hand, so that the library
	// does not search for one of its own, which can run a helper.
	probeOpts = append(probeOpts, internaloption.SkipDialSettingsValidation(), option.WithEndpoint(""), option.WithClientCertSource(noCertificate))
	got, endpoint, err := htransport.NewClient(ctx, probeOpts...)
	// What the library still fails on is its mTLS endpoint outside
	// googleapis.com, and options it takes for no HTTP client at all.
	if err != nil {
		return false, ""
	}
	// No prefix means default endpoints of the caller's own.
	if u, ok := strings.CutPrefix(endpoint, prefix); ok {
		universe = u
	}
	return got != probe, universe
}
