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
// library refuses a client certificate. In both cases the options are empty.
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
// else the environment's.
//
// Options cannot be inspected, so it asks the api library. Given an HTTP
// client the library builds nothing: it returns the last client among its
// options, and the default endpoint with the universe filled in.
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
	got, endpoint, err := htransport.NewClient(ctx, append([]option.ClientOption{
		option.WithHTTPClient(probe),
		internaloption.WithDefaultEndpointTemplate(prefix + "UNIVERSE_DOMAIN"),
	}, opts...)...)
	// An error means options the library rejects next to an HTTP client, a
	// quota project for one. They fail the real call too if opts hold a
	// client.
	if err != nil {
		return false, universe
	}
	// No prefix means an endpoint of the caller's, which wins over the mTLS
	// one whatever the universe.
	if u, ok := strings.CutPrefix(endpoint, prefix); ok {
		universe = u
	}
	return got != probe, universe
}
