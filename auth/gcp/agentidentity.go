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

package gcp

import (
	"context"
	"fmt"
)

// agentIdentityRequest is the JSON body for RetrieveCredentials (the auth
// provider is bound to the URL path, not the body).
type agentIdentityRequest struct {
	UserID      string   `json:"userId,omitempty"`
	Scopes      []string `json:"scopes,omitempty"`
	ContinueURI string   `json:"continueUri,omitempty"`
}

// agentIdentityResponse mirrors the RetrieveCredentialsResponse "result" oneof.
type agentIdentityResponse struct {
	Success *struct {
		Token  string `json:"token"`
		Header string `json:"header"`
	} `json:"success"`
	Pending            *struct{}      `json:"pending"`
	URIConsentRequired *consentDetail `json:"uriConsentRequired"`
	ConsentRejected    *struct{}      `json:"consentRejected"`
}

// consentDetail is the shared uri-consent payload across both services.
type consentDetail struct {
	AuthorizationURI string `json:"authorizationUri"`
	ConsentNonce     string `json:"consentNonce"`
}

// retrieveAgentIdentity calls the Agent Identity service, whose response is
// returned synchronously (no long-running-operation wrapper).
func (c *Client) retrieveAgentIdentity(ctx context.Context, req Request) (retrieveResult, error) {
	url := fmt.Sprintf("%s/v1/%s/credentials:retrieve", c.agentIdentityURL, req.Resource)
	body := agentIdentityRequest{UserID: req.UserID, Scopes: req.Scopes, ContinueURI: req.ContinueURI}

	var out agentIdentityResponse
	if err := c.doPost(ctx, url, body, &out); err != nil {
		return retrieveResult{}, err
	}

	switch {
	case out.Success != nil:
		return retrieveResult{status: statusOK, token: out.Success.Token, header: out.Success.Header}, nil
	case out.URIConsentRequired != nil:
		return retrieveResult{
			status:       statusConsentRequired,
			consentURI:   out.URIConsentRequired.AuthorizationURI,
			consentNonce: out.URIConsentRequired.ConsentNonce,
		}, nil
	case out.ConsentRejected != nil:
		return retrieveResult{status: statusRejected}, nil
	case out.Pending != nil:
		return retrieveResult{status: statusPending}, nil
	default:
		return retrieveResult{}, fmt.Errorf("gcp: agent identity returned an empty result for %q", req.Resource)
	}
}
