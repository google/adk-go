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
	"errors"
	"fmt"
)

// agentIdentityResponse mirrors the RetrieveCredentialsResponse "result" oneof.
type agentIdentityResponse struct {
	Success            *credentialPayload `json:"success"`
	Pending            *struct{}          `json:"pending"`
	URIConsentRequired *consentDetail     `json:"uriConsentRequired"`
	ConsentRejected    *struct{}          `json:"consentRejected"`
}

// consentDetail is the shared uri-consent payload across both services.
type consentDetail struct {
	AuthorizationURI string `json:"authorizationUri"`
	ConsentNonce     string `json:"consentNonce"`
}

// result collapses the response's "result" oneof into an outcome, erroring if
// the service returned no recognized arm.
// The request is unused here and taken anyway, to match connectorOperation.result:
// that one redacts caller-supplied values out of the service's message, and this
// response shape carries no service-controlled text to redact. Dropping the
// parameter would make the next arm added here look already covered.
func (r agentIdentityResponse) result(_ Request) (outcome, error) {
	switch {
	case r.Success != nil:
		return credOutcome{header: r.Success.Header, token: r.Success.Token}, nil
	case r.URIConsentRequired != nil:
		return consentOutcome{authURI: r.URIConsentRequired.AuthorizationURI, nonce: r.URIConsentRequired.ConsentNonce}, nil
	case r.ConsentRejected != nil:
		return rejectedOutcome{}, nil
	case r.Pending != nil:
		return pendingOutcome{}, nil
	default:
		return nil, errors.New("gcp: agent identity returned an empty result")
	}
}

// retrieveAgentIdentity calls the Agent Identity service, whose response is
// returned synchronously (no long-running-operation wrapper).
func (c *Client) retrieveAgentIdentity(ctx context.Context, req Request) (outcome, error) {
	url := fmt.Sprintf("%s/v1/%s/credentials:retrieve", c.agentIdentityURL, req.Resource)
	body := retrieveRequest{UserID: req.UserID, Scopes: req.Scopes, ContinueURI: req.ContinueURI}

	var out agentIdentityResponse
	if err := c.doPost(ctx, url, body, &out, req.UserID, req.ContinueURI); err != nil {
		return nil, err
	}
	return out.result(req)
}
