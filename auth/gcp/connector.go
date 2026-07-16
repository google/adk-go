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

// connectorRequest is the JSON body for RetrieveCredentials (the connector is
// bound to the URL path, not the body).
type connectorRequest struct {
	UserID       string   `json:"userId,omitempty"`
	Scopes       []string `json:"scopes,omitempty"`
	ContinueURI  string   `json:"continueUri,omitempty"`
	ForceRefresh bool     `json:"forceRefresh,omitempty"`
}

// connectorOperation is the google.longrunning.Operation wrapper the IAM
// Connector service returns. The service does not implement true LROs, so the
// terminal result is read inline from response/metadata. The Any-typed
// response/metadata carry an extra "@type" field that is ignored here.
type connectorOperation struct {
	Done     bool `json:"done"`
	Response *struct {
		Token  string `json:"token"`
		Header string `json:"header"`
	} `json:"response"`
	Metadata *struct {
		URIConsentRequired *consentDetail `json:"uriConsentRequired"`
		ConsentRejected    *struct{}      `json:"consentRejected"`
	} `json:"metadata"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// retrieveConnector calls the IAM Connector service and normalizes its
// Operation-wrapped response.
func (c *Client) retrieveConnector(ctx context.Context, req Request) (retrieveResult, error) {
	url := fmt.Sprintf("%s/v1alpha/%s/credentials:retrieve", c.connectorURL, req.Resource)
	body := connectorRequest{UserID: req.UserID, Scopes: req.Scopes, ContinueURI: req.ContinueURI}

	var op connectorOperation
	if err := c.doPost(ctx, url, body, &op); err != nil {
		return retrieveResult{}, err
	}

	if op.Error != nil {
		return retrieveResult{}, fmt.Errorf("gcp: connector operation failed: %s", op.Error.Message)
	}
	if op.Done {
		// A terminal operation must carry a credential; treat an empty result as
		// an error rather than polling to the timeout.
		if op.Response == nil {
			return retrieveResult{}, fmt.Errorf("gcp: connector operation done but returned no credential for %q", req.Resource)
		}
		return retrieveResult{status: statusOK, token: op.Response.Token, header: op.Response.Header}, nil
	}
	if md := op.Metadata; md != nil {
		switch {
		case md.URIConsentRequired != nil:
			return retrieveResult{
				status:       statusConsentRequired,
				consentURI:   md.URIConsentRequired.AuthorizationURI,
				consentNonce: md.URIConsentRequired.ConsentNonce,
			}, nil
		case md.ConsentRejected != nil:
			return retrieveResult{status: statusRejected}, nil
		}
	}
	// No terminal result and no consent requirement: keep polling.
	return retrieveResult{status: statusPending}, nil
}
