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

// connectorOperation is the google.longrunning.Operation wrapper the IAM
// Connector service returns. The service does not implement true LROs, so the
// terminal result is read inline from response/metadata. The Any-typed
// response/metadata carry an extra "@type" field that is ignored here.
type connectorOperation struct {
	Done     bool               `json:"done"`
	Response *credentialPayload `json:"response"`
	Metadata *struct {
		ConsentPending     *struct{}      `json:"consentPending"`
		URIConsentRequired *consentDetail `json:"uriConsentRequired"`
		ConsentRejected    *struct{}      `json:"consentRejected"`
	} `json:"metadata"`
	Error *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

// result collapses the Operation-wrapped response into an outcome.
func (o connectorOperation) result(resource string) (outcome, error) {
	if o.Error != nil {
		if o.Error.Message != "" {
			// Same treatment doPost gives a response body: the message is
			// service-controlled and otherwise bypasses both the cap and escaping.
			return nil, fmt.Errorf("gcp: connector operation failed (code %d): %q", o.Error.Code, truncateForError(o.Error.Message))
		}
		return nil, fmt.Errorf("gcp: connector operation failed (code %d)", o.Error.Code)
	}
	if o.Done {
		// A terminal operation must carry a credential; treat an empty result as
		// an error rather than polling to the timeout.
		if o.Response == nil {
			return nil, fmt.Errorf("gcp: connector operation done but returned no credential for %q", resource)
		}
		return credOutcome{header: o.Response.Header, token: o.Response.Token, expiresAt: parseExpireTime(o.Response.ExpireTime)}, nil
	}
	if md := o.Metadata; md != nil {
		switch {
		case md.URIConsentRequired != nil:
			return consentOutcome{authURI: md.URIConsentRequired.AuthorizationURI, nonce: md.URIConsentRequired.ConsentNonce}, nil
		case md.ConsentRejected != nil:
			return rejectedOutcome{}, nil
		case md.ConsentPending != nil:
			return pendingOutcome{}, nil
		}
	}
	// Absent/unknown status → pending: consent_pending means "just retry", and a
	// non-terminal operation should keep being polled.
	return pendingOutcome{}, nil
}

// retrieveConnector calls the IAM Connector service and normalizes its
// Operation-wrapped response.
func (c *Client) retrieveConnector(ctx context.Context, req Request) (outcome, error) {
	url := fmt.Sprintf("%s/v1alpha/%s/credentials:retrieve", c.connectorURL, req.Resource)
	body := retrieveRequest{UserID: req.UserID, Scopes: req.Scopes, ContinueURI: req.ContinueURI}

	var op connectorOperation
	if err := c.doPost(ctx, url, body, &op); err != nil {
		return nil, err
	}
	return op.result(req.Resource)
}
