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

// Package gcp is a hand-rolled REST client for the Google Cloud Agent Identity
// and IAM Connector credential services. Given a resource name it retrieves an
// end-user credential and maps it to an [auth.Credential], polling while the
// service reports a non-interactive "pending" state and surfacing interactive
// consent as an [auth.ConsentRequiredError].
//
// No generated Go client libraries exist for these (preview) services and their
// surface is a single RPC, so the client is hand-rolled over net/http to keep
// dependencies light. Calls to the credential services are authenticated with
// Application Default Credentials (cloud-platform scope) unless a custom
// *http.Client is supplied.
//
// This package holds only the transport-level client. The [auth.CredentialProvider]
// that resolves the acting user from the invocation context is a separate,
// higher layer.
package gcp
