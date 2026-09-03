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

package adkrest

import "net/http"

// DefaultMaxPayloadSize is the default maximum request body size (10 MiB)
// applied when ServerConfig.MaxPayloadSize is not set. It mitigates
// memory-exhaustion denial of service caused by oversized request bodies.
const DefaultMaxPayloadSize int64 = 10 << 20

// MaxBytesMiddleware limits the size of request bodies to maxBytes bytes by
// wrapping the request body in http.MaxBytesReader. Requests whose body
// exceeds the limit fail to be read: handlers that decode the body observe a
// *http.MaxBytesError (rendered by http.Error as 400 with "http: request body
// too large"). The middleware itself does not write a status code; that is left
// to the handler reading the body.
func MaxBytesMiddleware(maxBytes int64) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			r.Body = http.MaxBytesReader(w, r.Body, maxBytes)
			next.ServeHTTP(w, r)
		})
	}
}
