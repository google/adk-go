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

// Package sqlitedsn builds the DSN with which the session services open a
// SQLite database file through the github.com/glebarez/go-sqlite driver.
package sqlitedsn

import "strings"

// uriEscaper escapes the characters that end or encode the path of a SQLite
// URI filename.
var uriEscaper = strings.NewReplacer("%", "%25", "?", "%3F", "#", "%23")

// File returns a DSN that opens the database file at path with the
// parameters in query, an encoded query string without the leading "?".
//
// The driver cuts a plain DSN at its first "?", so a path holding "?" would
// open another file. File returns a SQLite URI filename
// (https://www.sqlite.org/uri.html) instead, with "%", "?" and "#" in path
// percent-encoded, so SQLite opens the file at path whatever its name. SQLite
// also reads query as URI parameters.
func File(path, query string) string {
	prefix := "file:"
	if strings.HasPrefix(path, "/") {
		// An empty authority, so that a path starting with "//" is not read
		// as one.
		prefix = "file://"
	}
	return prefix + uriEscaper.Replace(path) + "?" + query
}
