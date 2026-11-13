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

package artifact_test

import (
	"errors"
	"testing"

	"google.golang.org/adk/v2/artifact"
)

func TestValidateFileName(t *testing.T) {
	for _, tc := range []struct {
		name    string
		file    string
		wantErr string
	}{
		{name: "plain name", file: "report.txt"},
		{name: "user scoped name", file: "user:profile.png"},
		{name: "slash", file: "a/b.txt", wantErr: "invalid name: filename cannot contain path separators"},
		{name: "backslash", file: `a\b.txt`, wantErr: "invalid name: filename cannot contain path separators"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := artifact.ValidateFileName(tc.file)
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("ValidateFileName() = %v, want nil", err)
				}
				return
			}
			if err == nil || err.Error() != tc.wantErr || !errors.Is(err, artifact.ErrInputValidation) {
				t.Errorf("ValidateFileName() = %v, want %q wrapping ErrInputValidation", err, tc.wantErr)
			}
		})
	}
}
