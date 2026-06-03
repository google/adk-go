// Copyright 2025 Google LLC
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

package artifact

import (
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestArtifactKey(t *testing.T) {
	key := artifactKey{
		AppName:   "testapp",
		UserID:    "testuser",
		SessionID: "testsession",
		FileName:  "testfile",
		Version:   123,
	}
	var key2 artifactKey
	err := key2.Decode(key.Encode())
	if err != nil {
		t.Fatalf("error decoding key:%s", err)
	}
	if diff := cmp.Diff(key, key2); diff != "" {
		t.Errorf("key mismatch (-want +got):\n%s", diff)
	}
}
