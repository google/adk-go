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

package database

import (
	"os"
	"testing"
	"time"
)

func TestMain(m *testing.M) {
	// Force a stable timezone for database tests so timestamp round-trips
	// don't depend on the machine's local TZ.
	origLocal := time.Local
	time.Local = time.UTC
	code := m.Run()
	time.Local = origLocal
	os.Exit(code)
}
