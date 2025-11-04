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

package util

import (
	"flag"
	"strings"
)

// FormatFlagUsage returns a string containing the usage information for the given FlagSet.
func FormatFlagUsage(fs *flag.FlagSet) string {
	var b strings.Builder
	o := fs.Output()
	fs.SetOutput(&b)
	fs.PrintDefaults()
	fs.SetOutput(o)
	return b.String()
}
