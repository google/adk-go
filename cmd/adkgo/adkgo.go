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

// Package main is an entry point for CLI.
package main

import (
	"google.golang.org/adk/cmd/adkgo/root"
	_ "google.golang.org/adk/cmd/adkgo/root/build"
	_ "google.golang.org/adk/cmd/adkgo/root/build/webui"
	_ "google.golang.org/adk/cmd/adkgo/root/deploy"
	_ "google.golang.org/adk/cmd/adkgo/root/deploy/cloudrun"
	_ "google.golang.org/adk/cmd/adkgo/root/run"
	_ "google.golang.org/adk/cmd/adkgo/root/run/local"
)

func main() {
	root.Execute()
}
