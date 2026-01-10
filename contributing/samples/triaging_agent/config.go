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

package main

import (
	"os"
	"strconv"
)

const (
	GitHubBaseURL = "https://api.github.com"
)

var (
	GitHubToken         = getEnvOrFail("GITHUB_TOKEN")
	Owner               = getEnvOrDefault("OWNER", "google")
	Repo                = getEnvOrDefault("REPO", "adk-go")
	EventName           = os.Getenv("EVENT_NAME")
	IssueNumber         = os.Getenv("ISSUE_NUMBER")
	IssueTitle          = os.Getenv("ISSUE_TITLE")
	IssueBody           = os.Getenv("ISSUE_BODY")
	IssueCountToProcess = os.Getenv("ISSUE_COUNT_TO_PROCESS")
	IsInteractive       = isInteractiveEnv()
)

func getEnvOrFail(key string) string {
	val := os.Getenv(key)
	if val == "" {
		panic("environment variable " + key + " is not set")
	}
	return val
}

func getEnvOrDefault(key, defaultValue string) string {
	val := os.Getenv(key)
	if val == "" {
		return defaultValue
	}
	return val
}

func isInteractiveEnv() bool {
	val := os.Getenv("INTERACTIVE")
	return val == "1" || val == "true"
}

// ParseNumberString parses a number from the given string, returning the default value if parsing fails.
func ParseNumberString(numberStr string, defaultValue int) int {
	if numberStr == "" {
		return defaultValue
	}
	val, err := strconv.Atoi(numberStr)
	if err != nil {
		return defaultValue
	}
	return val
}
