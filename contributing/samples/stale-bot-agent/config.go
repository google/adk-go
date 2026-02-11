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
	"log"
	"os"
	"strconv"
)

var (
	GitHubBaseURL = "https://api.github.com"
	GitHubToken   string

	Owner string
	Repo  string

	// Labels
	StaleLabelName            = "stale"
	RequestClarificationLabel = "request clarification"

	// Thresholds (hours)
	STALE_HOURS_THRESHOLD             float64
	CLOSE_HOURS_AFTER_STALE_THRESHOLD float64

	// Performance
	ConcurrencyLimit int

	// GraphQL limits
	GraphQLCommentLimit  int
	GraphQLEditLimit     int
	GraphQLTimelineLimit int

	// Rate limiting
	SleepBetweenChunks float64
)

func InitConfig() {
	GitHubToken = os.Getenv("GITHUB_TOKEN")
	log.Printf("GITHUB_TOKEN length: %d", len(GitHubToken))
	if GitHubToken == "" {
		log.Fatal("GITHUB_TOKEN environment variable not set")
	}

	// Repo
	Owner = getEnv("OWNER", "google")
	Repo = getEnv("REPO", "adk-go")

	// Thresholds (hours)
	STALE_HOURS_THRESHOLD = getEnvFloat("STALE_HOURS_THRESHOLD", 168.0)
	CLOSE_HOURS_AFTER_STALE_THRESHOLD = getEnvFloat("CLOSE_HOURS_AFTER_STALE_THRESHOLD", 168.0)

	// Performance
	ConcurrencyLimit = getEnvInt("CONCURRENCY_LIMIT", 3)

	GraphQLCommentLimit = getEnvInt("GRAPHQL_COMMENT_LIMIT", 30)
	GraphQLEditLimit = getEnvInt("GRAPHQL_EDIT_LIMIT", 10)
	GraphQLTimelineLimit = getEnvInt("GRAPHQL_TIMELINE_LIMIT", 20)

	// Rate limiting
	SleepBetweenChunks = getEnvFloat("SLEEP_BETWEEN_CHUNKS", 1.5)

	// Sanity log
	log.Printf("Config loaded → repo=%s/%s stale=%.2fh close=%.2fh", Owner, Repo, STALE_HOURS_THRESHOLD, CLOSE_HOURS_AFTER_STALE_THRESHOLD)
}

func getEnv(key, fallback string) string {
	if v, ok := os.LookupEnv(key); ok {
		return v
	}
	return fallback
}

func getEnvInt(key string, fallback int) int {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	i, err := strconv.Atoi(v)
	if err != nil {
		return fallback
	}
	return i
}

func getEnvFloat(key string, fallback float64) float64 {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	f, err := strconv.ParseFloat(v, 64)
	if err != nil {
		return fallback
	}
	return f
}
