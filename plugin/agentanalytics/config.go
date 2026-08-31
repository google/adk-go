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

package agentanalytics

import (
	"log"
	"time"

	"google.golang.org/api/option"
)

// Logger interface to use standard log operations natively.
type Logger interface {
	Printf(format string, v ...any)
	Println(v ...any)
}

// RetryConfig holds retry parameters for operations like BigQuery Appends.
type RetryConfig struct {
	MaxRetries   int
	InitialDelay time.Duration
	MaxDelay     time.Duration
	Multiplier   float64
}

// Config represents settings mapping to BigQueryLoggerConfig.java.
type Config struct {
	Enabled        bool
	ProjectID      string
	DatasetID      string
	TableName      string
	MaxContentLen  int
	BatchSize      int
	BatchFlushIntv time.Duration
	QueueMaxSize   int
	ClientOptions  []option.ClientOption

	// Fields to cluster the table by.
	ClusteringFields []string

	CustomTags map[string]any

	// Whether to log multi-modal content.
	LogMultiModalContent bool

	// Max time to wait for shutdown.
	ShutdownTimeout time.Duration

	// Injected logger.
	Logger Logger

	// Retry configuration for appending rows.
	RetryConfig RetryConfig
}

// DefaultConfig returns the default configuration for the agent analytics plugin.
func DefaultConfig() Config {
	return Config{
		Enabled:              true,
		MaxContentLen:        500 * 1024,
		DatasetID:            "agent_analytics",
		TableName:            "events",
		ClusteringFields:     []string{"event_type", "agent", "user_id"},
		LogMultiModalContent: true,
		BatchSize:            100,
		BatchFlushIntv:       time.Second,
		ShutdownTimeout:      10 * time.Second,
		QueueMaxSize:         10000,
		CustomTags:           make(map[string]any),
		Logger:               log.Default(),
		RetryConfig: RetryConfig{
			MaxRetries:   3,
			InitialDelay: 100 * time.Millisecond,
			MaxDelay:     10 * time.Second,
			Multiplier:   2.0,
		},
	}
}
