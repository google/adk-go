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

// package config provides configs for the REST API server.
package config

import (
	"fmt"
	"os"
	"strconv"

	"github.com/joho/godotenv"
	"github.com/rs/cors"
)

// ADKAPIServerConfigs contains the configs for the ADK API server.
type ADKAPIServerConfigs struct {
	Env          string
	Port         int
	Cors         cors.Cors
	GeminiAPIKey string
}

// LoadConfig parses the arguments for the ADK API server and returns parsed configs.
func LoadConfig() (*ADKAPIServerConfigs, error) {
	config := &ADKAPIServerConfigs{}
	config.Env = os.Getenv("ENV")

	if err := godotenv.Load(); err != nil && config.Env == "" {
		return nil, err
	}

	allowedOrigin, _ := os.LookupEnv("ALLOWED_ORIGIN")
	config.Cors = *cors.New(cors.Options{
		AllowedOrigins: []string{allowedOrigin},
	})

	if webPort, ok := os.LookupEnv("PORT"); ok {
		port, err := strconv.ParseInt(webPort, 10, 32)
		if err != nil {
			return nil, err
		}
		config.Port = int(port)
	} else {
		config.Port = 8080
	}

	apiKey, ok := os.LookupEnv("GEMINI_API_KEY")
	if !ok {
		return nil, fmt.Errorf("GEMINI_API_KEY is not set")
	}
	config.GeminiAPIKey = apiKey
	return config, nil
}
