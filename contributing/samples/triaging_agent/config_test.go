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
	"fmt"
	"os"
	"testing"
)

func TestGetEnvOrFail(t *testing.T) {
	tests := []struct {
		name      string
		key       string
		setValue  string
		wantPanic bool
		wantValue string
		cleanup   bool
	}{
		{
			name:      "environment variable is set",
			key:       "TEST_VAR",
			setValue:  "test-value",
			wantPanic: false,
			wantValue: "test-value",
			cleanup:   true,
		},
		{
			name:      "environment variable is empty string",
			key:       "TEST_VAR_EMPTY",
			setValue:  "",
			wantPanic: true,
			cleanup:   true,
		},
		{
			name:      "environment variable is not set",
			key:       "TEST_VAR_NOT_SET",
			setValue:  "",
			wantPanic: true,
			cleanup:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Clean up before test
			err := os.Unsetenv(tt.key)
			if err != nil {
				panic(fmt.Errorf("failed to unset environment variable %s, error %w", tt.key, err))
			}

			// Setting the environment variable if needed
			if tt.setValue != "" || !tt.wantPanic {
				err := os.Setenv(tt.key, tt.setValue)
				if err != nil {
					panic(fmt.Errorf("failed to set environment variable %s, error %w", tt.key, err))
					return
				}
			}

			// Clean up after test
			defer func() {
				if tt.cleanup {
					err := os.Unsetenv(tt.key)
					if err != nil {
						panic(fmt.Errorf("failed to unset environment variable %s, error %w", tt.key, err))
						return
					}
				}
			}()

			// Test for panic
			if tt.wantPanic {
				defer func() {
					if r := recover(); r == nil {
						t.Errorf("getEnvOrFail() expected panic, but did not panic")
					}
				}()
			}

			got := getEnvOrFail(tt.key)

			if !tt.wantPanic {
				if got != tt.wantValue {
					t.Errorf("getEnvOrFail() = %v, want %v", got, tt.wantValue)
				}
			}
		})
	}
}

func TestGetEnvOrDefault(t *testing.T) {
	tests := []struct {
		name         string
		key          string
		defaultValue string
		setValue     string
		want         string
		cleanup      bool
	}{
		{
			name:         "environment variable is set",
			key:          "TEST_VAR",
			defaultValue: "default-value",
			setValue:     "actual-value",
			want:         "actual-value",
			cleanup:      true,
		},
		{
			name:         "environment variable is not set",
			key:          "TEST_VAR_NOT_SET",
			defaultValue: "default-value",
			setValue:     "",
			want:         "default-value",
			cleanup:      false,
		},
		{
			name:         "environment variable is empty string",
			key:          "TEST_VAR_EMPTY",
			defaultValue: "default-value",
			setValue:     "",
			want:         "default-value",
			cleanup:      true,
		},
		{
			name:         "default value is empty string",
			key:          "TEST_VAR",
			defaultValue: "",
			setValue:     "",
			want:         "",
			cleanup:      true,
		},
		{
			name:         "environment variable overrides default",
			key:          "TEST_VAR",
			defaultValue: "default",
			setValue:     "override",
			want:         "override",
			cleanup:      true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Clean up before test
			err := os.Unsetenv(tt.key)
			if err != nil {
				panic(fmt.Errorf("failed to unset environment variable %s, error %w", tt.key, err))
				return
			}

			// Set the environment variable if needed
			if tt.setValue != "" || tt.name == "environment variable is empty string" {
				err := os.Setenv(tt.key, tt.setValue)
				if err != nil {
					panic(fmt.Errorf("failed to set environment variable %s, error %w", tt.key, err))
					return
				}
			}

			// Clean up after test
			defer func() {
				if tt.cleanup {
					err := os.Unsetenv(tt.key)
					if err != nil {
						panic(fmt.Errorf("failed to unset environment variable %s, error %w", tt.key, err))
						return
					}
				}
			}()

			got := getEnvOrDefault(tt.key, tt.defaultValue)
			if got != tt.want {
				t.Errorf("getEnvOrDefault() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIsInteractiveEnv(t *testing.T) {
	tests := []struct {
		name     string
		setValue string
		want     bool
		cleanup  bool
	}{
		{
			name:     "INTERACTIVE is '1'",
			setValue: "1",
			want:     true,
			cleanup:  true,
		},
		{
			name:     "INTERACTIVE is 'true'",
			setValue: "true",
			want:     true,
			cleanup:  true,
		},
		{
			name:     "INTERACTIVE is 'True' (case sensitive, should be false)",
			setValue: "True",
			want:     false,
			cleanup:  true,
		},
		{
			name:     "INTERACTIVE is 'TRUE' (case sensitive, should be false)",
			setValue: "TRUE",
			want:     false,
			cleanup:  true,
		},
		{
			name:     "INTERACTIVE is '0'",
			setValue: "0",
			want:     false,
			cleanup:  true,
		},
		{
			name:     "INTERACTIVE is 'false'",
			setValue: "false",
			want:     false,
			cleanup:  true,
		},
		{
			name:     "INTERACTIVE is empty string",
			setValue: "",
			want:     false,
			cleanup:  true,
		},
		{
			name:     "INTERACTIVE is not set",
			setValue: "",
			want:     false,
			cleanup:  false,
		},
		{
			name:     "INTERACTIVE is 'yes'",
			setValue: "yes",
			want:     false,
			cleanup:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Clean up before test
			err := os.Unsetenv("INTERACTIVE")
			if err != nil {
				panic(fmt.Errorf("failed to unset environment variable %s, error %w", tt.setValue, err))
				return
			}

			// Set the environment variable if needed
			if tt.setValue != "" || tt.name == "INTERACTIVE is empty string" {
				err = os.Setenv("INTERACTIVE", tt.setValue)
				if err != nil {
					panic(fmt.Errorf("failed to set environment variable %s, error %w", tt.setValue, err))
					return
				}
			}

			// Clean up after test
			defer func() {
				if tt.cleanup {
					err := os.Unsetenv("INTERACTIVE")
					if err != nil {
						panic(fmt.Errorf("failed to unset environment variable %s, error %w", tt.setValue, err))
						return
					}
				}
			}()

			got := isInteractiveEnv()
			if got != tt.want {
				t.Errorf("isInteractiveEnv() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestParseNumberString(t *testing.T) {
	tests := []struct {
		name         string
		numberStr    string
		defaultValue int
		want         int
	}{
		{
			name:         "valid positive number",
			numberStr:    "42",
			defaultValue: 0,
			want:         42,
		},
		{
			name:         "valid negative number",
			numberStr:    "-10",
			defaultValue: 0,
			want:         -10,
		},
		{
			name:         "valid zero",
			numberStr:    "0",
			defaultValue: 5,
			want:         0,
		},
		{
			name:         "empty string returns default",
			numberStr:    "",
			defaultValue: 10,
			want:         10,
		},
		{
			name:         "invalid string returns default",
			numberStr:    "not-a-number",
			defaultValue: 99,
			want:         99,
		},
		{
			name:         "string with letters returns default",
			numberStr:    "123abc",
			defaultValue: 50,
			want:         50,
		},
		{
			name:         "string with spaces returns default",
			numberStr:    " 123 ",
			defaultValue: 30,
			want:         30,
		},
		{
			name:         "decimal number returns default",
			numberStr:    "3.14",
			defaultValue: 0,
			want:         0,
		},
		{
			name:         "very large number",
			numberStr:    "999999",
			defaultValue: 0,
			want:         999999,
		},
		{
			name:         "default value is negative",
			numberStr:    "",
			defaultValue: -5,
			want:         -5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ParseNumberString(tt.numberStr, tt.defaultValue)
			if got != tt.want {
				t.Errorf("ParseNumberString(%q, %d) = %d, want %d", tt.numberStr, tt.defaultValue, got, tt.want)
			}
		})
	}
}
