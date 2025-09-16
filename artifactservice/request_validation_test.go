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

package artifactservice

import (
	"fmt"
	"reflect"
	"testing"

	"google.golang.org/genai"
)

// Validator describes a type that can validate itself.
type Validator interface {
	Validate() error
}

type ValidatorTestCase struct {
	name       string
	req        Validator
	wantErr    bool
	wantErrMsg string
}

// Test suite for the SaveRequest Validate method
func TestSaveRequest_Validate(t *testing.T) {
	// Define test cases
	testCases := []ValidatorTestCase{
		{
			name: "Valid request from bytes",
			req: &SaveRequest{
				AppName:   "MyApp",
				UserID:    "user-123",
				SessionID: "sess-abc",
				FileName:  "file.txt",
				Part:      genai.NewPartFromBytes([]byte("data"), "text/plain"),
			},
			wantErr: false,
		},
		{
			name: "Valid request from text",
			req: &SaveRequest{
				AppName:   "MyApp",
				UserID:    "user-123",
				SessionID: "sess-abc",
				FileName:  "file.txt",
				Part:      genai.NewPartFromText("data"),
			},
			wantErr: false,
		},
		{
			name: "Missing AppName",
			req: &SaveRequest{
				UserID:    "user-123",
				SessionID: "sess-abc",
				FileName:  "file.txt",
				Part:      genai.NewPartFromBytes([]byte("data"), "text/plain"),
			},
			wantErr:    true,
			wantErrMsg: "invalid save request: missing required fields: AppName",
		},
		{
			name: "Missing multiple fields",
			req: &SaveRequest{
				AppName: "MyApp",
				Part:    genai.NewPartFromBytes([]byte("data"), "text/plain"),
			},
			wantErr:    true,
			wantErrMsg: "invalid save request: missing required fields: UserID, SessionID, FileName",
		},
		{
			name: "Missing Part (nil slice)",
			req: &SaveRequest{
				AppName:   "MyApp",
				UserID:    "user-123",
				SessionID: "sess-abc",
				FileName:  "file.txt",
				Part:      nil,
			},
			wantErr:    true,
			wantErrMsg: "invalid save request: missing required fields: Part",
		},
		{
			name: "Missing Part.Inline (nil slice)",
			req: &SaveRequest{
				AppName:   "MyApp",
				UserID:    "user-123",
				SessionID: "sess-abc",
				FileName:  "file.txt",
				Part:      genai.NewPartFromFunctionCall("example", nil),
			},
			wantErr:    true,
			wantErrMsg: "invalid save request: Part.InlineData or Part.Text have to be set",
		},
		{
			name:       "Completely empty request",
			req:        &SaveRequest{},
			wantErr:    true,
			wantErrMsg: "invalid save request: missing required fields: AppName, UserID, SessionID, FileName, Part",
		},
	}
	executeValidatorTestCases(t, "SaveRequest", testCases)
}

// Test suite for the LoadRequest Validate method
func TestLoadRequest_Validate(t *testing.T) {
	// Define test cases
	testCases := []ValidatorTestCase{
		{
			name: "Valid request",
			req: &LoadRequest{
				AppName:   "MyApp",
				UserID:    "user-123",
				SessionID: "sess-abc",
				FileName:  "file.txt",
			},
			wantErr: false,
		},
		{
			name: "Missing AppName",
			req: &LoadRequest{
				UserID:    "user-123",
				SessionID: "sess-abc",
				FileName:  "file.txt",
			},
			wantErr:    true,
			wantErrMsg: "invalid load request: missing required fields: AppName",
		},
		{
			name: "Missing multiple fields",
			req: &LoadRequest{
				AppName: "MyApp",
			},
			wantErr:    true,
			wantErrMsg: "invalid load request: missing required fields: UserID, SessionID, FileName",
		},
		{
			name:       "Completely empty request",
			req:        &LoadRequest{},
			wantErr:    true,
			wantErrMsg: "invalid load request: missing required fields: AppName, UserID, SessionID, FileName",
		},
	}
	executeValidatorTestCases(t, "LoadRequest", testCases)
}

// Test suite for the DeleteRequest Validate method
func TestDeleteRequest_Validate(t *testing.T) {
	// Define test cases
	testCases := []ValidatorTestCase{
		{
			name: "Valid request",
			req: &DeleteRequest{
				AppName:   "MyApp",
				UserID:    "user-123",
				SessionID: "sess-abc",
				FileName:  "file.txt",
			},
			wantErr: false,
		},
		{
			name: "Missing AppName",
			req: &DeleteRequest{
				UserID:    "user-123",
				SessionID: "sess-abc",
				FileName:  "file.txt",
			},
			wantErr:    true,
			wantErrMsg: "invalid delete request: missing required fields: AppName",
		},
		{
			name: "Missing multiple fields",
			req: &DeleteRequest{
				AppName: "MyApp",
			},
			wantErr:    true,
			wantErrMsg: "invalid delete request: missing required fields: UserID, SessionID, FileName",
		},
		{
			name:       "Completely empty request",
			req:        &DeleteRequest{},
			wantErr:    true,
			wantErrMsg: "invalid delete request: missing required fields: AppName, UserID, SessionID, FileName",
		},
	}
	executeValidatorTestCases(t, "DeleteRequest", testCases)
}

// Test suite for the ListRequest Validate method
func TestListRequest_Validate(t *testing.T) {
	// Define test cases
	testCases := []ValidatorTestCase{
		{
			name: "Valid request",
			req: &ListRequest{
				AppName:   "MyApp",
				UserID:    "user-123",
				SessionID: "sess-abc",
			},
			wantErr: false,
		},
		{
			name: "Missing AppName",
			req: &ListRequest{
				UserID:    "user-123",
				SessionID: "sess-abc",
			},
			wantErr:    true,
			wantErrMsg: "invalid list request: missing required fields: AppName",
		},
		{
			name: "Missing multiple fields",
			req: &ListRequest{
				AppName: "MyApp",
			},
			wantErr:    true,
			wantErrMsg: "invalid list request: missing required fields: UserID, SessionID",
		},
		{
			name:       "Completely empty request",
			req:        &ListRequest{},
			wantErr:    true,
			wantErrMsg: "invalid list request: missing required fields: AppName, UserID, SessionID",
		},
	}
	executeValidatorTestCases(t, "ListRequest", testCases)
}

// Test suite for the VersionsRequest Validate method
func TestVersionsRequest_Validate(t *testing.T) {
	// Define test cases
	testCases := []ValidatorTestCase{
		{
			name: "Valid request",
			req: &VersionsRequest{
				AppName:   "MyApp",
				UserID:    "user-123",
				SessionID: "sess-abc",
				FileName:  "file.txt",
			},
			wantErr: false,
		},
		{
			name: "Missing AppName",
			req: &VersionsRequest{
				UserID:    "user-123",
				SessionID: "sess-abc",
				FileName:  "file.txt",
			},
			wantErr:    true,
			wantErrMsg: "invalid versions request: missing required fields: AppName",
		},
		{
			name: "Missing multiple fields",
			req: &VersionsRequest{
				AppName: "MyApp",
			},
			wantErr:    true,
			wantErrMsg: "invalid versions request: missing required fields: UserID, SessionID, FileName",
		},
		{
			name:       "Completely empty request",
			req:        &VersionsRequest{},
			wantErr:    true,
			wantErrMsg: "invalid versions request: missing required fields: AppName, UserID, SessionID, FileName",
		},
	}
	executeValidatorTestCases(t, "VersionsRequest", testCases)
}

func executeValidatorTestCases(t *testing.T, requestTypeName string, testCases []ValidatorTestCase) {
	// Run the tests
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%s_%s", requestTypeName, tc.name), func(t *testing.T) {
			err := tc.req.Validate()

			if (err != nil) != tc.wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, tc.wantErr)
				return
			}
			if err != nil && err.Error() != tc.wantErrMsg {
				// NOTE: This simple string comparison works because our function produces a sorted, predictable output.
				t.Errorf("Validate() error msg = %q, wantErrMsg %q", err.Error(), tc.wantErrMsg)
			}
		})
	}
}

// Test suite for the reusable helper function using a slice of structs
func TestValidateRequiredStrings(t *testing.T) {
	testCases := []struct {
		name  string
		input []requiredField
		want  []string
	}{
		{
			name: "No missing fields",
			input: []requiredField{
				{Name: "FieldA", Value: "valueA"},
				{Name: "FieldB", Value: "valueB"},
			},
			want: nil,
		},
		{
			name: "One missing field",
			input: []requiredField{
				{Name: "FieldA", Value: "valueA"},
				{Name: "FieldB", Value: ""},
			},
			want: []string{"FieldB"},
		},
		{
			name: "Multiple missing fields",
			input: []requiredField{
				{Name: "FieldA", Value: ""},
				{Name: "FieldB", Value: "valueB"},
				{Name: "FieldC", Value: ""},
			},
			// The order now matches the input order
			want: []string{"FieldA", "FieldC"},
		},
		{
			name:  "Empty input slice",
			input: []requiredField{},
			want:  nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := validateRequiredStrings(tc.input)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("validateRequiredStrings() = %v, want %v", got, tc.want)
			}
		})
	}
}
