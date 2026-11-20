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
	"errors"
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
			wantErrMsg: "invalid save request: Part.InlineData or Part.Text has to be set",
		},
		{
			name:       "Completely empty request",
			req:        &SaveRequest{},
			wantErr:    true,
			wantErrMsg: "invalid save request: missing required fields: AppName, UserID, SessionID, FileName, Part",
		},
		{
			name: "FileName with path separator",
			req: &SaveRequest{
				AppName:   "MyApp",
				UserID:    "user-123",
				SessionID: "sess-abc",
				FileName:  "path/to/file.txt",
				Part:      genai.NewPartFromBytes([]byte("data"), "text/plain"),
			},
			wantErr:    true,
			wantErrMsg: "invalid name: filename cannot contain path separators",
		},
		{
			// An artifact reference part carries neither text nor inline data. The
			// shared contract must reject it so backends that don't understand
			// references (e.g. GCS) keep their behavior; only inMemoryService.Save
			// tolerates it via its own scope check.
			name: "Artifact reference part is rejected by the shared contract",
			req: &SaveRequest{
				AppName:   "MyApp",
				UserID:    "user-123",
				SessionID: "sess-abc",
				FileName:  "alias.txt",
				Part: &genai.Part{FileData: &genai.FileData{
					FileURI: "artifact://apps/MyApp/users/user-123/sessions/sess-abc/artifacts/file.txt/versions/1",
				}},
			},
			wantErr:    true,
			wantErrMsg: "invalid save request: Part.InlineData or Part.Text has to be set",
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
		{
			name: "FileName with path separator",
			req: &LoadRequest{
				AppName:   "MyApp",
				UserID:    "user-123",
				SessionID: "sess-abc",
				FileName:  "a/b.txt",
			},
			wantErr:    true,
			wantErrMsg: "invalid name: filename cannot contain path separators",
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
		{
			name: "FileName with path separator",
			req: &DeleteRequest{
				AppName:   "MyApp",
				UserID:    "user-123",
				SessionID: "sess-abc",
				FileName:  "dir/file.txt",
			},
			wantErr:    true,
			wantErrMsg: "invalid name: filename cannot contain path separators",
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
		{
			name: "FileName with path separator",
			req: &VersionsRequest{
				AppName:   "MyApp",
				UserID:    "user-123",
				SessionID: "sess-abc",
				FileName:  "folder/file.txt",
			},
			wantErr:    true,
			wantErrMsg: "invalid name: filename cannot contain path separators",
		},
	}
	executeValidatorTestCases(t, "VersionsRequest", testCases)
}

// Test suite for the GetArtifactVersionRequest Validate method
func TestGetArtifactVersionRequest_Validate(t *testing.T) {
	// Define test cases
	testCases := []ValidatorTestCase{
		{
			name: "Valid request",
			req: &GetArtifactVersionRequest{
				AppName:   "MyApp",
				UserID:    "user-123",
				SessionID: "sess-abc",
				FileName:  "file.txt",
			},
			wantErr: false,
		},
		{
			name: "Missing AppName",
			req: &GetArtifactVersionRequest{
				UserID:    "user-123",
				SessionID: "sess-abc",
				FileName:  "file.txt",
			},
			wantErr:    true,
			wantErrMsg: "invalid get artifact version request: missing required fields: AppName",
		},
		{
			name: "Missing multiple fields",
			req: &GetArtifactVersionRequest{
				AppName: "MyApp",
			},
			wantErr:    true,
			wantErrMsg: "invalid get artifact version request: missing required fields: UserID, SessionID, FileName",
		},
		{
			name:       "Completely empty request",
			req:        &GetArtifactVersionRequest{},
			wantErr:    true,
			wantErrMsg: "invalid get artifact version request: missing required fields: AppName, UserID, SessionID, FileName",
		},
		{
			name: "FileName with path separator",
			req: &GetArtifactVersionRequest{
				AppName:   "MyApp",
				UserID:    "user-123",
				SessionID: "sess-abc",
				FileName:  "folder/file.txt",
			},
			wantErr:    true,
			wantErrMsg: "invalid name: filename cannot contain path separators",
		},
	}
	executeValidatorTestCases(t, "GetArtifactVersionRequest", testCases)
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

// TestListRequest_Validate_EmptySessionID checks that an empty SessionID asks
// for user-scoped names only, while a request that also lacks another field
// still names SessionID among the missing fields.
func TestListRequest_Validate_EmptySessionID(t *testing.T) {
	testCases := []ValidatorTestCase{
		{
			name:    "empty SessionID lists user-scoped names",
			req:     &ListRequest{AppName: "MyApp", UserID: "user-123"},
			wantErr: false,
		},
		{
			name:       "empty SessionID and missing AppName",
			req:        &ListRequest{UserID: "user-123"},
			wantErr:    true,
			wantErrMsg: "invalid list request: missing required fields: AppName, SessionID",
		},
	}
	executeValidatorTestCases(t, "ListRequest", testCases)
}

// TestSaveRequest_Validate_EmptyTextAndNonContentParts checks that a part
// whose every field is zero is accepted as empty text, and that a part with no
// text and no inline data is still refused when any other field is set.
func TestSaveRequest_Validate_EmptyTextAndNonContentParts(t *testing.T) {
	const wantErrMsg = "invalid save request: Part.InlineData or Part.Text has to be set"
	saveRequest := func(part *genai.Part) *SaveRequest {
		return &SaveRequest{AppName: "MyApp", UserID: "user-123", SessionID: "sess-abc", FileName: "file.txt", Part: part}
	}
	testCases := []ValidatorTestCase{
		{
			name:    "empty text",
			req:     saveRequest(genai.NewPartFromText("")),
			wantErr: false,
		},
		{
			name:       "function response",
			req:        saveRequest(&genai.Part{FunctionResponse: &genai.FunctionResponse{Name: "fn"}}),
			wantErr:    true,
			wantErrMsg: wantErrMsg,
		},
		{
			name:       "executable code",
			req:        saveRequest(&genai.Part{ExecutableCode: &genai.ExecutableCode{Code: "x"}}),
			wantErr:    true,
			wantErrMsg: wantErrMsg,
		},
		{
			name:       "code execution result",
			req:        saveRequest(&genai.Part{CodeExecutionResult: &genai.CodeExecutionResult{Output: "x"}}),
			wantErr:    true,
			wantErrMsg: wantErrMsg,
		},
		{
			name:       "thought signature",
			req:        saveRequest(&genai.Part{ThoughtSignature: []byte("sig")}),
			wantErr:    true,
			wantErrMsg: wantErrMsg,
		},
		{
			name:       "tool call",
			req:        saveRequest(&genai.Part{ToolCall: &genai.ToolCall{ID: "1"}}),
			wantErr:    true,
			wantErrMsg: wantErrMsg,
		},
		{
			name:       "tool response",
			req:        saveRequest(&genai.Part{ToolResponse: &genai.ToolResponse{ID: "1"}}),
			wantErr:    true,
			wantErrMsg: wantErrMsg,
		},
		{
			name:       "thought flag",
			req:        saveRequest(&genai.Part{Thought: true}),
			wantErr:    true,
			wantErrMsg: wantErrMsg,
		},
		{
			name:       "audio transcription",
			req:        saveRequest(&genai.Part{AudioTranscription: &genai.Transcription{Text: "hello"}}),
			wantErr:    true,
			wantErrMsg: wantErrMsg,
		},
		{
			name:       "non-reference file data",
			req:        saveRequest(genai.NewPartFromURI("gs://bucket/file.txt", "text/plain")),
			wantErr:    true,
			wantErrMsg: wantErrMsg,
		},
	}
	executeValidatorTestCases(t, "SaveRequest", testCases)
}

// TestRequestValidate_PathSafetyAndSentinel checks that every Validate error
// wraps ErrInputValidation with its exact text, that the app name, user ID and
// session ID are path-checked in that order, and that a "user:" file name makes
// the session ID optional.
func TestRequestValidate_PathSafetyAndSentinel(t *testing.T) {
	textPart := genai.NewPartFromText("data")
	testCases := []struct {
		name string
		req  Validator
		// want is the exact error text; empty means the request is valid.
		want string
	}{
		// Existing errors keep their text and wrap the sentinel.
		{
			name: "save missing fields",
			req:  &SaveRequest{AppName: "app"},
			want: "invalid save request: missing required fields: UserID, SessionID, FileName, Part",
		},
		{
			name: "save unsupported part",
			req:  &SaveRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "f", Part: genai.NewPartFromFunctionCall("fn", nil)},
			want: "invalid save request: Part.InlineData or Part.Text has to be set",
		},
		{
			name: "save reference part",
			req:  &SaveRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "f", Part: genai.NewPartFromURI("artifact://apps/app/users/u/sessions/s/artifacts/g/versions/1", "")},
			want: "invalid save request: Part.InlineData or Part.Text has to be set",
		},
		{
			name: "save file name with slash",
			req:  &SaveRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "a/b.txt", Part: textPart},
			want: "invalid name: filename cannot contain path separators",
		},
		{
			name: "load file name with backslash",
			req:  &LoadRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "a\\b.txt"},
			want: "invalid name: filename cannot contain path separators",
		},
		{
			name: "load missing fields",
			req:  &LoadRequest{},
			want: "invalid load request: missing required fields: AppName, UserID, SessionID, FileName",
		},
		{
			name: "delete missing fields",
			req:  &DeleteRequest{AppName: "app", FileName: "f"},
			want: "invalid delete request: missing required fields: UserID, SessionID",
		},
		{
			name: "list missing fields",
			req:  &ListRequest{},
			want: "invalid list request: missing required fields: AppName, UserID, SessionID",
		},
		{
			name: "versions missing fields",
			req:  &VersionsRequest{UserID: "u"},
			want: "invalid versions request: missing required fields: AppName, SessionID, FileName",
		},
		{
			name: "get artifact version missing fields",
			req:  &GetArtifactVersionRequest{AppName: "app", UserID: "u", SessionID: "s"},
			want: "invalid get artifact version request: missing required fields: FileName",
		},
		// Path checks run on the app name, user ID and session ID in that order.
		{
			name: "save app name checked first",
			req:  &SaveRequest{AppName: "/bad", UserID: "../bad", SessionID: "victim/apps", FileName: "f.txt", Part: textPart},
			want: "app_name must not be an absolute path or start with a slash.",
		},
		{
			name: "load user ID checked before session ID",
			req:  &LoadRequest{AppName: "ok", UserID: "C:/bad", SessionID: "../bad", FileName: "f.txt"},
			want: "user_id must not be drive-qualified.",
		},
		{
			name: "delete session ID checked",
			req:  &DeleteRequest{AppName: "ok", UserID: "ok", SessionID: "victim/versions", FileName: "f.txt"},
			want: "session_id must not contain reserved path segments.",
		},
		{
			name: "versions null byte in user ID",
			req:  &VersionsRequest{AppName: "ok", UserID: "u\x00", SessionID: "s", FileName: "f.txt"},
			want: "user_id must not contain null bytes.",
		},
		{
			name: "get artifact version traversal in session ID",
			req:  &GetArtifactVersionRequest{AppName: "ok", UserID: "u", SessionID: "..", FileName: "f.txt"},
			want: "session_id must not contain traversal segments.",
		},
		// A "user:" file name makes the session ID optional and unchecked.
		{
			name: "save user name without session",
			req:  &SaveRequest{AppName: "app", UserID: "u", FileName: "user:safe.txt", Part: textPart},
		},
		{
			name: "load user name without session",
			req:  &LoadRequest{AppName: "app", UserID: "u", FileName: "user:safe.txt"},
		},
		{
			name: "delete user name without session",
			req:  &DeleteRequest{AppName: "app", UserID: "u", FileName: "user:safe.txt"},
		},
		{
			name: "versions user name without session",
			req:  &VersionsRequest{AppName: "app", UserID: "u", FileName: "user:safe.txt"},
		},
		{
			name: "get artifact version user name without session",
			req:  &GetArtifactVersionRequest{AppName: "app", UserID: "u", FileName: "user:safe.txt"},
		},
		{
			name: "save user name ignores an invalid session",
			req:  &SaveRequest{AppName: "app", UserID: "u", SessionID: "../../etc", FileName: "user:safe.txt", Part: textPart},
		},
		{
			name: "load user name ignores an invalid session",
			req:  &LoadRequest{AppName: "app", UserID: "u", SessionID: "victim/apps", FileName: "user:safe.txt"},
		},
		{
			name: "user name still path-checks the app name",
			req:  &DeleteRequest{AppName: "/leading/slash", UserID: "u", FileName: "user:safe.txt"},
			want: "app_name must not be an absolute path or start with a slash.",
		},
		{
			name: "user name does not list the session as missing",
			req:  &LoadRequest{UserID: "u", FileName: "user:safe.txt"},
			want: "invalid load request: missing required fields: AppName",
		},
		{
			name: "user name does not list the session as missing on save",
			req:  &SaveRequest{AppName: "app", FileName: "user:safe.txt", Part: textPart},
			want: "invalid save request: missing required fields: UserID",
		},
		{
			name: "session-scoped name still needs the session",
			req:  &VersionsRequest{AppName: "app", UserID: "u", FileName: "safe.txt"},
			want: "invalid versions request: missing required fields: SessionID",
		},
		// List path-checks the session ID only when it is set.
		{
			name: "list without session",
			req:  &ListRequest{AppName: "app", UserID: "u"},
		},
		{
			name: "list traversal in session ID",
			req:  &ListRequest{AppName: "app", UserID: "u", SessionID: "../s"},
			want: "session_id must not contain traversal segments.",
		},
		{
			name: "list reserved segment in app name",
			req:  &ListRequest{AppName: "victim/users", UserID: "u", SessionID: "s"},
			want: "app_name must not contain reserved path segments.",
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.req.Validate()
			if tc.want == "" {
				if err != nil {
					t.Fatalf("Validate() = %v, want nil", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("Validate() = nil, want %q", tc.want)
			}
			if !errors.Is(err, ErrInputValidation) {
				t.Errorf("errors.Is(%v, ErrInputValidation) = false, want true", err)
			}
			if err.Error() != tc.want {
				t.Errorf("Validate() error = %q, want %q", err.Error(), tc.want)
			}
		})
	}
}

// TestSaveRequestValidateAllowingFileData checks that the opt-in accepts the
// parts a file-data backend stores and refuses file data without a URI, as
// adk-python's GCS service does ("Artifact file_data must have a file_uri.",
// gcs_artifact_service.py).
func TestSaveRequestValidateAllowingFileData(t *testing.T) {
	newReq := func(part *genai.Part) *SaveRequest {
		return &SaveRequest{AppName: "app", UserID: "user1", SessionID: "sess1", FileName: "f.txt", Part: part}
	}
	accepted := []struct {
		name string
		req  *SaveRequest
	}{
		{name: "external file data", req: newReq(&genai.Part{FileData: &genai.FileData{FileURI: "gs://b/o", MIMEType: "application/pdf"}})},
		{name: "artifact reference", req: newReq(&genai.Part{FileData: &genai.FileData{FileURI: "artifact://apps/app/users/user1/sessions/sess1/artifacts/t.txt/versions/1"}})},
		{name: "empty text", req: newReq(genai.NewPartFromText(""))},
		{name: "inline data", req: newReq(genai.NewPartFromBytes([]byte("abc"), "image/png"))},
		{name: "inline data beside file data without a URI", req: newReq(&genai.Part{InlineData: &genai.Blob{Data: []byte("abc"), MIMEType: "image/png"}, FileData: &genai.FileData{}})},
		{name: "text beside file data without a URI", req: newReq(&genai.Part{Text: "x", FileData: &genai.FileData{}})},
	}
	for _, tc := range accepted {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.req.ValidateAllowingFileData(); err != nil {
				t.Errorf("ValidateAllowingFileData() = %v, want nil", err)
			}
		})
	}

	refused := []struct {
		name    string
		req     *SaveRequest
		wantMsg string
	}{
		{
			name:    "file data with an empty URI",
			req:     newReq(&genai.Part{FileData: &genai.FileData{MIMEType: "text/plain"}}),
			wantMsg: "invalid save request: Part.FileData.FileURI has to be set (Artifact file_data must have a file_uri.)",
		},
		{
			name:    "function call part",
			req:     newReq(genai.NewPartFromFunctionCall("f", nil)),
			wantMsg: "invalid save request: Part.InlineData or Part.Text has to be set",
		},
		{
			name: "missing app name",
			req: &SaveRequest{UserID: "user1", SessionID: "sess1", FileName: "f.txt", Part: &genai.Part{
				FileData: &genai.FileData{FileURI: "gs://b/o"},
			}},
			wantMsg: "invalid save request: missing required fields: AppName",
		},
	}
	for _, tc := range refused {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.req.ValidateAllowingFileData()
			if !errors.Is(err, ErrInputValidation) {
				t.Fatalf("ValidateAllowingFileData() = %v, want an error wrapping ErrInputValidation", err)
			}
			if err.Error() != tc.wantMsg {
				t.Errorf("ValidateAllowingFileData() error = %q, want %q", err.Error(), tc.wantMsg)
			}
		})
	}
}
