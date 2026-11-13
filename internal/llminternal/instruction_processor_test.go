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

package llminternal

import (
	"context"
	"errors"
	"io/fs"
	"iter"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
	artifactinternal "google.golang.org/adk/v2/internal/artifact"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

func TestInjectSessionState(t *testing.T) {
	// Define the structure for our test cases
	testCases := []struct {
		name                  string                 // Name of the sub-test
		template              string                 // Input template string
		state                 map[string]any         // Initial session state
		artifacts             map[string]*genai.Part // Artifacts for the mock service
		customArtifactService artifact.Service       // Optional custom artifact service (overrides default)
		expectNilService      bool                   // Flag to test with a nil artifact service
		want                  string                 // Expected successful output
		wantErr               bool                   // Whether we expect an error
		wantErrMsg            string                 // A substring of the expected error message
	}{
		// Corresponds to: test_inject_session_state
		{
			name:     "successful state injection",
			template: "Hello {user_name}, you are in {app_state} state.",
			state:    map[string]any{"user_name": "Foo", "app_state": "active"},
			want:     "Hello Foo, you are in active state.",
		},
		// Corresponds to: test_inject_session_state_with_artifact
		{
			name:     "successful artifact injection",
			template: "The artifact content is: {artifact.my_file}",
			artifacts: map[string]*genai.Part{
				"my_file": {Text: "This is my artifact content."},
			},
			want: "The artifact content is: This is my artifact content.",
		},
		// Corresponds to: test_inject_session_state_with_optional_state
		// and test_inject_session_state_with_optional_missing_state_returns_empty
		{
			name:     "optional missing state variable",
			template: "Optional value: {optional_value?}",
			state:    map[string]any{},
			want:     "Optional value: ",
		},
		// Corresponds to: test_inject_session_state_with_missing_state_raises_key_error
		{
			name:       "missing required state variable",
			template:   "Hello {missing_key}!",
			state:      map[string]any{"user_name": "Foo"},
			wantErr:    true,
			wantErrMsg: "state key does not exist",
		},
		// Corresponds to: test_inject_session_state_preserves_literal_dollar_brace_patterns
		{
			name:     "dollar-brace pattern is not replaced",
			template: "The formatString supports interpolation via ${expression} syntax.",
			state:    map[string]any{},
			want:     "The formatString supports interpolation via ${expression} syntax.",
		},
		// Corresponds to: test_inject_session_state_with_dollar_brace_and_session_state
		{
			name:     "dollar-brace pattern next to state variable",
			template: "Hello {user_name}! Interpolation via ${expression} syntax.",
			state:    map[string]any{"user_name": "Foo"},
			want:     "Hello Foo! Interpolation via ${expression} syntax.",
		},
		// Corresponds to: test_inject_session_state_preserves_escaped_braces
		{
			name:     "escaped braces are not replaced",
			template: `Literal \{expression\} syntax.`,
			state:    map[string]any{},
			want:     `Literal \{expression\} syntax.`,
		},
		{
			name:     "escaped opening brace is not replaced",
			template: `Literal \{expression} syntax.`,
			state:    map[string]any{"expression": "foo"},
			want:     `Literal \{expression} syntax.`,
		},
		// Corresponds to: test_inject_session_state_preserves_dollar_double_brace_patterns
		{
			name:     "dollar-double-brace pattern is not replaced",
			template: "Workflow syntax: ${{expression}} and {user_name}.",
			state:    map[string]any{"expression": "foo", "user_name": "bar"},
			want:     "Workflow syntax: ${{expression}} and bar.",
		},
		{
			name:     "placeholder at start of template",
			template: "{user_name}, welcome",
			state:    map[string]any{"user_name": "Foo"},
			want:     "Foo, welcome",
		},
		{
			name:     "dollar-brace pattern at start of template",
			template: "${user_name}, welcome",
			state:    map[string]any{"user_name": "Foo"},
			want:     "${user_name}, welcome",
		},
		{
			name:     "double-brace placeholder is replaced",
			template: "Say {{user_name}}",
			state:    map[string]any{"user_name": "Foo"},
			want:     "Say Foo",
		},
		// A text artifact can come back as inline data rather than Text:
		// gcsartifact stores a text Part as a text/plain object and loads it
		// back as bytes. adk-python injects the decoded text
		// (_instructions_utils.py, _artifact_to_text).
		{
			name:     "text artifact loaded as inline data",
			template: "The artifact content is: {artifact.notes.txt}",
			artifacts: map[string]*genai.Part{
				"notes.txt": genai.NewPartFromBytes([]byte("hello from GCS"), "text/plain"),
			},
			want: "The artifact content is: hello from GCS",
		},
		{
			name:     "text artifact with a charset parameter",
			template: "The artifact content is: {artifact.notes}",
			artifacts: map[string]*genai.Part{
				"notes": genai.NewPartFromBytes([]byte("café"), "Text/Markdown; charset=utf-8"),
			},
			want: "The artifact content is: café",
		},
		{
			name:     "json artifact with a charset parameter",
			template: "Config: {artifact.config}",
			artifacts: map[string]*genai.Part{
				"config": genai.NewPartFromBytes([]byte(`{"k":"v"}`), "application/JSON; charset=utf-8"),
			},
			want: `Config: {"k":"v"}`,
		},
		{
			name:     "text file uploaded as octet-stream",
			template: "Rows: {artifact.rows.csv}",
			artifacts: map[string]*genai.Part{
				"rows.csv": genai.NewPartFromBytes([]byte("a,b\n1,2"), "application/octet-stream"),
			},
			want: "Rows: a,b\n1,2",
		},
		{
			name:     "binary artifact still renders empty",
			template: "Image: {artifact.photo.png}",
			artifacts: map[string]*genai.Part{
				"photo.png": genai.NewPartFromBytes([]byte{0x89, 'P', 'N', 'G'}, "image/png"),
			},
			want: "Image: ",
		},
		// A binary artifact with a text file name is not decoded: Gemini
		// accepts its MIME type inline. TestArtifactText covers each rule.
		{
			name:     "pdf named like a text file is not decoded",
			template: "Doc: {artifact.notes.txt}",
			artifacts: map[string]*genai.Part{
				"notes.txt": genai.NewPartFromBytes([]byte("%PDF-1.7\n%\xe2\xe3\xcf\xd3"), "application/pdf"),
			},
			want: "Doc: ",
		},
		// Corresponds to: test_inject_session_state_with_missing_artifact_raises_key_error
		{
			name:     "missing required artifact",
			template: "The artifact content is: {artifact.missing_file}",
			artifacts: map[string]*genai.Part{
				"my_file": {Text: "This is my artifact content."},
			},
			wantErr:    true,
			wantErrMsg: "failed to load artifact missing_file: artifact not found: file does not exist",
		},
		// Corresponds to: test_inject_session_state_with_invalid_state_name_returns_original
		{
			name:     "invalid state name is not replaced",
			template: "Hello {invalid-key}!",
			state:    map[string]any{"user_name": "Foo"},
			want:     "Hello {invalid-key}!",
		},
		// Corresponds to: test_inject_session_state_with_invalid_prefix_state_name_returns_original
		{
			name:     "invalid prefix state name is not replaced",
			template: "Hello {invalid:key}!",
			state:    map[string]any{"user_name": "Foo"},
			want:     "Hello {invalid:key}!",
		},
		// Corresponds to: test_inject_session_state_with_valid_prefix_state
		{
			name:     "valid prefixed state variable",
			template: "Hello {app:user_name}!",
			state:    map[string]any{"app:user_name": "Foo"},
			want:     "Hello Foo!",
		},
		// Corresponds to: test_inject_session_state_with_none_state_value_returns_empty
		{
			name:     "state value is nil",
			template: "Value: {test_key}",
			state:    map[string]any{"test_key": nil},
			want:     "Value: ",
		},
		// Corresponds to: test_inject_session_state_with_optional_missing_artifact_returns_empty
		{
			name:     "optional missing artifact",
			template: "Optional artifact: {artifact.missing_file?}",
			artifacts: map[string]*genai.Part{
				"my_file": {Text: "This is my artifact content."},
			},
			want: "Optional artifact: ",
		},
		// Systemic artifact error on an optional placeholder must propagate, not silently render empty.
		{
			name:                  "optional artifact with systemic backend error propagates",
			template:              "Content: {artifact.my_file?}",
			customArtifactService: &errArtifactService{err: errArtifactBackendDown},
			wantErr:               true,
			wantErrMsg:            "backend down",
		},
		// fs.ErrNotExist returned by the service still renders empty for optional placeholders.
		{
			name:                  "optional artifact with fs.ErrNotExist renders empty",
			template:              "Content: {artifact.my_file?}",
			customArtifactService: &errArtifactService{err: fs.ErrNotExist},
			want:                  "Content: ",
		},
		// Corresponds to: test_inject_session_state_artifact_service_not_initialized_raises_value_error
		{
			name:             "artifact service not initialized",
			template:         "The artifact content is: {artifact.my_file}",
			expectNilService: true,
			wantErr:          true,
			wantErrMsg:       "artifact service is not initialized",
		},
		// Corresponds to: test_inject_session_state_with_empty_artifact_name_raises_key_error
		{
			name:       "empty artifact name",
			template:   "The artifact content is: {artifact.}",
			artifacts:  map[string]*genai.Part{},
			wantErr:    true,
			wantErrMsg: "failed to load artifact : request validation failed: invalid load request: missing required fields: FileName",
		},
		// Corresponds to: test_inject_session_state_with_multiple_variables_and_artifacts
		{
			name: "complex template with mixed variables and artifacts",
			template: `
Hello {user_name},
You are {user_age} years old.
Your favorite color is {favorite_color}.
The artifact says: {artifact.my_file}
And another optional artifact: {artifact.other_file?}
`,
			state: map[string]any{
				"user_name":      "Foo",
				"user_age":       30,
				"favorite_color": "blue",
			},
			artifacts: map[string]*genai.Part{
				"my_file": {Text: "This is my artifact content."},
			},
			want: `
Hello Foo,
You are 30 years old.
Your favorite color is blue.
The artifact says: This is my artifact content.
And another optional artifact: 
`,
		},
	}

	// Iterate over the test cases
	for _, tc := range testCases {
		// t.Run creates a sub-test, which makes test output cleaner and more organized.
		t.Run(tc.name, func(t *testing.T) {
			// Setup, create inMemorySessionService, inMemoryArtifactService and wrappers.
			sessionService := session.InMemoryService()
			createResp, err := sessionService.Create(t.Context(), &session.CreateRequest{
				AppName:   "testApp",
				UserID:    "testUser",
				SessionID: "testSession",
				State:     tc.state,
			})
			if err != nil {
				t.Fatalf("Failed to create session: %v", err)
			}

			// Setup Artifacts
			var artifacts agent.Artifacts
			if !tc.expectNilService {
				svc := artifact.Service(artifact.InMemoryService())
				if tc.customArtifactService != nil {
					svc = tc.customArtifactService
				}
				artifacts = &artifactinternal.Artifacts{
					Service:   svc,
					AppName:   "testApp",
					UserID:    "testUser",
					SessionID: "testSession",
				}
			}
			for filename, part := range tc.artifacts {
				if _, err := artifacts.Save(t.Context(), filename, part); err != nil {
					t.Fatalf("Failed to save artifact: %v", err)
				}
			}
			// Create invocation context
			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
				Artifacts: artifacts,
				Session:   createResp.Session,
			})

			// --- Execution ---
			got, err := InjectSessionState(ctx, tc.template)

			// --- Assertion ---
			if tc.wantErr {
				if err == nil {
					t.Fatal("expected an error but got none")
				}
				if tc.wantErrMsg != "" && !strings.Contains(err.Error(), tc.wantErrMsg) {
					t.Errorf("expected error message to contain %q, but got %q", tc.wantErrMsg, err.Error())
				}
			} else {
				if err != nil {
					t.Fatalf("did not expect an error but got: %v", err)
				}
				if got != tc.want {
					// Use %q to clearly show differences in strings, especially with whitespace.
					t.Errorf("got %q, want %q", got, tc.want)
				}
			}
		})
	}
}

// TestArtifactText pins each rule artifactText applies to inline data, in
// adk-python's order: MIME types Gemini accepts inline are never decoded,
// then text MIME types and text file extensions are.
func TestArtifactText(t *testing.T) {
	t.Parallel()
	testCases := []struct {
		name, fileName, mimeType, data, want string
	}{
		// Gemini accepts these inline, so they are not decoded, whatever the name.
		{name: "pdf", fileName: "notes.txt", mimeType: "application/pdf", data: "%PDF-1.7\n%\xe2\xe3\xcf\xd3", want: ""},
		{name: "image", fileName: "photo.xml", mimeType: "image/png", data: "\x89PNG", want: ""},
		{name: "audio", fileName: "memo.csv", mimeType: "audio/wav", data: "RIFF\x24\x00\x00\x00WAVE", want: ""},
		{name: "video", fileName: "clip.json", mimeType: "video/mp4", data: "\x00\x00\x00\x18ftyp", want: ""},
		// Gemini rejects SVG and XML images inline, so adk-python decodes them.
		{name: "svg image", fileName: "logo", mimeType: "image/svg", data: "<svg/>", want: "<svg/>"},
		{name: "svg+xml image", fileName: "logo", mimeType: "image/svg+xml", data: "<svg/>", want: "<svg/>"},
		{name: "xml image", fileName: "logo", mimeType: "image/xml", data: "<svg/>", want: "<svg/>"},
		{name: "csv", fileName: "rows", mimeType: "application/csv", data: "a,b", want: "a,b"},
		{name: "svg+xml application", fileName: "logo", mimeType: "application/svg+xml", data: "<svg/>", want: "<svg/>"},
		{name: "xml application", fileName: "feed", mimeType: "application/xml", data: "<feed/>", want: "<feed/>"},
		{name: "space before the parameters", fileName: "config", mimeType: "application/json ; charset=utf-8", data: `{"k":"v"}`, want: `{"k":"v"}`},
		{name: "txt extension", fileName: "readme.txt", mimeType: "application/octet-stream", data: "read me", want: "read me"},
		{name: "upper-case json extension", fileName: "config.JSON", mimeType: "application/octet-stream", data: `{"k":"v"}`, want: `{"k":"v"}`},
		{name: "xml extension", fileName: "layout.xml", mimeType: "application/octet-stream", data: "<a/>", want: "<a/>"},
		{name: "other binary", fileName: "blob.bin", mimeType: "application/octet-stream", data: "\x00\x01", want: ""},
		// One U+FFFD per run of invalid bytes. Python's errors='replace' gives
		// "caf\uFFFD \uFFFD\uFFFD" here.
		{name: "invalid UTF-8", fileName: "notes", mimeType: "text/plain", data: "caf\xe9 \xe2\xe3", want: "caf\uFFFD \uFFFD"},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := artifactText(tc.fileName, genai.NewPartFromBytes([]byte(tc.data), tc.mimeType))
			if got != tc.want {
				t.Errorf("artifactText(%q, %q) = %q, want %q", tc.fileName, tc.mimeType, got, tc.want)
			}
		})
	}
}

var errDatabaseFailure = errors.New("database connection refused")

// errArtifactBackendDown is a sentinel for a systemic artifact service failure.
var errArtifactBackendDown = errors.New("backend down")

// errArtifactService is a mock artifact.Service that always returns a fixed error from Load.
type errArtifactService struct {
	err error
}

func (s *errArtifactService) Save(_ context.Context, _ *artifact.SaveRequest) (*artifact.SaveResponse, error) {
	return nil, errors.ErrUnsupported
}

func (s *errArtifactService) Load(_ context.Context, _ *artifact.LoadRequest) (*artifact.LoadResponse, error) {
	return nil, s.err
}

func (s *errArtifactService) Delete(_ context.Context, _ *artifact.DeleteRequest) error {
	return errors.ErrUnsupported
}

func (s *errArtifactService) List(_ context.Context, _ *artifact.ListRequest) (*artifact.ListResponse, error) {
	return nil, errors.ErrUnsupported
}

func (s *errArtifactService) Versions(_ context.Context, _ *artifact.VersionsRequest) (*artifact.VersionsResponse, error) {
	return nil, errors.ErrUnsupported
}

func (s *errArtifactService) GetArtifactVersion(_ context.Context, _ *artifact.GetArtifactVersionRequest) (*artifact.GetArtifactVersionResponse, error) {
	return nil, errors.ErrUnsupported
}

type errorMockState struct{}

func (s *errorMockState) Get(key string) (any, error) {
	if key == "systemic_error" {
		return nil, errDatabaseFailure
	}
	return nil, session.ErrStateKeyNotExist
}

func (s *errorMockState) Set(_ string, _ any) error {
	return nil
}

func (s *errorMockState) All() iter.Seq2[string, any] {
	return nil
}

type errorMockSession struct{}

func (s *errorMockSession) ID() string {
	return "testSession"
}

func (s *errorMockSession) AppName() string {
	return "testApp"
}

func (s *errorMockSession) UserID() string {
	return "testUser"
}

func (s *errorMockSession) State() session.State {
	return &errorMockState{}
}

func (s *errorMockSession) Events() session.Events {
	return nil
}

func (s *errorMockSession) LastUpdateTime() time.Time {
	return time.Now()
}

func TestProcessStateVariable_SystemicErrorPropagates(t *testing.T) {
	ctx := icontext.NewInvocationContext(context.Background(), icontext.InvocationContextParams{
		Session: &errorMockSession{},
	})

	t.Run("missing optional key", func(t *testing.T) {
		val, err := replaceMatch(ctx, "{missing_key?}")
		if err != nil {
			t.Fatalf("expected no error for missing optional key, got: %v", err)
		}
		if val != "" {
			t.Errorf("expected empty string, got: %q", val)
		}
	})

	t.Run("missing required key", func(t *testing.T) {
		_, err := replaceMatch(ctx, "{missing_key}")
		if err == nil || !errors.Is(err, session.ErrStateKeyNotExist) {
			t.Fatalf("expected ErrStateKeyNotExist, got: %v", err)
		}
	})

	t.Run("systemic error on optional key", func(t *testing.T) {
		_, err := replaceMatch(ctx, "{systemic_error?}")
		if err == nil || !errors.Is(err, errDatabaseFailure) {
			t.Fatalf("expected systemic error to propagate even for optional key, got: %v", err)
		}
	})
}

func TestInstructionsRequestProcessor_LiveInjectionAppendsSuffix(t *testing.T) {
	const suffix = "\ncall the task_completed function to exit"
	errProvider := errors.New("provider failed")
	for _, tt := range []struct {
		name    string
		state   *State
		suffix  string
		want    string
		wantErr error
	}{
		{
			name:   "instruction",
			state:  &State{Instruction: "base instruction"},
			suffix: suffix,
			want:   "base instruction" + suffix,
		},
		{
			name: "instruction provider",
			state: &State{InstructionProvider: func(agent.ReadonlyContext) (string, error) {
				return "dynamic instruction", nil
			}},
			suffix: suffix,
			want:   "dynamic instruction" + suffix,
		},
		{
			name: "instruction provider error",
			state: &State{InstructionProvider: func(agent.ReadonlyContext) (string, error) {
				return "partial", errProvider
			}},
			suffix:  suffix,
			wantErr: errProvider,
		},
		{
			name:  "empty suffix",
			state: &State{Instruction: "base instruction"},
			want:  "base instruction",
		},
		{
			name:   "suffix without a base instruction",
			state:  &State{},
			suffix: suffix,
			want:   suffix,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			mockAgent := &mockLLMAgent{
				Agent: utils.Must(agent.New(agent.Config{Name: "test_agent"})),
				s:     tt.state,
			}
			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
				Agent: mockAgent,
			})
			f := &Flow{LiveInjection: &LiveInjection{InstructionSuffix: tt.suffix}}
			req := &model.LLMRequest{}

			var gotErr error
			for _, err := range instructionsRequestProcessor(ctx, req, f) {
				if err != nil {
					gotErr = err
				}
			}
			if !errors.Is(gotErr, tt.wantErr) {
				t.Fatalf("instructionsRequestProcessor() error = %v, want %v", gotErr, tt.wantErr)
			}
			if tt.wantErr != nil {
				return
			}

			if got := systemInstructionText(req); got != tt.want {
				t.Errorf("instruction = %q, want %q", got, tt.want)
			}
		})
	}
}

func systemInstructionText(req *model.LLMRequest) string {
	if req.Config == nil {
		return ""
	}
	return strings.Join(utils.TextParts(req.Config.SystemInstruction), "")
}
