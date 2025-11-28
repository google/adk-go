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

package functiontool_test

import (
	"encoding/json"
	"fmt"
	"iter"
	"net/http"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/internal/httprr"
	"google.golang.org/adk/internal/testutil"
	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/internal/typeutil"
	"google.golang.org/adk/model"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

func ExampleNew() {
	type SumArgs struct {
		A int `json:"a"` // an integer to sum
		B int `json:"b"` // another integer to sum
	}
	type SumResult struct {
		Sum int `json:"sum"` // the sum of two integers
	}

	handler := func(ctx tool.Context, input SumArgs) (SumResult, error) {
		return SumResult{Sum: input.A + input.B}, nil
	}
	sumTool, err := functiontool.New(functiontool.Config{
		Name:        "sum",
		Description: "sums two integers",
	}, handler)
	if err != nil {
		panic(err)
	}
	_ = sumTool // use the tool
}

//go:generate go test -httprecord=.*

func TestFunctionTool_Simple(t *testing.T) {
	ctx := t.Context()
	// TODO: this model creation code was copied from model/genai_test.go. Refactor so both tests can share.
	modelName := "gemini-2.0-flash"
	replayTrace := filepath.Join("testdata", t.Name()+".httprr")
	cfg := newGeminiTestClientConfig(t, replayTrace)
	m, err := gemini.NewModel(ctx, modelName, cfg)
	if err != nil {
		t.Fatalf("model.NewGeminiModel(%q) failed: %v", modelName, err)
	}

	type Args struct {
		City string `json:"city"`
	}
	type Result struct {
		Report string `json:"report"`
		Status string `json:"status"`
	}
	resultSet := map[string]Result{
		"london": {
			Status: "success",
			Report: "The current weather in London is cloudy with a temperature of18 degrees Celsius and a chance of rain.",
		},
		"paris": {
			Status: "success",
			Report: "The weather in Paris is sunny with a temperature of 25 derees Celsius.",
		},
	}

	weatherReport := func(ctx tool.Context, input Args) (Result, error) {
		city := strings.ToLower(input.City)
		if ret, ok := resultSet[city]; ok {
			return ret, nil
		}
		return Result{}, fmt.Errorf("weather information for %q is not available", city)
	}

	weatherReportTool, err := functiontool.New(
		functiontool.Config{
			Name:        "get_weather_report",
			Description: "Retrieves the current weather report for a specified city.",
		},
		weatherReport)
	if err != nil {
		t.Fatalf("NewFunctionTool failed: %v", err)
	}

	for _, tc := range []struct {
		name    string
		prompt  string
		want    Result
		isError bool
	}{
		{
			name:    "london",
			prompt:  "Report the current weather of the capital city of U.K.",
			want:    resultSet["london"],
			isError: false,
		},
		{
			name:    "paris",
			prompt:  "How is the weather of Paris now?",
			want:    resultSet["paris"],
			isError: false,
		},
		{
			name:    "new york",
			prompt:  "Tell me about the current weather in New York",
			want:    Result{},
			isError: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// TODO: replace with testing using LLMAgent, instead of directly calling the model.
			var req model.LLMRequest
			requestProcessor, ok := weatherReportTool.(toolinternal.RequestProcessor)
			if !ok {
				t.Fatal("weatherReportTool does not implement itype.RequestProcessor")
			}
			if err := requestProcessor.ProcessRequest(nil, &req); err != nil {
				t.Fatalf("weatherReportTool.ProcessRequest failed: %v", err)
			}
			if req.Config == nil || len(req.Config.Tools) != 1 {
				t.Fatalf("weatherReportTool.ProcessRequest did not configure tool info in LLMRequest: %v", req)
			}
			req.Contents = genai.Text(tc.prompt)
			resp, err := readFirstResponse[*genai.FunctionCall](
				m.GenerateContent(ctx, &req, false),
			)
			if err != nil {
				t.Fatalf("GenerateContent(%v) failed: %v", req, err)
			}
			if resp.Name != "get_weather_report" || len(resp.Args) == 0 {
				t.Fatalf("unexpected function call %v", resp)
			}
			// Call the function.
			funcTool, ok := weatherReportTool.(toolinternal.FunctionTool)
			if !ok {
				t.Fatal("weatherReportTool does not implement itype.RequestProcessor")
			}
			callResult, err := funcTool.Run(nil, resp.Args)
			if tc.isError {
				if err == nil {
					t.Fatalf("weatherReportTool.Run(%v) expected to fail but got success with result %v", resp.Args, callResult)
				}
				return
			}
			if err != nil {
				t.Fatalf("weatherReportTool.Run failed: %v", err)
			}
			got, err := typeutil.ConvertToWithJSONSchema[map[string]any, Result](callResult, nil)
			if err != nil {
				t.Fatalf("weatherReportTool.Run returned unexpected result of type %[1]T: %[1]v", callResult)
			}
			want := tc.want
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("weatherReportTool.Run returned unexpected result (-want +got):\n%s", diff)
			}
		})
	}
}

func TestFunctionTool_DifferentFunctionDeclarations_ConsolidatedInOneGenAiTool(t *testing.T) {
	// First tool
	identityFunc := func(ctx tool.Context, x int) (int, error) {
		return x, nil
	}
	identityTool, err := functiontool.New(functiontool.Config{
		Name:        "identity",
		Description: "returns the input value",
	}, identityFunc)
	if err != nil {
		panic(err)
	}

	// Second tool
	stringIdentityFunc := func(ctx tool.Context, input string) (string, error) {
		return input, nil
	}
	stringIdentityTool, err := functiontool.New(
		functiontool.Config{
			Name:        "string_identity",
			Description: "returns the input value",
		},
		stringIdentityFunc)
	if err != nil {
		t.Fatalf("NewFunctionTool failed: %v", err)
	}

	var req model.LLMRequest
	requestProcessor, ok := identityTool.(toolinternal.RequestProcessor)
	if !ok {
		t.Fatal("identityTool does not implement itype.RequestProcessor")
	}
	if err := requestProcessor.ProcessRequest(nil, &req); err != nil {
		t.Fatalf("identityTool.ProcessRequest failed: %v", err)
	}
	requestProcessor, ok = stringIdentityTool.(toolinternal.RequestProcessor)
	if !ok {
		t.Fatal("stringIdentityTool does not implement itype.RequestProcessor")
	}
	if err := requestProcessor.ProcessRequest(nil, &req); err != nil {
		t.Fatalf("stringIdentityTool.ProcessRequest failed: %v", err)
	}

	if len(req.Config.Tools) != 1 {
		t.Errorf("number of tools should be one, got: %d", len(req.Config.Tools))
	}
	if len(req.Config.Tools[0].FunctionDeclarations) != 2 {
		t.Errorf("number of function declarations should be two, got: %d", len(req.Config.Tools[0].FunctionDeclarations))
	}
}

func TestFunctionTool_ReturnsBasicType(t *testing.T) {
	type Args struct {
		City string `json:"city"`
	}
	resultSet := map[string]string{
		"london": "The current weather in London is cloudy with a temperature of18 degrees Celsius and a chance of rain.",
		"paris":  "The weather in Paris is sunny with a temperature of 25 derees Celsius.",
	}

	weatherReport := func(ctx tool.Context, input Args) (string, error) {
		city := strings.ToLower(input.City)
		if ret, ok := resultSet[city]; ok {
			return ret, nil
		}
		return fmt.Sprintf("Weather information for %q is not available.", city), nil
	}

	weatherReportTool, err := functiontool.New(
		functiontool.Config{
			Name:        "get_weather_report",
			Description: "Retrieves the current weather report for a specified city.",
		},
		weatherReport)
	if err != nil {
		t.Fatalf("NewFunctionTool failed: %v", err)
	}

	for _, tc := range []struct {
		args   map[string]any
		name   string
		prompt string
		want   string
	}{
		{
			args:   map[string]any{"city": "london"},
			name:   "london",
			prompt: "Report the current weather of the capital city of U.K.",
			want:   resultSet["london"],
		},
		{
			args:   map[string]any{"city": "paris"},
			name:   "paris",
			prompt: "How is the weather of Paris now?",
			want:   resultSet["paris"],
		},
		{
			args:   map[string]any{"city": "new york"},
			name:   "new york",
			prompt: "Tell me about the current weather in New York",
			want:   `Weather information for "new york" is not available.`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// TODO: replace with testing using LLMAgent, instead of directly calling the model.
			var req model.LLMRequest
			requestProcessor, ok := weatherReportTool.(toolinternal.RequestProcessor)
			if !ok {
				t.Fatal("weatherReportTool does not implement itype.RequestProcessor")
			}
			if err := requestProcessor.ProcessRequest(nil, &req); err != nil {
				t.Fatalf("weatherReportTool.ProcessRequest failed: %v", err)
			}
			if req.Config == nil || len(req.Config.Tools) != 1 {
				t.Fatalf("weatherReportTool.ProcessRequest did not configure tool info in LLMRequest: %v", req)
			}
			// Call the function.
			funcTool, ok := weatherReportTool.(toolinternal.FunctionTool)
			if !ok {
				t.Fatal("weatherReportTool does not implement itype.RequestProcessor")
			}
			callResult, err := funcTool.Run(nil, tc.args)
			if err != nil {
				t.Fatalf("weatherReportTool.Run failed: %v", err)
			}
			got, err := typeutil.ConvertToWithJSONSchema[map[string]any, map[string]string](callResult, nil)
			if err != nil {
				t.Fatalf("weatherReportTool.Run returned unexpected result of type %[1]T: %[1]v", callResult)
			}
			gotVal, ok := got["result"]
			if !ok {
				t.Fatalf("function response, incorrect %q value", got["result"])
			}
			want := tc.want
			if diff := cmp.Diff(want, gotVal); diff != "" {
				t.Errorf("weatherReportTool.Run returned unexpected result (-want +got):\n%s", diff)
			}
		})
	}
}

// newGeminiTestClientConfig returns the genai.ClientConfig configured for record and replay.
func newGeminiTestClientConfig(t *testing.T, rrfile string) *genai.ClientConfig {
	t.Helper()
	rr, err := testutil.NewGeminiTransport(rrfile)
	if err != nil {
		t.Fatal(err)
	}
	apiKey := ""
	if recording, _ := httprr.Recording(rrfile); !recording {
		apiKey = "fakekey"
	}
	return &genai.ClientConfig{
		HTTPClient: &http.Client{Transport: rr},
		APIKey:     apiKey,
	}
}

func readFirstResponse[T any](s iter.Seq2[*model.LLMResponse, error]) (T, error) {
	var zero T
	do := func(s iter.Seq2[*model.LLMResponse, error]) (any, error) {
		for resp, err := range s {
			if err != nil {
				return zero, err
			}
			if resp.Content == nil || len(resp.Content.Parts) == 0 {
				return zero, fmt.Errorf("encountered an empty response: %v", resp)
			}
			for _, p := range resp.Content.Parts {
				switch any(zero).(type) {
				case string:
					if p.Text != "" {
						return p.Text, nil
					}
				case *genai.FunctionCall:
					if p.FunctionCall != nil {
						return p.FunctionCall, nil
					}
				case *genai.FunctionResponse:
					if p.FunctionResponse != nil {
						return p.FunctionResponse, nil
					}
				}
			}
			return zero, fmt.Errorf("response does not contain data for %T: %v", zero, resp)
		}
		return zero, fmt.Errorf("no response message was received")
	}
	v, err := do(s)
	if err != nil {
		return zero, err
	}
	if v, ok := v.(T); ok {
		return v, nil
	}
	panic(fmt.Sprintf("do extracted unexpected type = %[1]T(%[1]v), want %T", v, zero))
}

func TestFunctionTool_CustomSchema(t *testing.T) {
	type Args struct {
		// Either apple or orange, nothing else.
		Fruit string `json:"fruit"`
	}
	ischema, err := jsonschema.For[Args](nil)
	if err != nil {
		t.Fatalf("jsonschema.For[Args]() failed: %v", err)
	}
	fruit, ok := ischema.Properties["fruit"]
	if !ok {
		t.Fatalf("unexpeced jsonschema: missing 'fruit': %+v", ischema)
	}
	fruit.Description = "print the remaining quantity of the item."
	fruit.Enum = []any{"mandarin", "kiwi"}

	inventoryTool, err := functiontool.New(functiontool.Config{
		Name:        "print_quantity",
		Description: "print the remaining quantity of the given fruit.",
		InputSchema: ischema,
	}, func(ctx tool.Context, input Args) (any, error) {
		fruit := strings.ToLower(input.Fruit)
		if fruit != "mandarin" && fruit != "kiwi" {
			t.Errorf("unexpected fruit: %q", fruit)
		}
		return nil, nil // always return nil.
	})
	if err != nil {
		t.Fatalf("NewFunctionTool failed: %v", err)
	}

	t.Run("ProcessRequest", func(t *testing.T) {
		var req model.LLMRequest
		requestProcessor, ok := inventoryTool.(toolinternal.RequestProcessor)
		if !ok {
			t.Fatal("inventoryTool does not implement itype.RequestProcessor")
		}
		if err := requestProcessor.ProcessRequest(nil, &req); err != nil {
			t.Fatalf("inventoryTool.ProcessRequest failed: %v", err)
		}
		decl := toolDeclaration(req.Config)
		if decl == nil {
			t.Fatalf("inventoryTool.ProcessRequest did not configure function declaration: %v", req)
			// to prevent SA5011: possible nil pointer dereference (staticcheck)
			return
		}
		if got, want := decl.Name, inventoryTool.Name(); got != want {
			t.Errorf("inventoryTool function declaration name = %q, want %q", got, want)
		}
		if got, want := decl.Description, inventoryTool.Description(); got != want {
			t.Errorf("inventoryTool function declaration description = %q, want %q", got, want)
		}
		if got, want := stringify(decl.ParametersJsonSchema), stringify(ischema); got != want {
			t.Errorf("inventoryTool function declaration parameter json schema = %q, want %q", got, want)
		}
		if got, want := stringify(decl.ResponseJsonSchema), stringify(&jsonschema.Schema{}); got != want {
			t.Errorf("inventoryTool function response json schema = %q, want %q", got, want)
		}
	})

	t.Run("Run", func(t *testing.T) {
		testCases := []struct {
			name    string
			in      map[string]any
			wantErr bool
		}{
			{
				name:    "valid_item",
				in:      map[string]any{"fruit": "mandarin"},
				wantErr: false,
			},
			{
				name:    "invalid_item",
				in:      map[string]any{"fruit": "banana"},
				wantErr: true,
			},
			{
				name:    "unexpected_type",
				in:      map[string]any{"fruit": 1},
				wantErr: true,
			},
			{
				name:    "nil",
				in:      nil,
				wantErr: true,
			},
		}
		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				funcTool, ok := inventoryTool.(toolinternal.FunctionTool)
				if !ok {
					t.Fatal("inventoryTool does not implement itype.RequestProcessor")
				}
				ret, err := funcTool.Run(nil, tc.in)
				// ret is expected to be nil always.
				if tc.wantErr && err == nil {
					t.Errorf("inventoryTool.Run = (%v, %v), want error", ret, err)
				}
				if !tc.wantErr && (err != nil || ret != nil) {
					// TODO: fix, for "valid_item" case now it returns empty map instead of nil
					if len(ret) != 0 {
						t.Errorf("inventoryTool.Run = (%v, %v), want (nil, nil)", ret, err)
					}
				}
			})
		}
	})
}

func toolDeclaration(cfg *genai.GenerateContentConfig) *genai.FunctionDeclaration {
	if cfg == nil || len(cfg.Tools) == 0 {
		return nil
	}
	t := cfg.Tools[0]
	if len(t.FunctionDeclarations) == 0 {
		return nil
	}
	return t.FunctionDeclarations[0]
}

func stringify(v any) string {
	x, err := json.MarshalIndent(v, "", " ")
	if err != nil {
		panic(err)
	}
	return string(x)
}
