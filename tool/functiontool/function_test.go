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
	"errors"
	"fmt"
	"iter"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/agent/llmagent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/testutil"
	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/internal/typeutil"
	"google.golang.org/adk/model"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
	"google.golang.org/adk/tool/toolconfirmation"
)

type SumArgs struct {
	A int `json:"a"` // an integer to sum
	B int `json:"b"` // another integer to sum
}
type SumResult struct {
	Sum int `json:"sum"` // the sum of two integers
}

func sumFunc(ctx tool.Context, input SumArgs) (SumResult, error) {
	return SumResult{Sum: input.A + input.B}, nil
}

func ExampleNew() {
	sumTool, err := functiontool.New(functiontool.Config{
		Name:        "sum",
		Description: "sums two integers",
	}, sumFunc)
	if err != nil {
		panic(err)
	}
	_ = sumTool // use the tool
}

func createToolContext(t *testing.T) tool.Context {
	invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})
	return toolinternal.NewToolContext(invCtx, "", &session.EventActions{}, nil)
}

//go:generate go test -httprecord=.*

func TestFunctionTool_Simple(t *testing.T) {
	ctx := t.Context()
	// TODO: this model creation code was copied from model/genai_test.go. Refactor so both tests can share.
	modelName := "gemini-2.5-flash"
	replayTrace := filepath.Join("testdata", t.Name()+".httprr")
	cfg := testutil.NewGeminiTestClientConfig(t, replayTrace)
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
			callResult, err := funcTool.Run(createToolContext(t), resp.Args)
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
	type IntInput struct {
		X int `json:"x"`
	}
	type IntOutput struct {
		Result int `json:"result"`
	}
	identityFunc := func(ctx tool.Context, input IntInput) (IntOutput, error) {
		return IntOutput{Result: input.X}, nil
	}
	identityTool, err := functiontool.New(functiontool.Config{
		Name:        "identity",
		Description: "returns the input value",
	}, identityFunc)
	if err != nil {
		panic(err)
	}

	// Second tool
	type StringInput struct {
		Value string `json:"value"`
	}
	type StringOutput struct {
		Result string `json:"result"`
	}
	stringIdentityFunc := func(ctx tool.Context, input StringInput) (StringOutput, error) {
		return StringOutput{Result: input.Value}, nil
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
			callResult, err := funcTool.Run(createToolContext(t), tc.args)
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

func TestFunctionTool_MapInput(t *testing.T) {
	type Output struct {
		Sum int `json:"sum"`
	}
	sumTool, err := functiontool.New(
		functiontool.Config{
			Name:        "sum_map",
			Description: "sums numbers provided in a map input",
		},
		func(ctx tool.Context, input map[string]int) (Output, error) {
			return Output{Sum: input["a"] + input["b"]}, nil
		})
	if err != nil {
		t.Fatalf("NewFunctionTool failed: %v", err)
	}

	funcTool, ok := sumTool.(toolinternal.FunctionTool)
	if !ok {
		t.Fatal("sumTool does not implement itype.RequestProcessor")
	}
	callResult, err := funcTool.Run(createToolContext(t), map[string]any{"a": 2, "b": 3})
	if err != nil {
		t.Fatalf("sumTool.Run failed: %v", err)
	}
	got, err := typeutil.ConvertToWithJSONSchema[map[string]any, Output](callResult, nil)
	if err != nil {
		t.Fatalf("sumTool.Run returned unexpected result of type %[1]T: %[1]v", callResult)
	}
	want := Output{Sum: 5}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("sumTool.Run returned unexpected result (-want +got):\n%s", diff)
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
				ret, err := funcTool.Run(createToolContext(t), tc.in)
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

type SimpleArgs struct {
	Num int
}

func okFunc(_ tool.Context, _ SimpleArgs) (string, error) {
	return "ok", nil
}

func TestToolConfirmation(t *testing.T) {
	testCases := []struct {
		name                    string
		toolConfig              functiontool.Config
		args                    map[string]any
		confirmFunctionResponse *genai.FunctionResponse // User's confirmation response
		want                    []*genai.Content
	}{
		{
			name: "No Confirmation Required",
			toolConfig: functiontool.Config{
				Name: "test_tool",
			},
			args: map[string]any{"Num": 1},
			want: []*genai.Content{
				genai.NewContentFromFunctionCall("test_tool", map[string]any{"Num": 1}, "model"),
				genai.NewContentFromFunctionResponse("test_tool", map[string]any{"result": "ok"}, "user"),
			},
		},
		{
			name: "Confirmation Required",
			toolConfig: functiontool.Config{
				Name:                "test_tool",
				RequireConfirmation: true,
			},
			args: map[string]any{"Num": 1},
			want: []*genai.Content{
				genai.NewContentFromFunctionCall("test_tool", map[string]any{"Num": 1}, "model"),
				genai.NewContentFromFunctionCall(toolconfirmation.FunctionCallName, map[string]any{
					"originalFunctionCall": &genai.FunctionCall{
						Args: map[string]any{"Num": 1},
						Name: "test_tool",
					},
					"toolConfirmation": toolconfirmation.ToolConfirmation{
						Hint: "Please approve or reject the tool call test_tool() by responding with a FunctionResponse with an expected ToolConfirmation payload.",
					},
				}, "model"),
				genai.NewContentFromFunctionResponse("test_tool", map[string]any{
					"error": errors.New("error tool \"test_tool\" requires confirmation, please approve or reject"),
				}, "user"),
			},
		},
		{
			name: "Confirmation Required and is confirmed",
			toolConfig: functiontool.Config{
				Name:                "test_tool",
				RequireConfirmation: true,
			},
			args:                    map[string]any{"Num": 1},
			confirmFunctionResponse: &genai.FunctionResponse{Name: toolconfirmation.FunctionCallName, Response: map[string]any{"confirmed": true}},
			want: []*genai.Content{
				genai.NewContentFromFunctionCall("test_tool", map[string]any{"Num": 1}, "model"),
				genai.NewContentFromFunctionCall(toolconfirmation.FunctionCallName, map[string]any{
					"originalFunctionCall": &genai.FunctionCall{
						Args: map[string]any{"Num": 1},
						Name: "test_tool",
					},
					"toolConfirmation": toolconfirmation.ToolConfirmation{
						Hint: "Please approve or reject the tool call test_tool() by responding with a FunctionResponse with an expected ToolConfirmation payload.",
					},
				}, "model"),
				genai.NewContentFromFunctionResponse("test_tool", map[string]any{
					"error": errors.New("error tool \"test_tool\" requires confirmation, please approve or reject"),
				}, "user"),
				genai.NewContentFromFunctionResponse("test_tool", map[string]any{"result": "ok"}, "user"),
			},
		},
		{
			name: "Confirmation Required and is rejected",
			toolConfig: functiontool.Config{
				Name:                "test_tool",
				RequireConfirmation: true,
			},
			args:                    map[string]any{"Num": 1},
			confirmFunctionResponse: &genai.FunctionResponse{Name: toolconfirmation.FunctionCallName, Response: map[string]any{"confirmed": false}},
			want: []*genai.Content{
				genai.NewContentFromFunctionCall("test_tool", map[string]any{"Num": 1}, "model"),
				genai.NewContentFromFunctionCall(toolconfirmation.FunctionCallName, map[string]any{
					"originalFunctionCall": &genai.FunctionCall{
						Args: map[string]any{"Num": 1},
						Name: "test_tool",
					},
					"toolConfirmation": toolconfirmation.ToolConfirmation{
						Hint: "Please approve or reject the tool call test_tool() by responding with a FunctionResponse with an expected ToolConfirmation payload.",
					},
				}, "model"),
				genai.NewContentFromFunctionResponse("test_tool", map[string]any{
					"error": errors.New("error tool \"test_tool\" requires confirmation, please approve or reject"),
				}, "user"),
				genai.NewContentFromFunctionResponse("test_tool", map[string]any{
					"error": errors.New("error tool \"test_tool\" call is rejected"),
				}, "user"),
			},
		},
		{
			name: "Conditional Confirmation Not Required",
			toolConfig: functiontool.Config{
				Name: "test_tool",
				RequireConfirmationProvider: func(args SimpleArgs) bool {
					return args.Num < 5
				},
			},
			args: map[string]any{"Num": 7},
			want: []*genai.Content{
				genai.NewContentFromFunctionCall("test_tool", map[string]any{"Num": 7}, "model"),
				genai.NewContentFromFunctionResponse("test_tool", map[string]any{"result": "ok"}, "user"),
			},
		},
		{
			name: "Conditional Confirmation Required",
			toolConfig: functiontool.Config{
				Name: "test_tool",
				RequireConfirmationProvider: func(args SimpleArgs) bool {
					return args.Num < 5
				},
			},
			args: map[string]any{"Num": 4},
			want: []*genai.Content{
				genai.NewContentFromFunctionCall("test_tool", map[string]any{"Num": 4}, "model"),
				genai.NewContentFromFunctionCall(toolconfirmation.FunctionCallName, map[string]any{
					"originalFunctionCall": &genai.FunctionCall{
						Args: map[string]any{"Num": 4},
						Name: "test_tool",
					},
					"toolConfirmation": toolconfirmation.ToolConfirmation{
						Hint: "Please approve or reject the tool call test_tool() by responding with a FunctionResponse with an expected ToolConfirmation payload.",
					},
				}, "model"),
				genai.NewContentFromFunctionResponse("test_tool", map[string]any{
					"error": errors.New("error tool \"test_tool\" requires confirmation, please approve or reject"),
				}, "user"),
			},
		},
		{
			name: "Conditional Confirmation Required and is confirmed",
			toolConfig: functiontool.Config{
				Name: "test_tool",
				RequireConfirmationProvider: func(args SimpleArgs) bool {
					return args.Num < 5
				},
			},
			args:                    map[string]any{"Num": 4},
			confirmFunctionResponse: &genai.FunctionResponse{Name: toolconfirmation.FunctionCallName, Response: map[string]any{"confirmed": true}},
			want: []*genai.Content{
				genai.NewContentFromFunctionCall("test_tool", map[string]any{"Num": 4}, "model"),
				genai.NewContentFromFunctionCall(toolconfirmation.FunctionCallName, map[string]any{
					"originalFunctionCall": &genai.FunctionCall{
						Args: map[string]any{"Num": 4},
						Name: "test_tool",
					},
					"toolConfirmation": toolconfirmation.ToolConfirmation{
						Hint: "Please approve or reject the tool call test_tool() by responding with a FunctionResponse with an expected ToolConfirmation payload.",
					},
				}, "model"),
				genai.NewContentFromFunctionResponse("test_tool", map[string]any{
					"error": errors.New("error tool \"test_tool\" requires confirmation, please approve or reject"),
				}, "user"),
				genai.NewContentFromFunctionResponse("test_tool", map[string]any{"result": "ok"}, "user"),
			},
		},
		{
			name: "Conditional Confirmation Required and is rejected",
			toolConfig: functiontool.Config{
				Name: "test_tool",
				RequireConfirmationProvider: func(args SimpleArgs) bool {
					return args.Num < 5
				},
			},
			args:                    map[string]any{"Num": 4},
			confirmFunctionResponse: &genai.FunctionResponse{Name: toolconfirmation.FunctionCallName, Response: map[string]any{"confirmed": false}},
			want: []*genai.Content{
				genai.NewContentFromFunctionCall("test_tool", map[string]any{"Num": 4}, "model"),
				genai.NewContentFromFunctionCall(toolconfirmation.FunctionCallName, map[string]any{
					"originalFunctionCall": &genai.FunctionCall{
						Args: map[string]any{"Num": 4},
						Name: "test_tool",
					},
					"toolConfirmation": toolconfirmation.ToolConfirmation{
						Hint: "Please approve or reject the tool call test_tool() by responding with a FunctionResponse with an expected ToolConfirmation payload.",
					},
				}, "model"),
				genai.NewContentFromFunctionResponse("test_tool", map[string]any{
					"error": errors.New("error tool \"test_tool\" requires confirmation, please approve or reject"),
				}, "user"),
				genai.NewContentFromFunctionResponse("test_tool", map[string]any{
					"error": errors.New("error tool \"test_tool\" call is rejected"),
				}, "user"),
			},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			mockModel := &testutil.MockModel{
				Responses: []*genai.Content{
					genai.NewContentFromFunctionCall("test_tool", tc.args, genai.RoleModel),
				},
			}

			// Setup tool
			myTool, err := functiontool.New(tc.toolConfig, okFunc)
			if err != nil {
				t.Fatalf("Failed to create tool: %v", err)
			}

			a, err := llmagent.New(llmagent.Config{
				Name:  "simple agent",
				Model: mockModel,
				Tools: []tool.Tool{myTool},
			})
			if err != nil {
				t.Fatalf("failed to create llm agent: %v", err)
			}

			runner := testutil.NewTestAgentRunner(t, a)
			eventCount := 0

			ev := runner.Run(t, "id", "message")

			var confirmFunctionCall *genai.FunctionCall
			for got, err := range ev {
				if err != nil && err.Error() == "no data" {
					break
				}
				if err != nil {
					// Check if an error was expected
					t.Fatalf("runner returned unexpected error: %v", err)
					// If error was expected, we can stop here or check for a specific error type.
					return
				}

				if eventCount >= len(tc.want) {
					t.Fatalf("stream generated more values than the expected %d. Got: %+v", len(tc.want), got.Content)
				}

				if diff := cmp.Diff(tc.want[eventCount], got.Content, cmpopts.IgnoreFields(genai.FunctionCall{}, "ID"),
					cmp.Transformer("StringifyMapErrors", func(m map[string]any) map[string]any {
						out := make(map[string]any, len(m))
						for k, v := range m {
							// Check if the value inside the map is an error
							if err, ok := v.(error); ok {
								out[k] = err.Error() // Convert to string
							} else {
								out[k] = v // Keep as is
							}
						}
						return out
					}), cmpopts.IgnoreFields(genai.FunctionResponse{}, "ID")); diff != "" {
					t.Errorf("LoopAgent Run() mismatch (-want +got):\n%s", diff)
				}
				for _, p := range got.Content.Parts {
					if p.FunctionCall != nil && p.FunctionCall.Name == toolconfirmation.FunctionCallName {
						confirmFunctionCall = p.FunctionCall
					}
				}
				eventCount++
			}

			if confirmFunctionCall != nil && tc.confirmFunctionResponse != nil {
				tc.confirmFunctionResponse.ID = confirmFunctionCall.ID
				ev := runner.RunContent(t, "id", &genai.Content{
					Parts: []*genai.Part{{FunctionResponse: tc.confirmFunctionResponse}},
				})
				for got, err := range ev {
					if err != nil && err.Error() == "no data" {
						break
					}
					if err != nil {
						// Check if an error was expected
						t.Fatalf("runner returned unexpected error: %v", err)
						// If error was expected, we can stop here or check for a specific error type.
						return
					}

					if eventCount >= len(tc.want) {
						t.Fatalf("stream generated more values than the expected %d. Got: %+v", len(tc.want), got.Content)
					}

					if diff := cmp.Diff(tc.want[eventCount], got.Content, cmpopts.IgnoreFields(genai.FunctionCall{}, "ID"),
						cmp.Transformer("StringifyMapErrors", func(m map[string]any) map[string]any {
							out := make(map[string]any, len(m))
							for k, v := range m {
								// Check if the value inside the map is an error
								if err, ok := v.(error); ok {
									out[k] = err.Error() // Convert to string
								} else {
									out[k] = v // Keep as is
								}
							}
							return out
						}), cmpopts.IgnoreFields(genai.FunctionResponse{}, "ID")); diff != "" {
						t.Errorf("LoopAgent Run() mismatch (-want +got):\n%s", diff)
					}
					for _, p := range got.Content.Parts {
						if p.FunctionCall != nil && p.FunctionCall.Name == toolconfirmation.FunctionCallName {
							confirmFunctionCall = p.FunctionCall
						}
					}
					eventCount++
				}
			}

			// Final check on the number of events
			if eventCount != len(tc.want) {
				t.Errorf("unexpected stream length, want %d got %d", len(tc.want), eventCount)
			}
		})
	}
}

// Mock types for TArgs and TResults
type TestArgs struct {
	Name string
}

type TestResult struct {
	Value int
}

func TestNew_RequireConfirmationProvider_Validation(t *testing.T) {
	// A dummy handler to satisfy the function signature
	dummyHandler := func(_ tool.Context, _ TestArgs) (TestResult, error) {
		return TestResult{Value: 1}, nil
	}

	expectedError := fmt.Sprintf("error RequireConfirmationProvider must be a function with signature func(%T) bool", TestArgs{})

	tests := []struct {
		name         string
		provider     any  // The RequireConfirmationProvider value to test
		expectsError bool // Substring expected in the error message; empty if no error expected
	}{
		// --- Happy Paths ---
		{
			name:         "Valid: Nil provider is allowed",
			provider:     nil,
			expectsError: false,
		},
		{
			name:         "Valid: Correct function signature",
			provider:     func(args TestArgs) bool { return true },
			expectsError: false,
		},

		// --- Edge Cases / Validation Errors ---
		{
			name:         "Invalid: Provider is not a function (it's a struct)",
			provider:     struct{}{},
			expectsError: true,
		},
		{
			name:         "Invalid: Provider is not a function (it's a primitive)",
			provider:     123,
			expectsError: true,
		},
		{
			name:         "Invalid: Function has 0 arguments",
			provider:     func() bool { return true },
			expectsError: true,
		},
		{
			name:         "Invalid: Function has too many arguments (2)",
			provider:     func(a TestArgs, b int) bool { return true },
			expectsError: true,
		},
		{
			name:         "Invalid: Argument type mismatch (int instead of TestArgs)",
			provider:     func(n int) bool { return true },
			expectsError: true,
		},
		{
			name:         "Invalid: Argument type mismatch (pointer vs value)",
			provider:     func(a *TestArgs) bool { return true },
			expectsError: true,
		},
		{
			name:         "Invalid: Function returns nothing",
			provider:     func(args TestArgs) {},
			expectsError: true,
		},
		{
			name:         "Invalid: Function returns too many values",
			provider:     func(args TestArgs) (bool, error) { return true, nil },
			expectsError: true,
		},
		{
			name:         "Invalid: Return type mismatch (returns int instead of bool)",
			provider:     func(args TestArgs) int { return 1 },
			expectsError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Construct config with the provider under test
			cfg := functiontool.Config{
				RequireConfirmationProvider: tt.provider,
			}

			tool, err := functiontool.New(cfg, dummyHandler)

			// Check results
			if !tt.expectsError {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if tool == nil {
					t.Error("expected valid tool, got nil")
				}
			} else {
				if err == nil {
					t.Error("expected error but got nil")
				} else if !strings.Contains(err.Error(), expectedError) {
					t.Errorf("error message mismatch.\nExpected substring: %q\nGot: %q", expectedError, err.Error())
				}
			}
		})
	}
}

func TestNew_InvalidInputType(t *testing.T) {
	testCases := []struct {
		name       string
		createTool func() (tool.Tool, error)
		wantErrMsg string
	}{
		{
			name: "string_input",
			createTool: func() (tool.Tool, error) {
				return functiontool.New(functiontool.Config{
					Name:        "string_tool",
					Description: "a tool with string input",
				}, func(ctx tool.Context, input string) (string, error) {
					return input, nil
				})
			},
			wantErrMsg: "input must be a struct type, got: string",
		},
		{
			name: "int_input",
			createTool: func() (tool.Tool, error) {
				return functiontool.New(functiontool.Config{
					Name:        "int_tool",
					Description: "a tool with int input",
				}, func(ctx tool.Context, input int) (int, error) {
					return input, nil
				})
			},
			wantErrMsg: "input must be a struct type, got: int",
		},
		{
			name: "bool_input",
			createTool: func() (tool.Tool, error) {
				return functiontool.New(functiontool.Config{
					Name:        "bool_tool",
					Description: "a tool with bool input",
				}, func(ctx tool.Context, input bool) (bool, error) {
					return input, nil
				})
			},
			wantErrMsg: "input must be a struct type, got: bool",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := tc.createTool()
			if err == nil {
				t.Fatalf("functiontool.New() succeeded, want error containing %q", tc.wantErrMsg)
			}
			if !errors.Is(err, functiontool.ErrInvalidArgument) {
				t.Fatalf("functiontool.New() error = %v, want %v", err, functiontool.ErrInvalidArgument)
			}
		})
	}
}

func TestFunctionTool_PanicRecovery(t *testing.T) {
	type Args struct {
		Value string `json:"value"`
	}

	panicHandler := func(ctx tool.Context, input Args) (string, error) {
		panic("intentional panic for testing")
	}

	panicTool, err := functiontool.New(functiontool.Config{
		Name:        "panic_tool",
		Description: "a tool that always panics",
	}, panicHandler)
	if err != nil {
		t.Fatalf("NewFunctionTool failed: %v", err)
	}

	funcTool, ok := panicTool.(toolinternal.FunctionTool)
	if !ok {
		t.Fatal("panicTool does not implement toolinternal.FunctionTool")
	}

	result, err := funcTool.Run(createToolContext(t), map[string]any{"value": "test"})
	if err == nil {
		t.Fatal("expected error from panic recovery, got nil")
	}
	if result != nil {
		t.Errorf("expected nil result, got %v", result)
	}

	expectedErrParts := []string{
		"panic in tool",
		"panic_tool",
		"intentional panic for testing",
		"stack:",
	}
	for _, part := range expectedErrParts {
		if !strings.Contains(err.Error(), part) {
			t.Errorf("expected error to contain %q, but it did not. Error: %v", part, err)
		}
	}
}
