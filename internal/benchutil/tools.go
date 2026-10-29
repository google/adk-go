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

package benchutil

import (
	"context"
	"fmt"
	"iter"
	"strings"
	"sync/atomic"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// ToolInstruction is a support-agent instruction of realistic length with
// one state placeholder, so every model call pays for templating.
var ToolInstruction = "You are a customer support assistant for {user_name}. " + strings.Repeat(
	"Answer questions about the weather, orders and currency conversion. "+
		"Call a tool whenever the answer depends on live data, and never guess a value a tool can return. "+
		"Keep answers short, quote the numbers the tool returned, and say which tool you used. ", 4)

// ToolState is the session state ToolInstruction needs.
func ToolState() map[string]any {
	return map[string]any{"user_name": "Ada", "user:tier": "gold"}
}

type WeatherArgs struct {
	City string `json:"city" jsonschema:"the city to report on"`
	Unit string `json:"unit,omitempty" jsonschema:"celsius or fahrenheit"`
}

type WeatherResult struct {
	TempC      float64 `json:"temp_c"`
	Conditions string  `json:"conditions"`
	Humidity   int     `json:"humidity"`
}

type OrderArgs struct {
	OrderID      string `json:"order_id" jsonschema:"the order number"`
	IncludeItems bool   `json:"include_items,omitempty"`
}

type OrderItem struct {
	SKU   string  `json:"sku"`
	Qty   int     `json:"qty"`
	Price float64 `json:"price"`
}

type OrderResult struct {
	Status string      `json:"status"`
	Items  []OrderItem `json:"items,omitempty"`
}

type ConvertArgs struct {
	Amount float64 `json:"amount"`
	From   string  `json:"from" jsonschema:"ISO 4217 code"`
	To     string  `json:"to" jsonschema:"ISO 4217 code"`
}

type ConvertResult struct {
	Value float64 `json:"value"`
	Rate  float64 `json:"rate"`
}

// Tools returns three function tools with struct arguments and results, the
// shape of a typical tool-using agent.
func Tools() ([]tool.Tool, error) {
	weather, err := functiontool.New(functiontool.Config{Name: "get_weather", Description: "Returns the current weather for a city."},
		func(_ agent.Context, in WeatherArgs) (WeatherResult, error) {
			return WeatherResult{TempC: 22.5, Conditions: "sunny in " + in.City, Humidity: 40}, nil
		})
	if err != nil {
		return nil, err
	}
	order, err := functiontool.New(functiontool.Config{Name: "lookup_order", Description: "Looks up an order by its number."},
		func(_ agent.Context, in OrderArgs) (OrderResult, error) {
			return OrderResult{Status: "shipped", Items: []OrderItem{{SKU: in.OrderID + "-1", Qty: 2, Price: 9.99}}}, nil
		})
	if err != nil {
		return nil, err
	}
	convert, err := functiontool.New(functiontool.Config{Name: "convert_currency", Description: "Converts an amount between currencies."},
		func(_ agent.Context, in ConvertArgs) (ConvertResult, error) {
			return ConvertResult{Value: in.Amount * 1.1, Rate: 1.1}, nil
		})
	if err != nil {
		return nil, err
	}
	return []tool.Tool{weather, order, convert}, nil
}

// ToolCallModel answers like a model that uses one tool per turn: a request
// whose last content is not a function response gets a get_weather call, and
// one that answers a call gets the final text. A turn is therefore two model
// calls around one tool call. Responses are fresh on every call because the
// flow stores and mutates them.
type ToolCallModel struct {
	Calls        atomic.Int64
	LastContents atomic.Int64
	LastTools    atomic.Int64
}

func (m *ToolCallModel) Name() string { return "tool-call-stub" }

func (m *ToolCallModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	m.Calls.Add(1)
	m.LastContents.Store(int64(len(req.Contents)))
	m.LastTools.Store(int64(len(req.Tools)))
	answered := false
	if n := len(req.Contents); n > 0 && req.Contents[n-1] != nil {
		for _, p := range req.Contents[n-1].Parts {
			if p != nil && p.FunctionResponse != nil {
				answered = true
			}
		}
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		if answered {
			yield(&model.LLMResponse{
				Content:      genai.NewContentFromText("It is 22.5°C and sunny in Paris. "+BenchText, genai.RoleModel),
				FinishReason: genai.FinishReasonStop,
				TurnComplete: true,
			}, nil)
			return
		}
		yield(&model.LLMResponse{
			Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{
				FunctionCall: &genai.FunctionCall{Name: "get_weather", Args: map[string]any{"city": "Paris", "unit": "celsius"}},
			}}},
			FinishReason: genai.FinishReasonStop,
			TurnComplete: true,
		}, nil)
	}
}

// CheckToolTurn reports an error unless the last turn made exactly two model
// calls, the second of which saw a function response.
func (m *ToolCallModel) CheckToolTurn(callsBefore int64) error {
	if got := m.Calls.Load() - callsBefore; got != 2 {
		return fmt.Errorf("model calls in turn = %d, want 2", got)
	}
	return nil
}
