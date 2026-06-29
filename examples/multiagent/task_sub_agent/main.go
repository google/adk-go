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

// Package main demonstrates how a "task mode" agent can act as a sub-agent
// to an LLM agent, effectively extracting structured data from a
// conversational flow.
//
// The coordinator agent delegates interactions to two task sub-agents:
//
//   - order_collector collects the user's food order (from a menu of Pizza,
//     Burger, Salad) and returns a structured list of OrderItem.
//   - payment_collector collects credit card and CVV information, returning
//     a PaymentInfo object.
//
// Once both tasks finish, the coordinator invokes the place_order tool with
// the structured data returned by the sub-agents.
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

// OrderItem represents a single ordered food item.
type OrderItem struct {
	Name     string `json:"name" jsonschema:"Name of the food item ordered"`
	Quantity int    `json:"quantity" jsonschema:"Quantity ordered"`
}

// PaymentInfo is the output schema for the payment collection task.
type PaymentInfo struct {
	CreditCardNumber string `json:"credit_card_number"`
	CVV              string `json:"cvv"`
}

// PlaceOrderInput is the input for the place_order tool.
type PlaceOrderInput struct {
	Orders      []OrderItem `json:"orders"`
	PaymentInfo PaymentInfo `json:"payment_info"`
}

// PlaceOrderOutput is the output of the place_order tool.
type PlaceOrderOutput struct {
	Result string `json:"result"`
}

// placeOrder mocks an order placement operation.
func placeOrder(_ agent.Context, in PlaceOrderInput) (PlaceOrderOutput, error) {
	totalItems := 0
	for _, item := range in.Orders {
		totalItems += item.Quantity
	}
	return PlaceOrderOutput{
		Result: fmt.Sprintf("Successfully placed order for %d items.", totalItems),
	}, nil
}

// ConfirmationInput is the (empty) input for the confirmation tool.
type ConfirmationInput struct{}

// ConfirmationOutput is the output of the confirmation tool.
type ConfirmationOutput struct {
	Result string `json:"result"`
}

// confirmation confirms proceeding with the order.
func confirmation(_ agent.Context, _ ConfirmationInput) (ConfirmationOutput, error) {
	return ConfirmationOutput{Result: "Proceeding with order."}, nil
}

func main() {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, "gemini-flash-latest", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	placeOrderTool, err := functiontool.New(functiontool.Config{
		Name:        "place_order",
		Description: "Mock an order placement operation.",
	}, placeOrder)
	if err != nil {
		log.Fatalf("Failed to create place_order tool: %v", err)
	}

	confirmationTool, err := functiontool.New(functiontool.Config{
		Name:                "confirmation",
		Description:         "Confirm proceeding with the order.",
		RequireConfirmation: true,
	}, confirmation)
	if err != nil {
		log.Fatalf("Failed to create confirmation tool: %v", err)
	}

	// Output schema for order_collector: a list of OrderItem.
	// Mirrors Python's `output_schema=list[OrderItem]`.
	orderListSchema := &genai.Schema{
		Type: genai.TypeArray,
		Items: &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"name": {
					Type:        genai.TypeString,
					Description: "Name of the food item ordered",
				},
				"quantity": {
					Type:        genai.TypeInteger,
					Description: "Quantity ordered",
				},
			},
			Required: []string{"name", "quantity"},
		},
	}

	// Output schema for payment_collector: a PaymentInfo object.
	paymentInfoSchema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"credit_card_number": {Type: genai.TypeString},
			"cvv":                {Type: genai.TypeString},
		},
		Required: []string{"credit_card_number", "cvv"},
	}

	orderCollector, err := llmagent.New(llmagent.Config{
		Name:         "order_collector",
		Model:        model,
		Description:  "Collects the food order from the user.",
		Mode:         llmagent.ModeTask,
		OutputSchema: orderListSchema,
		Tools:        []tool.Tool{confirmationTool},
		Instruction: `You are an order collection assistant for a food delivery service.
Our menu today has exactly 3 items: 1. Pizza, 2. Burger, 3. Salad.
Ask the user what they would like to order and collect their choice and quantity.
Do not offer anything else.
If the combined quantity of items exceeds 5, you MUST use the ` + "`confirmation`" + ` tool to get user's confirmation before proceeding.
Do not ask for confirmation in natural language, always use the confirmation tool.
Once you have their final order and confirmation if needed, finish your task.
`,
	})
	if err != nil {
		log.Fatalf("Failed to create order_collector: %v", err)
	}

	paymentCollector, err := llmagent.New(llmagent.Config{
		Name:         "payment_collector",
		Model:        model,
		Description:  "Collects credit card and CVV from the user.",
		Mode:         llmagent.ModeTask,
		OutputSchema: paymentInfoSchema,
		Instruction: `You are a payment collection assistant.
Ask the user for their credit card number and CVV.
Once you have both pieces of information, finish your task.
`,
	})
	if err != nil {
		log.Fatalf("Failed to create payment_collector: %v", err)
	}

	rootAgent, err := llmagent.New(llmagent.Config{
		Name:      "coordinator",
		Model:     model,
		SubAgents: []agent.Agent{orderCollector, paymentCollector},
		Tools:     []tool.Tool{placeOrderTool},
		Instruction: `You are a helpful coordinator for a food delivery service.
You need both order and payment information to place an order.
You must verify food order with 'order_collector'
`,
	})
	if err != nil {
		log.Fatalf("Failed to create coordinator: %v", err)
	}

	config := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(rootAgent),
	}

	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
