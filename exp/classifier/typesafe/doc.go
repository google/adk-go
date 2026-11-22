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

// Package typesafe implements [google.golang.org/adk/v2/exp/classifier.Classifier]
// for TypeSafe's Jev decision models via the /v1/systemone HTTP API.
//
// This package is experimental; its API and behavior may change or be removed
// in a later version without notice.
//
// Jev returns calibrated probability distributions from a single forward pass
// rather than generating text. Every question passed to [Classifier.Classify]
// must set Instructions, and [classifier.Binary] questions are sent as Jev
// "noul" questions. Only text and JSON inputs are accepted. A [genai.Content]
// containing non-text parts is rejected before any network call, as is any
// other google.golang.org/genai struct value, such as a []*genai.Content
// history.
//
//	clf, err := typesafe.New(typesafe.Config{}) // reads TYPESAFE_API_KEY on Classify
//	if err != nil {
//		return err
//	}
//	resp, err := clf.Classify(ctx, "My invoice was charged twice.", map[string]classifier.Question{
//		"department": classifier.Choice{
//			Instructions: "Which team should handle this ticket?",
//			Criteria: map[string]any{
//				"billing":   "Invoices, refunds, or subscription charges",
//				"technical": nil,
//			},
//		},
//	})
package typesafe
