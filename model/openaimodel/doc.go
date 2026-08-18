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

// Package openaimodel provides a client for interacting with OpenAI's API.
//
// EXPERIMENTAL: This package is experimental and its behavior may change or be
// removed in the future.
//
// It implements the model.LLM interface, making it compatible with
// providers that expose the OpenAI Responses API surface. This package
// allows for easy integration of OpenAI's language models into applications.
//
// Clients construct a ClientConfig and pass it to NewModel:
//
//	ctx := context.Background()
//	cfg := &openaimodel.ClientConfig{APIKey: os.Getenv("OPENAI_API_KEY")}
//	llm, err := openaimodel.NewModel(ctx, openai.ChatModelGPT4oMini, cfg)
//	if err != nil {
//		log.Fatal(err)
//	}
package openaimodel
