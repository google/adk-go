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

// Package codeexecutor lets an LLM agent run the code its model writes. Set
// an executor on llmagent.Config.CodeExecutor.
//
// NewBuiltInExecutor returns the executor that lets a Gemini model run the
// code itself: the agent adds the model's built-in code-execution tool to
// every request, and the model returns the code it ran and its output as
// parts of its response. Nothing runs in the agent's process. The agent
// accepts a model name only when the name, with any "models/", Vertex AI or
// provider path removed, starts with "gemini-" (case-sensitive). For any
// other name the run fails before the model is called, with the error
// "Gemini code execution tool is not supported for model <name>". Setting the
// environment variable ADK_DISABLE_GEMINI_MODEL_ID_CHECK to "true" or "1", in
// any letter case, turns that check off for the whole process.
//
// To run code in an environment of your own, implement CodeExecutor:
// ExecuteCode runs one piece of code and returns what it printed as a Result,
// and Settings returns the options that readers of the executor apply, such
// as the delimiters that fence code in model text. A Settings field left
// unset means Python's default for that field, and Settings.WithDefaults
// fills the defaults in. The CodeExecutor example shows a minimal executor.
//
// The agent runs an executor that is not a BuiltIn itself, in its own
// process. It runs the first code block of each model response, an
// executable-code part or a block fenced in the text by one of
// Settings.CodeBlockDelimiters: it yields an event with the code, calls
// ExecuteCode, yields an event with the result and calls the model again. In
// later requests the code and the result are sent as fenced text. After
// Settings.ErrorRetryAttempts consecutive executions with a non-empty Stderr,
// the agent runs no more code in that invocation. Only with
// Settings.OptimizeDataFile does the agent tell the model to fence its code
// and pass CSV data from user messages as input files; otherwise the agent's
// instruction has to say how to fence code. Running such an executor needs an
// artifact service on the runner (runner.Config.ArtifactService), or the run
// fails with "artifact service is not initialized" after the first execution.
// Under agent.RunConfig.SupportCFC an agent without a parent agent uses a
// built-in executor instead. The doc comment of llmagent.Config.CodeExecutor
// gives the details, and the example of llmagent.Config named CodeExecutor
// builds an agent that runs a custom executor.
package codeexecutor
