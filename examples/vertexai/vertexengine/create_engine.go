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

package main

import (
	"context"
	"fmt"
	"log"
	"os"

	aiplatform "cloud.google.com/go/aiplatform/apiv1"
	"cloud.google.com/go/aiplatform/apiv1/aiplatformpb"
	"google.golang.org/api/option"
)

// main defines an example of how to initialize a vertex ai reasoning engine
func main() {
	ctx := context.Background()
	projectID := os.Getenv("GOOGLE_CLOUD_PROJECT")
	if projectID == "" {
		log.Fatalf("Env var GOOGLE_CLOUD_PROJECT is not set")
	}
	location := os.Getenv("GOOGLE_CLOUD_LOCATION")
	if location == "" {
		log.Fatalf("Env var GOOGLE_CLOUD_LOCATION is not set")
	}

	err := createReasoningEngine(ctx, projectID, location, "adk-go", "A reasoning engine created by an adk go sample")
	if err != nil {
		log.Fatalf("Failed to create reasoning engine: %v", err)
	}
}

func createReasoningEngine(ctx context.Context, projectID, location, displayName, description string) error {
	// Construct the parent resource name
	parent := fmt.Sprintf("projects/%s/locations/%s", projectID, location)

	// Construct the regional endpoint
	endpoint := fmt.Sprintf("%s-aiplatform.googleapis.com:443", location)

	// Create the client
	client, err := aiplatform.NewReasoningEngineClient(ctx, option.WithEndpoint(endpoint))
	if err != nil {
		return fmt.Errorf("failed to create ReasoningEngineClient: %w", err)
	}
	defer func() {
		if err := client.Close(); err != nil {
			log.Printf("Warning: failed to close client: %v", err)
		}
	}()

	// Define the ReasoningEngine object
	reasoningEngine := &aiplatformpb.ReasoningEngine{
		DisplayName: displayName,
		Description: description,
	}

	// Create the request
	req := &aiplatformpb.CreateReasoningEngineRequest{
		Parent:          parent,
		ReasoningEngine: reasoningEngine,
	}

	// Call the CreateReasoningEngine method
	op, err := client.CreateReasoningEngine(ctx, req)
	if err != nil {
		return fmt.Errorf("failed to call CreateReasoningEngine: %w", err)
	}

	// Wait for the long-running operation to complete
	resp, err := op.Wait(ctx)
	if err != nil {
		return fmt.Errorf("failed to wait for operation: %w", err)
	}

	fmt.Printf("Successfully created ReasoningEngine: %s\n", resp.Name)
	return nil
}
