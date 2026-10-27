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

package agentengine

import (
	"context"
	"io"
	"net"
	"os"
	"strings"
	"testing"

	"cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	"cloud.google.com/go/longrunning/autogen/longrunningpb"
	"github.com/google/go-cmp/cmp"
	"github.com/spf13/cobra"
	"google.golang.org/api/option"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/types/known/anypb"
)

func environmentCommand() *cobra.Command {
	cmd := &cobra.Command{Use: "agentengine", RunE: agentEngineCmd.RunE, SilenceErrors: true, SilenceUsage: true}
	registerEnvironmentFlags(cmd, &flags.agentEngine)
	return cmd
}

func TestEnvironmentFlagsRegistered(t *testing.T) {
	for name, wantType := range map[string]string{"env": "stringArray", "default_env": "bool"} {
		flag := agentEngineCmd.PersistentFlags().Lookup(name)
		if flag == nil || flag.Value.Type() != wantType {
			t.Errorf("agentengine --%s is not registered as %s", name, wantType)
		}
	}
	if flag := agentEngineCmd.PersistentFlags().Lookup("default_env"); flag != nil && flag.DefValue != "true" {
		t.Errorf("--default_env default = %q, want true", flag.DefValue)
	}
}

// captureEnvironmentOutput covers the CLI's process-wide progress printer too,
// rather than checking only errors returned by the parser.
func captureEnvironmentOutput(t *testing.T, run func()) string {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "output")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := f.Close(); err != nil {
			t.Error(err)
		}
	})
	saved := os.Stdout
	os.Stdout = f
	defer func() { os.Stdout = saved }()
	run()
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		t.Fatal(err)
	}
	data, err := io.ReadAll(f)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

func TestEnvironmentFlagsRejectInvalidBeforePreparingDeployment(t *testing.T) {
	for name, args := range map[string][]string{
		"missing equals": {"--env=DO_NOT_LOG_THIS_VALUE"},
		"empty name":     {"--env==DO_NOT_LOG_THIS_VALUE"},
		"leading digit":  {"--env=9INVALID=DO_NOT_LOG_THIS_VALUE"},
		"hyphen":         {"--env=INVALID-NAME=DO_NOT_LOG_THIS_VALUE"},
		"space":          {"--env=INVALID NAME=DO_NOT_LOG_THIS_VALUE"},
		"duplicate":      {"--env=VALID=DO_NOT_LOG_THIS_VALUE", "--env=VALID=OTHER_PRIVATE_VALUE"},
	} {
		t.Run(name, func(t *testing.T) {
			// The invalid entry point also bounds the test if env validation is
			// removed: it still cannot reach archiving, credentials or an API.
			resetFlags(t, "not-a-go-file")
			base := flags.build.tempDir
			cmd := environmentCommand()
			cmd.SetArgs(args)
			output := captureEnvironmentOutput(t, func() {
				err := cmd.Execute()
				if err == nil || !strings.Contains(err.Error(), "--env entry") {
					t.Errorf("Execute() error = %v, want an --env validation error", err)
				}
			})
			if strings.Contains(output, "DO_NOT_LOG_THIS_VALUE") || strings.Contains(output, "OTHER_PRIVATE_VALUE") {
				t.Error("CLI printed an environment value")
			}
			entries, err := os.ReadDir(base)
			if err != nil {
				t.Fatal(err)
			}
			if len(entries) != 0 {
				t.Errorf("invalid --env created %d temporary entries", len(entries))
			}
			if flags.source.origEntryPointPath != "" {
				t.Error("invalid --env reached entry-point processing")
			}
		})
	}
}

// Requests pass through Cobra, computeFlags, the actual generated gRPC client
// and completed long-running operations. No Google credentials or live API are
// used; the server only listens on loopback.
func TestEnvironmentDeploymentRequests(t *testing.T) {
	defaults := map[string]string{
		"GOOGLE_CLOUD_REGION": "test-region",
		"NUM_WORKERS":         "1",
		"GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY":         "true",
		"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true",
	}
	tests := []struct {
		name       string
		args       []string
		update     bool
		want       map[string]string
		wantUpdate bool
	}{
		{name: "create defaults", want: defaults},
		{name: "create overrides and preserves values", args: []string{"--env=NUM_WORKERS=3", "--env=SERVICE_URL=https://example.test/?a=1,b=2", "--env=_empty="}, want: map[string]string{
			"GOOGLE_CLOUD_REGION": "test-region", "NUM_WORKERS": "3", "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY": "true", "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true",
			"SERVICE_URL": "https://example.test/?a=1,b=2", "_empty": "",
		}},
		{name: "create without defaults", args: []string{"--default_env=false", "--env=lower_case=only-custom"}, want: map[string]string{"lower_case": "only-custom"}},
		{name: "create empty environment", args: []string{"--default_env=false"}, want: map[string]string{}},
		{name: "update preserves environment by default", update: true},
		{name: "update overrides defaults", args: []string{"--env=NUM_WORKERS=5"}, update: true, want: map[string]string{
			"GOOGLE_CLOUD_REGION": "test-region", "NUM_WORKERS": "5", "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY": "true", "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true",
		}, wantUpdate: true},
		{name: "update custom environment", args: []string{"--default_env=false", "--env=ONLY=custom"}, update: true, want: map[string]string{"ONLY": "custom"}, wantUpdate: true},
		{name: "update explicitly enables defaults", args: []string{"--default_env=true"}, update: true, want: defaults, wantUpdate: true},
		{name: "update clears environment", args: []string{"--default_env=false"}, update: true, want: map[string]string{}, wantUpdate: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resetFlags(t, "not-a-go-file")
			flags.gcloud.projectName = "test-project"
			flags.gcloud.region = "test-region"
			cmd := environmentCommand()
			cmd.SetArgs(tt.args)
			// Run the real command callback up to its deliberately invalid
			// entry point. This also exercises explicit --default_env=true
			// detection, which a value-only check would miss.
			output := captureEnvironmentOutput(t, func() {
				if err := cmd.Execute(); err == nil || strings.Contains(err.Error(), "--env entry") {
					t.Fatalf("Execute() error = %v, want the bounded entry-point error", err)
				}
			})
			if strings.Contains(output, "only-custom") || strings.Contains(output, "https://example.test/") {
				t.Error("CLI printed an environment value")
			}
			flags.source.entryPointPath = "main.go"
			if err := flags.computeFlags(); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(flags.build.archivePath, []byte("test archive"), 0o600); err != nil {
				t.Fatal(err)
			}

			server := &environmentDeploymentServer{created: make(chan *aiplatformpb.CreateReasoningEngineRequest, 1), updated: make(chan *aiplatformpb.UpdateReasoningEngineRequest, 1)}
			opts := environmentClientOptions(t, server)
			var spec *aiplatformpb.ReasoningEngineSpec
			if tt.update {
				flags.agentEngine.agentEngineID = "test-engine"
				if err := flags.gcloudUpdateAgentEngine(opts...); err != nil {
					t.Fatal(err)
				}
				req := <-server.updated
				wantMask := []string{"spec.source_code_spec", "spec.class_methods"}
				if tt.wantUpdate {
					wantMask = append(wantMask, "spec.deployment_spec.env")
				}
				if diff := cmp.Diff(wantMask, req.GetUpdateMask().GetPaths()); diff != "" {
					t.Errorf("update mask mismatch (-want +got):\n%s", diff)
				}
				spec = req.GetReasoningEngine().GetSpec()
				if !tt.wantUpdate && spec.GetDeploymentSpec() != nil {
					t.Error("update without environment flags changed deployment settings")
				}
			} else {
				if err := flags.gcloudDeployToAgentEngine(opts...); err != nil {
					t.Fatal(err)
				}
				req := <-server.created
				if req.GetParent() != "projects/test-project/locations/test-region" {
					t.Errorf("create parent = %q", req.GetParent())
				}
				spec = req.GetReasoningEngine().GetSpec()
			}
			if string(spec.GetSourceCodeSpec().GetInlineSource().GetSourceArchive()) != "test archive" || len(spec.GetClassMethods()) == 0 {
				t.Error("environment configuration lost the source archive or class methods")
			}
			if tt.want != nil {
				got := make(map[string]string)
				for _, env := range spec.GetDeploymentSpec().GetEnv() {
					if _, exists := got[env.GetName()]; exists {
						t.Errorf("request contains duplicate variable %q", env.GetName())
					}
					got[env.GetName()] = env.GetValue()
				}
				if diff := cmp.Diff(tt.want, got); diff != "" {
					t.Errorf("environment mismatch (-want +got):\n%s", diff)
				}
			}
		})
	}
}

type environmentDeploymentServer struct {
	aiplatformpb.UnimplementedReasoningEngineServiceServer
	created chan *aiplatformpb.CreateReasoningEngineRequest
	updated chan *aiplatformpb.UpdateReasoningEngineRequest
}

func completedEnvironmentOperation() (*longrunningpb.Operation, error) {
	response, err := anypb.New(&aiplatformpb.ReasoningEngine{Name: "test-engine"})
	if err != nil {
		return nil, err
	}
	return &longrunningpb.Operation{Name: "test-operation", Done: true, Result: &longrunningpb.Operation_Response{Response: response}}, nil
}

func (s *environmentDeploymentServer) CreateReasoningEngine(_ context.Context, req *aiplatformpb.CreateReasoningEngineRequest) (*longrunningpb.Operation, error) {
	s.created <- req
	return completedEnvironmentOperation()
}

func (s *environmentDeploymentServer) UpdateReasoningEngine(_ context.Context, req *aiplatformpb.UpdateReasoningEngineRequest) (*longrunningpb.Operation, error) {
	s.updated <- req
	return completedEnvironmentOperation()
}

func environmentClientOptions(t *testing.T, server *environmentDeploymentServer) []option.ClientOption {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	grpcServer := grpc.NewServer()
	aiplatformpb.RegisterReasoningEngineServiceServer(grpcServer, server)
	t.Cleanup(grpcServer.Stop)
	go func() { _ = grpcServer.Serve(listener) }()
	connection, err := grpc.NewClient(listener.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = connection.Close() })
	return []option.ClientOption{option.WithGRPCConn(connection), option.WithoutAuthentication()}
}
