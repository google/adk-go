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

// Package agentengine handles command line parameters and execution logic for agentengine deployment.

package agentengine

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	aiplatform "cloud.google.com/go/aiplatform/apiv1"
	"cloud.google.com/go/aiplatform/apiv1/aiplatformpb"
	"github.com/spf13/cobra"
	"google.golang.org/api/option"

	"google.golang.org/adk/cmd/adkgo/internal/deploy"
	"google.golang.org/adk/internal/cli/util"
)

type gCloudFlags struct {
	region      string
	projectName string
}

type agentEngineServiceFlags struct {
	name        string
	displayName string
	serverPort  int
	api         bool // enable api or not
}

type buildFlags struct {
	tempDir             string
	execPath            string
	execFile            string
	dockerfileBuildPath string
	archivePath         string
}

type sourceFlags struct {
	srcBasePath        string
	entryPointPath     string
	origEntryPointPath string
	sourceDir          string
}

type deployAgentEngineFlags struct {
	gcloud      gCloudFlags
	agentEngine agentEngineServiceFlags
	build       buildFlags
	source      sourceFlags
}

var flags deployAgentEngineFlags

// agentEngineCmd represents the agentEngine command
var agentEngineCmd = &cobra.Command{
	Use:   "agentengine",
	Short: "Deploys the application to Agent Engine.",
	Long:  `Deploys the application to Agent Engine. It creates a source archive, uploads it to create a Reasoning Engine, and cleans up temporary files.`,
	RunE: func(cmd *cobra.Command, args []string) error {
		return flags.deployOnagentEngine()
	},
}

// init creates flags and adds subcommand to parent
func init() {
	deploy.DeployCmd.AddCommand(agentEngineCmd)

	agentEngineCmd.PersistentFlags().StringVarP(&flags.gcloud.region, "region", "r", "", "GCP Region")
	agentEngineCmd.PersistentFlags().StringVarP(&flags.gcloud.projectName, "project_name", "p", "", "GCP Project Name")
	agentEngineCmd.PersistentFlags().StringVarP(&flags.agentEngine.name, "name", "s", "", "Agent Engine name")
	agentEngineCmd.PersistentFlags().StringVarP(&flags.build.tempDir, "temp_dir", "t", "", "Temp dir for build, defaults to os.TempDir() if not specified")
	agentEngineCmd.PersistentFlags().IntVar(&flags.agentEngine.serverPort, "server_port", 8080, "agentEngine server port")
	agentEngineCmd.PersistentFlags().StringVarP(&flags.source.entryPointPath, "entry_point_path", "e", "", "Path to an entry point (go 'main')")
	agentEngineCmd.PersistentFlags().StringVarP(&flags.source.sourceDir, "source_dir", "d", "", "Directory to archive, defaults to current working directory")
	agentEngineCmd.PersistentFlags().BoolVar(&flags.agentEngine.api, "api", true, "Enable API")
}

// computeFlags uses command line arguments to create a full config
func (f *deployAgentEngineFlags) computeFlags() error {
	return util.LogStartStop("Computing flags & preparing temp",
		func(p util.Printer) error {
			f.source.origEntryPointPath = flags.source.entryPointPath
			absp, err := filepath.Abs(flags.source.entryPointPath)
			if err != nil {
				return fmt.Errorf("cannot make an absolute path from '%v': %w", f.source.entryPointPath, err)
			}
			f.source.entryPointPath = absp

			if flags.build.tempDir == "" {
				flags.build.tempDir = os.TempDir()
			}
			absp, err = filepath.Abs(flags.build.tempDir)
			if err != nil {
				return fmt.Errorf("cannot make an absolute path from '%v': %w", f.build.tempDir, err)
			}
			f.build.tempDir, err = os.MkdirTemp(absp, "agentEngine_"+time.Now().Format("20060102_150405__")+"*")
			if err != nil {
				return fmt.Errorf("cannot create a temporary sub directory in '%v': %w", absp, err)
			}
			p("Using temp dir:", f.build.tempDir)

			// come up with a executable name based on entry point path
			dir, file := path.Split(f.source.entryPointPath)
			f.source.srcBasePath = dir
			f.source.entryPointPath = file
			if f.build.execPath == "" {
				exec, err := util.StripExtension(f.source.entryPointPath, ".go")
				if err != nil {
					return fmt.Errorf("cannot strip '.go' extension from entry point path '%v': %w", f.source.entryPointPath, err)
				}
				f.build.execFile = exec
				f.build.execPath = path.Join(f.build.tempDir, exec)
			}
			f.build.dockerfileBuildPath = path.Join(f.build.tempDir, "Dockerfile")
			f.build.archivePath = path.Join(f.build.tempDir, "archive.tgz")

			dateTimeString := time.Now().Format(time.RFC3339)
			f.agentEngine.displayName = f.agentEngine.name
			if f.agentEngine.displayName == "" {
				f.agentEngine.displayName = "ADK Agent: " + dateTimeString
			}

			return nil
		})
}

func (f *deployAgentEngineFlags) cleanTemp() error {
	return util.LogStartStop("Cleaning temp",
		func(p util.Printer) error {
			p("Clean temp starting with", f.build.tempDir)
			err := os.RemoveAll(f.build.tempDir)
			if err != nil {
				return fmt.Errorf("failed to clean temp directory %v: %w", f.build.tempDir, err)
			}
			return nil
		})
}

// prepareDockerfile creates a temporary Dockerfile which will be executed by agentEngine
func (f *deployAgentEngineFlags) prepareDockerfile() error {
	return util.LogStartStop("Preparing Dockerfile",
		func(p util.Printer) error {
			p("Writing:", f.build.dockerfileBuildPath)

			var b strings.Builder
			b.WriteString(`
FROM golang:1.25 as builder
WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o ` + f.build.execFile + ` ` + f.source.origEntryPointPath + `

FROM gcr.io/distroless/static-debian11

COPY --from=builder /app/` + f.build.execFile + `  /app/` + f.build.execFile + `
EXPOSE ` + strconv.Itoa(flags.agentEngine.serverPort) + `
# Command to run the executable when the container starts
CMD ["/app/` + f.build.execFile + `", "web", "-port", "` + strconv.Itoa(flags.agentEngine.serverPort) + `"`)

			if flags.agentEngine.api {
				b.WriteString(`, "api"`)
			}

			b.WriteString(`]`)
			return os.WriteFile(f.build.dockerfileBuildPath, []byte(b.String()), 0o600)
		})
}

// createArchive creates a tar archive containing the source code and Dockerfile
func (f *deployAgentEngineFlags) createArchive() error {
	return util.LogStartStop("Creating source archive",
		func(p util.Printer) error {
			workspaceRoot := f.source.sourceDir
			if workspaceRoot == "" {
				var err error
				workspaceRoot, err = os.Getwd()
				if err != nil {
					return fmt.Errorf("cannot get current working directory: %w", err)
				}
			}
			p("Creating:", f.build.archivePath)
			cmd := exec.Command("tar", "-czf", f.build.archivePath,
				"-C", workspaceRoot, "--exclude=.git", "--exclude=adkgo", ".",
				"-C", f.build.tempDir, "Dockerfile")
			return util.LogCommand(cmd, p)
		})
}

// gcloudDeployToAgentEngine invokes gcloud to deploy source on agentEngine
func (f *deployAgentEngineFlags) gcloudDeployToAgentEngine() error {
	return util.LogStartStop("Deploying to Agent Engine",
		func(p util.Printer) error {
			ctx := context.Background()
			parent := fmt.Sprintf("projects/%s/locations/%s", f.gcloud.projectName, f.gcloud.region)
			endpoint := fmt.Sprintf("%s-aiplatform.googleapis.com:443", f.gcloud.region)
			client, err := aiplatform.NewReasoningEngineClient(ctx, option.WithEndpoint(endpoint))
			if err != nil {
				return fmt.Errorf("cannot create ReasoningEngineClient: %w", err)
			}
			defer func() {
				if err := client.Close(); err != nil {
					p("Warning: failed to close ReasoningEngineClient: %v", err)
				}
			}()

			archiveContent, err := os.ReadFile(f.build.archivePath)
			if err != nil {
				return fmt.Errorf("cannot read archive file: %w", err)
			}

			req := &aiplatformpb.CreateReasoningEngineRequest{
				Parent: parent,
				ReasoningEngine: &aiplatformpb.ReasoningEngine{
					DisplayName: f.agentEngine.displayName,
					Spec: &aiplatformpb.ReasoningEngineSpec{
						DeploymentSource: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec_{
							SourceCodeSpec: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec{
								Source: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec_InlineSource_{
									InlineSource: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec_InlineSource{
										SourceArchive: archiveContent,
									},
								},
								LanguageSpec: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec_ImageSpec_{
									ImageSpec: &aiplatformpb.ReasoningEngineSpec_SourceCodeSpec_ImageSpec{},
								},
							},
						},
						AgentFramework: "google-adk",
						DeploymentSpec: &aiplatformpb.ReasoningEngineSpec_DeploymentSpec{
							Env: []*aiplatformpb.EnvVar{
								{Name: "GOOGLE_CLOUD_REGION", Value: f.gcloud.region},
								{Name: "NUM_WORKERS", Value: "1"},
								{Name: "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY", Value: "true"},
								{Name: "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", Value: "true"},
							},
							SecretEnv: []*aiplatformpb.SecretEnvVar{
								{Name: "GOOGLE_API_KEY", SecretRef: &aiplatformpb.SecretRef{Secret: "GOOGLE_API_KEY", Version: "latest"}},
							},
						},
					},
				},
			}
			p("Sending CreateReasoningEngine request...")
			op, err := client.CreateReasoningEngine(ctx, req)
			if err != nil {
				return fmt.Errorf("CreateReasoningEngine failed: %w", err)
			}

			p("Waiting for operation to complete...")
			re, err := op.Wait(ctx)
			if err != nil {
				return fmt.Errorf("operation failed: %w", err)
			}

			p("Deployed Reasoning Engine:", re.Name)
			p("Display Name:", re.DisplayName)

			return nil
		})
}

// deployOnagentEngine executes the sequence of actions preparing and deploying the agent to agentEngine
func (f *deployAgentEngineFlags) deployOnagentEngine() error {
	fmt.Println(flags)

	err := f.computeFlags()
	if err != nil {
		return err
	}
	err = f.prepareDockerfile()
	if err != nil {
		return err
	}
	err = f.createArchive()
	if err != nil {
		return err
	}
	err = f.gcloudDeployToAgentEngine()
	if err != nil {
		return err
	}
	err = f.cleanTemp()
	if err != nil {
		return err
	}

	return nil
}
