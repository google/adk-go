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

// Package cloudrun handles command line parameters and execution logic for cloudrun deployment.
package cloudrun

import (
	"fmt"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"google.golang.org/adk/v2/cmd/adkgo/internal/deploy"
	"google.golang.org/adk/v2/internal/cli/util"
)

type gCloudFlags struct {
	region      string
	projectName string
}

type triggerConfigFlags struct {
	maxRetries int
	baseDelay  time.Duration
	maxDelay   time.Duration
	maxRuns    int
}

type cloudRunServiceFlags struct {
	serviceName     string
	serverPort      int
	a2aAgentCardURL string
	a2a             bool // enable a2a or not
	api             bool // enable REST API or not
	debugAPI        bool // enable debug in REST API or not
	webui           bool // enable webui or not
	pubsub          bool // enable pubsub trigger or not
	pubsubTrigger   triggerConfigFlags
	eventarc        bool // enable eventarc trigger or not
	eventarcTrigger triggerConfigFlags
}

type localProxyFlags struct {
	port int
}

type buildFlags struct {
	tempDir             string
	execPath            string
	execFile            string
	dockerfileBuildPath string
}

type sourceFlags struct {
	srcBasePath    string
	entryPointPath string
}

type deployCloudRunFlags struct {
	gcloud   gCloudFlags
	cloudRun cloudRunServiceFlags
	proxy    localProxyFlags
	build    buildFlags
	source   sourceFlags
}

var flags deployCloudRunFlags

// cloudrunCmd represents the cloudrun command
var cloudrunCmd = &cobra.Command{
	Use:   "cloudrun",
	Short: "Deploys the application to cloudrun.",
	Long: `Deployment prepares a Dockerfile which is fed with locally compiled server executable containing Web UI static files.
	Service on Cloudrun is created using this information. 
	Local proxy adding authentication is started. 
	`,
	RunE: func(cmd *cobra.Command, args []string) error {
		return flags.deployOnCloudRun()
	},
}

// init creates flags and adds subcommand to parent
func init() {
	deploy.DeployCmd.AddCommand(cloudrunCmd)

	cloudrunCmd.PersistentFlags().StringVarP(&flags.gcloud.region, "region", "r", "", "GCP Region")
	cloudrunCmd.PersistentFlags().StringVarP(&flags.gcloud.projectName, "project_name", "p", "", "GCP Project Name")
	cloudrunCmd.PersistentFlags().StringVarP(&flags.cloudRun.serviceName, "service_name", "s", "", "Cloud Run Service name")
	cloudrunCmd.PersistentFlags().StringVarP(&flags.build.tempDir, "temp_dir", "t", "", "Temp dir for build, defaults to os.TempDir() if not specified")
	cloudrunCmd.PersistentFlags().IntVar(&flags.proxy.port, "proxy_port", 8081, "Local proxy port")
	cloudrunCmd.PersistentFlags().IntVar(&flags.cloudRun.serverPort, "server_port", 8080, "Cloudrun server port")
	cloudrunCmd.PersistentFlags().StringVarP(&flags.source.entryPointPath, "entry_point_path", "e", "", "Path to an entry point (go 'main')")
	cloudrunCmd.PersistentFlags().BoolVar(&flags.cloudRun.a2a, "a2a", true, "Enable A2A")
	cloudrunCmd.PersistentFlags().StringVarP(&flags.cloudRun.a2aAgentCardURL, "a2a_agent_url", "a", "http://127.0.0.1:8081", "A2A agent card URL as advertised in the public agent card")
	cloudrunCmd.PersistentFlags().BoolVar(&flags.cloudRun.api, "api", true, "Enable API")
	cloudrunCmd.PersistentFlags().BoolVar(&flags.cloudRun.debugAPI, "debug_api", false, "Enable Debug API - requires '--api'")
	cloudrunCmd.PersistentFlags().BoolVar(&flags.cloudRun.webui, "webui", true, "Enable Web UI")
	cloudrunCmd.PersistentFlags().BoolVar(&flags.cloudRun.pubsub, "pubsub", false, "Enable PubSub subrouter")
	cloudrunCmd.PersistentFlags().IntVar(&flags.cloudRun.pubsubTrigger.maxRetries, "pubsub_max_retries", 3, "Maximum retries for HTTP 429 errors from PubSub triggers")
	cloudrunCmd.PersistentFlags().DurationVar(&flags.cloudRun.pubsubTrigger.baseDelay, "pubsub_base_delay", 1*time.Second, "Base delay for PubSub trigger retry exponential backoff")
	cloudrunCmd.PersistentFlags().DurationVar(&flags.cloudRun.pubsubTrigger.maxDelay, "pubsub_max_delay", 10*time.Second, "Maximum delay for PubSub trigger retry exponential backoff")
	cloudrunCmd.PersistentFlags().IntVar(&flags.cloudRun.pubsubTrigger.maxRuns, "pubsub_max_concurrent_runs", 100, "Maximum concurrent PubSub trigger runs")
	cloudrunCmd.PersistentFlags().BoolVar(&flags.cloudRun.eventarc, "eventarc", false, "Enable Eventarc subrouter")
	cloudrunCmd.PersistentFlags().IntVar(&flags.cloudRun.eventarcTrigger.maxRetries, "eventarc_max_retries", 3, "Maximum retries for HTTP 429 errors from Eventarc triggers")
	cloudrunCmd.PersistentFlags().DurationVar(&flags.cloudRun.eventarcTrigger.baseDelay, "eventarc_base_delay", 1*time.Second, "Base delay for Eventarc trigger retry exponential backoff")
	cloudrunCmd.PersistentFlags().DurationVar(&flags.cloudRun.eventarcTrigger.maxDelay, "eventarc_max_delay", 10*time.Second, "Maximum delay for Eventarc trigger retry exponential backoff")
	cloudrunCmd.PersistentFlags().IntVar(&flags.cloudRun.eventarcTrigger.maxRuns, "eventarc_max_concurrent_runs", 100, "Maximum concurrent Eventarc trigger runs")
}

// computeFlags uses command line arguments to create a full config
func (f *deployCloudRunFlags) computeFlags() error {
	return util.LogStartStop("Computing flags & preparing temp",
		func(p util.Printer) error {
			if f.cloudRun.debugAPI && !f.cloudRun.api {
				return fmt.Errorf("cannot enable Debug API without having enabled API")
			}

			// Checked before the temp dir is created so a rejected value does
			// not leave one behind. It is checked whether or not --a2a is set:
			// the flag can be flipped later, and a value that can never be
			// emitted safely is worth reporting either way.
			if err := util.ValidateDockerfileSafe(f.cloudRun.a2aAgentCardURL, "--a2a_agent_url"); err != nil {
				return err
			}

			// The raw flag value, kept for the rejection message below: what
			// the check sees is a basename derived from this, which a user who
			// typed a whole path cannot find in their own command line.
			origEntryPointPath := f.source.entryPointPath

			absp, err := filepath.Abs(f.source.entryPointPath)
			if err != nil {
				return fmt.Errorf("cannot make an absolute path from '%v': %w", f.source.entryPointPath, err)
			}
			f.source.entryPointPath = absp

			// Deriving and checking the executable name happens before the temp
			// dir is created, for the same reason as the --a2a_agent_url check
			// above: everything here can fail, and cleanTemp only runs after a
			// fully successful deploy, so a rejected invocation would otherwise
			// leave an empty cloudrun_<timestamp>_* directory behind. Only
			// execPath below actually needs the temp dir.
			dir, file := path.Split(f.source.entryPointPath)
			f.source.srcBasePath = dir
			f.source.entryPointPath = file
			if f.build.execPath == "" {
				exec, err := util.StripExtension(f.source.entryPointPath, ".go")
				if err != nil {
					return fmt.Errorf("cannot strip '.go' extension from entry point path '%v': %w", f.source.entryPointPath, err)
				}
				f.build.execFile = exec
			}

			// execFile becomes both COPY operands, and COPY splits its operands
			// on whitespace, so a bare filename is not enough: it takes the
			// allowlist even though this Dockerfile has no RUN instruction.
			if err := util.ValidateShellArgSafe(f.build.execFile,
				fmt.Sprintf("executable name derived from --entry_point_path %q:", origEntryPointPath)); err != nil {
				return err
			}

			if f.build.tempDir == "" {
				f.build.tempDir = os.TempDir()
			}
			absp, err = filepath.Abs(f.build.tempDir)
			if err != nil {
				return fmt.Errorf("cannot make an absolute path from '%v': %w", f.build.tempDir, err)
			}
			f.build.tempDir, err = os.MkdirTemp(absp, "cloudrun_"+time.Now().Format("20060102_150405__")+"*")
			if err != nil {
				return fmt.Errorf("cannot create a temporary sub directory in '%v': %w", absp, err)
			}
			p("Using temp dir:", f.build.tempDir)

			if f.build.execPath == "" {
				f.build.execPath = path.Join(f.build.tempDir, f.build.execFile)
			}
			f.build.dockerfileBuildPath = path.Join(f.build.tempDir, "Dockerfile")

			return nil
		})
}

func (f *deployCloudRunFlags) cleanTemp() error {
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

// compileEntryPoint builds locally the server using flags and environment variables in order to be run in CloudRun containter
func (f *deployCloudRunFlags) compileEntryPoint() error {
	return util.LogStartStop("Compiling server",
		func(p util.Printer) error {
			p("Using", f.source.entryPointPath, "as entry point")
			// for help on ldflags you can run go build -ldflags="--help" ./examples/quickstart/main.go
			//    -s    disable symbol table
			//    -w    disable DWARF generation
			//   using those flags reduces the size of an executable
			cmd := exec.Command("go", "build", "-ldflags", "-s -w", "-o", f.build.execPath, f.source.entryPointPath)

			cmd.Dir = f.source.srcBasePath
			// build using staticallly linked libs, for linux/amd64
			cmd.Env = append(os.Environ(), "CGO_ENABLED=0", "GOOS=linux", "GOARCH=amd64")
			return util.LogCommand(cmd, p)
		})
}

// prepareDockerfile creates a temporary Dockerfile which will be executed by CloudRun
func (f *deployCloudRunFlags) prepareDockerfile() error {
	return util.LogStartStop("Preparing Dockerfile",
		func(p util.Printer) error {
			p("Writing:", f.build.dockerfileBuildPath)

			// Every value below is read off the receiver, not off the package
			// global. computeFlags validates the receiver, so a read of the
			// global here would let the guard and the line it guards come apart
			// for any caller that is not the cobra RunE.
			var b strings.Builder
			b.WriteString(`
FROM gcr.io/distroless/static-debian11

COPY ` + f.build.execFile + `  /app/` + f.build.execFile + `
EXPOSE ` + strconv.Itoa(f.cloudRun.serverPort) + `
# Command to run the executable when the container starts
CMD ["/app/` + f.build.execFile + `", "web", "-port", "` + strconv.Itoa(f.cloudRun.serverPort) + `"`)

			if f.cloudRun.api {
				b.WriteString(`, "api", "-webui_address", "127.0.0.1:` + strconv.Itoa(f.proxy.port) + `"`)

				if f.cloudRun.debugAPI {
					b.WriteString(`, "-include_debug_api"`)
				}
			}
			if f.cloudRun.a2a {
				b.WriteString(`, "a2a", "--a2a_agent_url", "` + f.cloudRun.a2aAgentCardURL + `"`)
			}
			if f.cloudRun.webui {
				b.WriteString(`, "webui", "--api_server_address", "http://127.0.0.1:` + strconv.Itoa(f.proxy.port) + `/api"`)
			}
			if f.cloudRun.pubsub {
				b.WriteString(`, "pubsub"`)
				fmt.Fprintf(&b, `, "--trigger_max_retries", "%d"`, f.cloudRun.pubsubTrigger.maxRetries)
				fmt.Fprintf(&b, `, "--trigger_base_delay", "%s"`, f.cloudRun.pubsubTrigger.baseDelay.String())
				fmt.Fprintf(&b, `, "--trigger_max_delay", "%s"`, f.cloudRun.pubsubTrigger.maxDelay.String())
				fmt.Fprintf(&b, `, "--trigger_max_concurrent_runs", "%d"`, f.cloudRun.pubsubTrigger.maxRuns)
			}
			if f.cloudRun.eventarc {
				b.WriteString(`, "eventarc"`)
				fmt.Fprintf(&b, `, "--trigger_max_retries", "%d"`, f.cloudRun.eventarcTrigger.maxRetries)
				fmt.Fprintf(&b, `, "--trigger_base_delay", "%s"`, f.cloudRun.eventarcTrigger.baseDelay.String())
				fmt.Fprintf(&b, `, "--trigger_max_delay", "%s"`, f.cloudRun.eventarcTrigger.maxDelay.String())
				fmt.Fprintf(&b, `, "--trigger_max_concurrent_runs", "%d"`, f.cloudRun.eventarcTrigger.maxRuns)
			}
			b.WriteString(`]`)
			return os.WriteFile(f.build.dockerfileBuildPath, []byte(b.String()), 0o600)
		})
}

// gcloudDeployToCloudRun invokes gcloud to deploy source on CloudRun
func (f *deployCloudRunFlags) gcloudDeployToCloudRun() error {
	return util.LogStartStop("Deploying to Cloud Run",
		func(p util.Printer) error {
			params := []string{
				"run", "deploy", f.cloudRun.serviceName,
				"--source", ".",
				"--set-secrets=GOOGLE_API_KEY=GOOGLE_API_KEY:latest",
				"--region", f.gcloud.region,
				"--project", f.gcloud.projectName,
				"--ingress", "all",
				"--no-allow-unauthenticated",
			}

			cmd := exec.Command("gcloud", params...)

			cmd.Dir = f.build.tempDir
			return util.LogCommand(cmd, p)
		})
}

// runGcloudProxy invokes gcloud to create a proxy which will add authentication headers to requests
func (f *deployCloudRunFlags) runGcloudProxy() error {
	return util.LogStartStop("Running local gcloud authenticating proxy",
		func(p util.Printer) error {
			targetWidth := 80

			p(strings.Repeat("-", targetWidth))
			p(util.CenterString("", targetWidth))
			p(util.CenterString("Running ADK Web UI on http://127.0.0.1:"+strconv.Itoa(f.proxy.port)+"/ui/    <-- open this", targetWidth))
			p(util.CenterString("ADK REST API on http://127.0.0.1:"+strconv.Itoa(f.proxy.port)+"/api/         ", targetWidth))
			p(util.CenterString("", targetWidth))
			p(util.CenterString("Press Ctrl-C to stop", targetWidth))
			p(util.CenterString("", targetWidth))
			p(strings.Repeat("-", targetWidth))

			cmd := exec.Command("gcloud", "run", "services", "proxy", f.cloudRun.serviceName, "--project", f.gcloud.projectName, "--port", strconv.Itoa(f.proxy.port), "--region", f.gcloud.region)
			return util.LogCommand(cmd, p)
		})
}

// deployOnCloudRun executes the sequence of actions preparing and deploying the agent to CloudRun. Then runs authenticating proxy to newly deployed service
func (f *deployCloudRunFlags) deployOnCloudRun() error {
	fmt.Println(flags)

	err := f.computeFlags()
	if err != nil {
		return err
	}
	err = f.compileEntryPoint()
	if err != nil {
		return err
	}
	err = f.prepareDockerfile()
	if err != nil {
		return err
	}
	err = f.gcloudDeployToCloudRun()
	if err != nil {
		return err
	}
	err = f.cleanTemp()
	if err != nil {
		return err
	}
	err = f.runGcloudProxy()
	if err != nil {
		return err
	}

	return nil
}
