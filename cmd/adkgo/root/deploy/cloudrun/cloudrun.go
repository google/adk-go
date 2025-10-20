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

// package cloudrun handles command line parameters and execution logic for cloudrun deployment
package cloudrun

import (
	"fmt"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/spf13/cobra"
	"google.golang.org/adk/cmd/adkgo/root/deploy"
	"google.golang.org/adk/internal/cli/util"
)

type gCloudFlags struct {
	region      string
	projectName string
}

type cloudRunServiceFlags struct {
	serviceName string
	serverPort  int
}

type localProxyFlags struct {
	port int
}

type buildFlags struct {
	tempDir             string
	uiDistDir           string
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

func init() {
	deploy.DeployCmd.AddCommand(cloudrunCmd)

	cloudrunCmd.PersistentFlags().StringVarP(&flags.gcloud.region, "region", "r", "", "GCP Region")
	cloudrunCmd.PersistentFlags().StringVarP(&flags.gcloud.projectName, "project_name", "p", "", "GCP Project Name")
	cloudrunCmd.PersistentFlags().StringVarP(&flags.cloudRun.serviceName, "service_name", "s", "", "Cloud Run Service name")
	cloudrunCmd.PersistentFlags().StringVarP(&flags.build.tempDir, "temp_dir", "t", "", "Temp dir for build")
	cloudrunCmd.PersistentFlags().IntVar(&flags.proxy.port, "proxy_port", 8081, "Local proxy port")
	cloudrunCmd.PersistentFlags().IntVar(&flags.cloudRun.serverPort, "server_port", 8080, "Cloudrun server port")
	cloudrunCmd.PersistentFlags().StringVarP(&flags.source.entryPointPath, "entry_point_path", "e", "", "Path to an entry point (go 'main')")
}

func (f *deployCloudRunFlags) computeFlags() error {
	return util.LogStartStop("Computing flags",
		func(p util.Printer) error {
			absp, err := filepath.Abs(flags.source.entryPointPath)
			if err != nil {
				return fmt.Errorf("cannot make an absolute path from '%v': %w", f.source.entryPointPath, err)
			}
			f.source.entryPointPath = absp
			absp, err = filepath.Abs(flags.build.tempDir)
			if err != nil {
				return fmt.Errorf("cannot make an absolute path from '%v': %w", f.build.tempDir, err)
			}
			f.build.tempDir = absp

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

			f.build.uiDistDir = path.Join(f.build.tempDir, "webui_distr")
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
			err = os.MkdirAll(f.build.tempDir, os.ModeDir|0700)
			if err != nil {
				return fmt.Errorf("failed to create the target directory %v: %w", f.build.tempDir, err)
			}
			return nil
		})
}

func (f *deployCloudRunFlags) compileEntryPoint() error {
	return util.LogStartStop("Compiling server",
		func(p util.Printer) error {
			p("Using", f.source.entryPointPath, "as entry point")
			cmd := exec.Command("go", "build", "-o", f.build.execPath, f.source.entryPointPath)

			cmd.Dir = f.source.srcBasePath
			cmd.Env = append(os.Environ(), "CGO_ENABLED=0", "GOOS=linux")
			return util.LogCommand(cmd, p)
		})
}

func (f *deployCloudRunFlags) prepareDockerfile() error {
	return util.LogStartStop("Preparing Dockerfile",
		func(p util.Printer) error {
			p("Writing:", f.build.dockerfileBuildPath)
			c := `
FROM gcr.io/distroless/static-debian11

COPY ` + f.build.execFile + `  /app/` + f.build.execFile + `
EXPOSE ` + strconv.Itoa(flags.cloudRun.serverPort) + `
# Command to run the executable when the container starts
CMD ["/app/` + f.build.execFile + `", "--port", "` + strconv.Itoa(flags.cloudRun.serverPort) + `", "--webui_address", "127.0.0.1:` + strconv.Itoa(f.proxy.port) + `", "--api_server_address", "http://localhost:` + strconv.Itoa(f.proxy.port) + `/api"]
 `
			return os.WriteFile(f.build.dockerfileBuildPath, []byte(c), 0600)
		})
}

func (f *deployCloudRunFlags) gcloudDeployToCloudRun() error {
	return util.LogStartStop("Deploying to Cloud Run",
		func(p util.Printer) error {
			cmd := exec.Command("gcloud", "run", "deploy", f.cloudRun.serviceName,
				"--source", ".",
				"--set-secrets=GOOGLE_API_KEY=GOOGLE_API_KEY:latest",
				"--region", f.gcloud.region,
				"--project", f.gcloud.projectName,
				"--ingress", "all",
				"--no-allow-unauthenticated")

			cmd.Dir = f.build.tempDir
			return util.LogCommand(cmd, p)
		})
}

func (f *deployCloudRunFlags) runGcloudProxy() error {
	return util.LogStartStop("Running local gcloud authenticating proxy",
		func(p util.Printer) error {
			targetWidth := 80

			p(strings.Repeat("-", targetWidth))
			p(util.CenterString("", targetWidth))
			p(util.CenterString("Running ADK Web UI on http://localhost:"+strconv.Itoa(f.proxy.port)+"/ui/    <-- open this", targetWidth))
			p(util.CenterString("ADK REST API on http://localhost:"+strconv.Itoa(f.proxy.port)+"/api/         ", targetWidth))
			p(util.CenterString("", targetWidth))
			p(util.CenterString("Press Ctrl-C to stop", targetWidth))
			p(util.CenterString("", targetWidth))
			p(strings.Repeat("-", targetWidth))

			cmd := exec.Command("gcloud", "run", "services", "proxy", f.cloudRun.serviceName, "--project", f.gcloud.projectName, "--port", strconv.Itoa(f.proxy.port))

			cmd.Dir = f.build.tempDir
			return util.LogCommand(cmd, p)
		})
}

func (f *deployCloudRunFlags) deployOnCloudRun() error {
	fmt.Println(flags)

	err := f.computeFlags()
	if err != nil {
		return err
	}
	err = f.cleanTemp()
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
	err = f.runGcloudProxy()
	if err != nil {
		return err
	}

	return nil
}
