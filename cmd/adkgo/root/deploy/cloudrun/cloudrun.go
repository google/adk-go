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
	"strconv"

	"github.com/spf13/cobra"
	"google.golang.org/adk/cmd/adkgo/root/deploy"
	"google.golang.org/adk/cmd/util"
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
	uiBuildDir          string
	uiDistDir           string
	execPath            string
	dockerfileBuildPath string
}

type sourceFlags struct {
	uiDir          string
	srcBasePath    string
	entryPointPath string
}

type webUIDeployFlags struct {
	backendUri string
}

type deployCloudRunFlags struct {
	gcloud   gCloudFlags
	cloudRun cloudRunServiceFlags
	proxy    localProxyFlags
	build    buildFlags
	source   sourceFlags
	webUI    webUIDeployFlags
}

var flags deployCloudRunFlags

// cloudrunCmd represents the cloudrun command
var cloudrunCmd = &cobra.Command{
	Use:   "cloudrun",
	Short: "Deploys the application to cloudrun.",
	Long: `Deployment prepares a Dockerfile which is fed with locally compiled server executable and Web UI static files.
	Service on Cloud run is created using this information. 
	Local proxy adding authentication is optionally started. 
	`,
	RunE: func(cmd *cobra.Command, args []string) error {
		return flags.deployOnCloudRun()
	},
}

func init() {
	deploy.DeployCmd.AddCommand(cloudrunCmd)

	cloudrunCmd.PersistentFlags().StringVarP(&flags.gcloud.region, "region", "r", "", "GCP Region")
	cloudrunCmd.PersistentFlags().StringVarP(&flags.gcloud.projectName, "projectName", "p", "", "GCP Project Name")
	cloudrunCmd.PersistentFlags().StringVarP(&flags.cloudRun.serviceName, "serviceName", "s", "", "Cloud Run Service name")
	cloudrunCmd.PersistentFlags().StringVarP(&flags.build.tempDir, "tempDir", "t", "", "Temp dir for build")
	cloudrunCmd.PersistentFlags().IntVar(&flags.proxy.port, "proxyPort", 8081, "Local proxy port")
	cloudrunCmd.PersistentFlags().IntVar(&flags.cloudRun.serverPort, "serverPort", 8080, "Cloudrun server port")
	cloudrunCmd.PersistentFlags().StringVarP(&flags.source.uiDir, "webUIDir", "a", "", "ADK Web UI base dir")
	cloudrunCmd.PersistentFlags().StringVarP(&flags.webUI.backendUri, "backendUri", "b", "", "ADK REST API uri")
	cloudrunCmd.PersistentFlags().StringVarP(&flags.source.entryPointPath, "entryPoint", "e", "", "Path to an entry point (go 'main')")
	cloudrunCmd.PersistentFlags().StringVarP(&flags.source.srcBasePath, "srcPath", "", "", "Path to an entry point (go 'main')")

}

func (f *deployCloudRunFlags) computeFlags() error {
	fmt.Println("Compute flags starting")
	f.build.uiBuildDir = path.Join(f.build.tempDir, "ui")
	f.build.uiDistDir = path.Join(f.source.uiDir, "/dist/agent_framework_web/browser")
	f.build.execPath = path.Join(f.build.tempDir, "server")
	f.build.dockerfileBuildPath = path.Join(f.build.tempDir, "Dockerfile")

	fmt.Println("Compute flags finished")
	return nil
}

func (f *deployCloudRunFlags) cleanTemp() error {
	err := util.LogStartStop("Cleaning temp",
		func(p util.Printer) error {
			p("Clean temp starting with", f.build.tempDir)
			err := os.RemoveAll(f.build.tempDir)
			if err != nil {
				return err
			}
			return os.MkdirAll(f.build.tempDir, os.ModeDir|0700)
		})
	return err
}

func (f *deployCloudRunFlags) makeDirs() error {
	err := util.LogStartStop("Make build dirs",
		func(p util.Printer) error {
			p("Making", f.build.uiBuildDir)
			return os.MkdirAll(f.build.uiBuildDir, os.ModeDir|0700)
		})
	return err
}

func (f *deployCloudRunFlags) setBackendForAdkWebUI() error {

	err := util.LogStartStop("Setting backend for Adk Web UI",
		func(p util.Printer) error {
			cmd := exec.Command("npm", "run", "inject-backend", "--backend="+f.webUI.backendUri)
			cmd.Dir = f.source.uiDir
			return util.LogCommand(cmd, p)
		})
	return err
}

func (f *deployCloudRunFlags) makeDistForAdkWebUI() error {
	err := util.LogStartStop("Making dist for Adk Web UI",
		func(p util.Printer) error {
			cmd := exec.Command("ng", "build", "--output-path="+f.build.uiBuildDir)

			cmd.Dir = f.source.uiDir
			return util.LogCommand(cmd, p)
		})
	return err
}

func (f *deployCloudRunFlags) compileEntryPoint() error {
	err := util.LogStartStop("Compiling server",
		func(p util.Printer) error {
			p("Using", f.source.entryPointPath, "as entry point")
			cmd := exec.Command("go", "build", "-o", f.build.execPath, f.source.entryPointPath)

			cmd.Dir = f.source.srcBasePath
			cmd.Env = append(os.Environ(), "CGO_ENABLED=0", "GOOS=linux")
			return util.LogCommand(cmd, p)
		})
	return err
}

func (f *deployCloudRunFlags) prepareDockerfile() error {
	err := util.LogStartStop("Preparing Dockerfile",
		func(p util.Printer) error {
			p("Writing:", f.build.dockerfileBuildPath)
			c := `
FROM golang:1.22-alpine AS builder

WORKDIR /app

COPY server  /app/server
COPY ui  /app/ui

FROM gcr.io/distroless/static-debian11

# Set the working directory
WORKDIR /app

# Copy the built executable from the builder stage
COPY --from=builder /app/server /app/server
COPY --from=builder /app/ui /app/ui

EXPOSE 8080

# Command to run the executable when the container starts
CMD ["/app/server", "--port", "8080", "--front_address", "` + f.webUI.backendUri + `"]
`
			return os.WriteFile(f.build.dockerfileBuildPath, []byte(c), 0600)
		})
	return err
}

func (f *deployCloudRunFlags) gcloudDeployToCloudRun() error {
	err := util.LogStartStop("Deploying to Cloud Run",
		func(p util.Printer) error {
			cmd := exec.Command("gcloud", "run", "deploy", f.cloudRun.serviceName,
				"--source", ".",
				"--set-secrets=GOOGLE_API_KEY=ADK_KEY:latest",
				"--region", f.gcloud.region,
				"--project", f.gcloud.projectName)

			cmd.Dir = f.build.tempDir
			return util.LogCommand(cmd, p)
		})
	return err
}

func (f *deployCloudRunFlags) runGcloudProxy() error {
	err := util.LogStartStop("Running local gcloud authenticating proxy",
		func(p util.Printer) error {
			cmd := exec.Command("gcloud", "run", "services", "proxy", f.cloudRun.serviceName, "--project", f.gcloud.projectName, "--port", strconv.Itoa(f.proxy.port))

			cmd.Dir = f.build.tempDir
			return util.LogCommand(cmd, p)
		})
	return err
}

func (f *deployCloudRunFlags) deployOnCloudRun() error {
	fmt.Println(flags)
	var err error

	err = f.computeFlags()
	if err != nil {
		return err
	}
	err = f.cleanTemp()
	if err != nil {
		return err
	}
	err = f.makeDirs()
	if err != nil {
		return err
	}
	err = f.setBackendForAdkWebUI()
	if err != nil {
		return err
	}
	err = f.makeDistForAdkWebUI()
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
